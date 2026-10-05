-- Stage 1 — session-scoped TEMP tables (t_patient, t_kontakt, t_lab_tag, t_diag).
-- Generated from practice_1/CKD_risk_model_dataframe.sql (CheCKD-Trainingsdatensatz, N. Conze).
-- Runtime copy for the Flower ClientApp (data/sql_loader.py); edit the practice_1 original and
-- re-sync this file. The params CTE literals are psycopg named binds (%(name)s).

/*
 CheCKD - Trainingsdatensatz, Stufe 1
 Einmal je Sitzung ausfuehren, unabhaengig von Stichtag und Horizont.
 Legt die temporaeren Tabellen t_patient, t_lab_tag, t_diag, t_kontakt an.
 Reihenfolge ist wichtig: t_patient zuerst (GFR-Geschlechtsabgleich),
 Sterbedatum-Schaetzung am Ende (braucht t_kontakt und t_lab_tag).
 Aufraeumen: DROP TABLE IF EXISTS t_lab_tag, t_diag, t_kontakt, t_patient;

 Developer comment:
-- Zur Optimierung der Laufzeit arbeitet die folgende Query mit temporary tables und Indexierung.
-- Eine Durchführung der Abfrage ohne temporary tables führt zu ineffizienten nested loops und Überlastung des Query planners
 */

-- ============================================================
--  1) Patienten (lebend und verstorben)
-- ============================================================
DROP TABLE IF EXISTS t_patient;
CREATE TEMP TABLE t_patient AS
SELECT p.patientid,
       p.geburtsdatum,
       upper(left(p.geschlecht, 1))   AS geschlecht,
       (vp.patientid IS NOT NULL)     AS verstorben,
       vp.sterbedatum::date           AS sterbedatum,
       NULL::date                     AS sterbedatum_eff,
       NULL::boolean                  AS sterbedatum_geschaetzt
FROM alle_patienten p
LEFT JOIN view.verstorbene_patienten vp ON vp.patientid = p.patientid
WHERE NOT EXISTS (SELECT 1 FROM test_patienten t WHERE t.patientid = p.patientid)
  AND p.geburtsdatum IS NOT NULL
  AND upper(left(p.geschlecht, 1)) IN ('M', 'W');

CREATE INDEX ON t_patient (patientid);
ANALYZE t_patient;

-- ============================================================
--  2) Kontakte
-- ============================================================
DROP TABLE IF EXISTS t_kontakt;
CREATE TEMP TABLE t_kontakt AS
SELECT patientid, datum::date AS datum
FROM arzt_patienten_kontakt
WHERE patientid IS NOT NULL;

CREATE INDEX ON t_kontakt (patientid, datum);
ANALYZE t_kontakt;


-- ============================================================
--  3) Labor: filtern, parsen, standardisieren, deduplizieren
-- ============================================================
DROP TABLE IF EXISTS t_lab_tag;
CREATE TEMP TABLE t_lab_tag AS
WITH
-- Ident-CTEs zuerst, damit lab_basis sie als Vorfilter nutzen kann
hba1c_idents       AS (
    SELECT testident, system_soll, prio
    FROM (VALUES
        ('HBA1C',        'prozent',  1),
        ('HBA1CA',       'mmol_mol', 1),
        ('HBAKAP',       'prozent',  1),
        ('HBA1CI',       'mmol_mol', 1),
        ('HBA1CM',       'mmol_mol', 1),
        ('4HBA1C',       'mmol_mol', 1),
        ('HBAC',         'prozent',  1),
        ('HBA1M',        'mmol_mol', 1),
        ('BHBA1',        'prozent',  1),
        ('HBA1C (IFCC)', 'mmol_mol', 1),
        ('A1CIFC',       'mmol_mol', 1),
        ('WA1CIFC',      'mmol_mol', 1),
        ('WHBA1',        'prozent',  1),
        ('HBA1CP',       'prozent',  1),
        ('HBAC-2',       'mmol_mol', 1),
        ('HBA1IF',       'mmol_mol', 1),
        ('HBACX',        'prozent',  1),
        ('HBAC2X',       'mmol_mol', 1),
        ('HHA1C',        'prozent',  1),
        ('HBA1CN',       'mmol_mol', 1),
        ('HBA1CIPR',     'prozent',  1),
        ('HBAC-I',       'mmol_mol', 1),
        -- prio 9: System nicht am Ident entscheidbar
        ('HB1C',         NULL,       9),   -- Einheit '%' in d001/e003/e035, Einheit 'mmol/mol' in e031
        ('HBA1',         'prozent',  9)    -- Einheit leer, Werte teils im Text, in e016 0 numerisch
    ) AS t(testident, system_soll, prio)),

gfr_idents         AS (
    SELECT testident, basis, formel, prio
    FROM (VALUES
        ('CKDEPI','kreatinin','CKD-EPI',1),
        ('CKD-EP','kreatinin','CKD-EPI',1),
        ('GFRCKD','kreatinin','CKD-EPI',1),
        ('GFR-CKD','kreatinin','CKD-EPI',1),
        ('GFRCK1','kreatinin','CKD-EPI',1),
        ('WGFRCKD','kreatinin','CKD-EPI',1),
        ('GFR_CKD_','kreatinin','CKD-EPI',1),
        ('MDRD','kreatinin','MDRD',2),
        ('MDRDM','kreatinin','MDRD',2),
        ('MDRDW','kreatinin','MDRD',2),
        ('GFR (MDRD-Formel)','kreatinin','MDRD',2),
        ('GFRBIS1','kreatinin','BIS1',3),
        ('GFR (BIS1-Formel)','kreatinin','BIS1',3),
        ('gfrswrz','kreatinin','Schwartz',9),
        ('GFR2','kreatinin','unbekannt',5),
        ('GFRK','kreatinin','unbekannt',5),
        ('GFRA','kreatinin','unbekannt',5),
        ('GFR-M','kreatinin','unbekannt',5),
        ('GFR1','kreatinin','unbekannt',5),
        ('GFRC','kreatinin','unbekannt',5),
        ('GFRT','kreatinin','unbekannt',5),
        ('eGFR','kreatinin','unbekannt',5),
        ('EGFR','kreatinin','unbekannt',5),
        ('EGFML','kreatinin','unbekannt',5),
        ('DMPD','kreatinin','unbekannt',9),
        ('gfr','kreatinin','unbekannt',5),
        ('ÜGFR','kreatinin','unbekannt',5),
        ('ÜGFRW','kreatinin','unbekannt',5),
        ('EGFRWH','kreatinin','stratifiziert',4),
        ('EGFRMH','kreatinin','stratifiziert',4),
        ('eGFRWH','kreatinin','stratifiziert',4),
        ('eGFRMH','kreatinin','stratifiziert',4),
        ('eGFRML','kreatinin','stratifiziert',4),
        ('eGFRWL','kreatinin','stratifiziert',4),
        ('GFRM','kreatinin','stratifiziert',4),
        ('GFRW','kreatinin','stratifiziert',4),
        ('GFREWH','kreatinin','stratifiziert',4),
        ('GFREMH','kreatinin','stratifiziert',4),
        ('GFREWN','kreatinin','stratifiziert',4),
        ('GFREMN','kreatinin','stratifiziert',4),
        ('GFR','cystatin_c','Cystatin C',8),
        ('GFRCYC','cystatin_c','Cystatin C',8),
        ('GFRCY','cystatin_c','Cystatin C',8),
        ('4CysGFR','cystatin_c','Cystatin C',8),
        ('GFRERW','cystatin_c','Cystatin C',8),
        ('CGFR','cystatin_c','Cystatin C',8),
        ('cgfr','cystatin_c','Cystatin C',8)
    ) AS t(testident, basis, formel, prio) ),

albuminurie_idents AS (
    SELECT testident, einheit_soll, verfahren, prio
    FROM (VALUES
        ('ALBU','mg/l','quantitativ',20),
        ('ALBTU','mg/l','quantitativ',20),
        ('ALBUU','mg/l','quantitativ',20),
        ('ALBU01','mg/l','quantitativ',20),
        ('ALBU-A','mg/l','quantitativ',20),
        ('ALBU-N','mg/l','quantitativ',20),
        ('ALBUC','mg/l','quantitativ',20),
        ('UALB','mg/l','quantitativ',20),
        ('MALUE','mg/l','quantitativ',20),
        ('ALB UP','mg/l','quantitativ',20),
        ('MIKROA','mg/l','quantitativ',20),
        ('MIKRAU','mg/l','quantitativ',20),
        ('UABS','mg/l','quantitativ',20),
        ('albu','mg/l','quantitativ',20),
        ('ALB/U1','mg/dl','quantitativ',20),   -- Faktor 10
        ('Micral','mg/l','semiquantitativ',99) -- Teststreifen, kein quantitativer Assay
    ) AS t(testident, einheit_soll, verfahren, prio) ),

uacr_idents        AS (SELECT testident, einheit_soll, prio
    FROM (VALUES
        ('ALBCRE','mg/g Krea',20),
        ('ALBTCR','mg/g Krea',20),
        ('ALBK-Q','mg/g Krea',20),
        ('L-QAlbKr','mg/g Krea',20),
        ('QAlbKrea','mg/g Krea',20),
        ('MALUEK','mg/g Krea',20),
        ('MIKRK2','mg/g Krea',2),
        ('MIKRKR','mg/g Krea',3),
        ('ALB/C','mg/g Krea',20),
        ('alqalbu','mg/g Krea',20),
        ('MIKRUN','mg/g Krea',99),    -- Umrechnung von qalbu, keine eigene Messung
        ('qalbu','mg/mmol Krea',1),   -- Primaerquelle: feinste Auflösung
        ('ALKRRU','mg/mmol Krea',20),
        ('MA/CRE','mg/mmol Krea',20)
    ) AS t(testident, einheit_soll, prio)),

lab_basis AS (
    SELECT
        l.patientid,
        l.laborbefundid,
        l.berichtsdatum::date AS befund_datum,
        l.testident,
        l.testbezeichnung,
        l.einheit,
        l.ergebniswert,
        regexp_replace(COALESCE(l.ergebnistexte, ''),
                       '^\s*Eingang:\s*\d{2}\.\d{2}\.\d{4}\s*\n?', '') AS text
    FROM laborwerte l
    WHERE l.berichtsdatum IS NOT NULL
      AND l.patientid    IS NOT NULL
      -- VORFILTER: nur Idente aus den vier Katalogen, bevor die Regex laeuft.
      -- Bewusst grob (Grossschreibung), der praezise Join folgt in den *_roh-CTEs.
      AND trim(upper(l.testident)) IN (
              SELECT trim(upper(testident)) FROM gfr_idents
              UNION SELECT trim(upper(testident)) FROM uacr_idents
              UNION SELECT trim(upper(testident)) FROM albuminurie_idents
              UNION SELECT testident FROM hba1c_idents)
),

hba1c_roh AS (
    SELECT
        b.patientid,
        b.laborbefundid,
        b.befund_datum AS hba1c_datum,
        b.testident,
        b.testbezeichnung,
        b.einheit,
        h.system_soll,
        h.prio,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN b.ergebniswert
            ELSE replace(substring(b.text from '^\s*[<>]?\s*([0-9]+(?:[.,][0-9]+)?)'),
                         ',', '.')::numeric
        END AS wert_roh,
        CASE
            WHEN b.ergebniswert IS NOT NULL   THEN NULL
            WHEN b.text ~ '^\s*<'             THEN 'unten'
            WHEN b.text ~ '^\s*>'             THEN 'oben'
        END AS zensiert
    FROM lab_basis b
    JOIN hba1c_idents h ON trim(upper(b.testident)) = h.testident
    WHERE (
            b.ergebniswert IS NOT NULL
            -- Text besteht im Wesentlichen nur aus der Zahl. Nachlauf-Token darf Ziffern
            -- enthalten (e024: "10.0HBA1C"), muss aber mit einem Buchstaben beginnen,
            -- sonst würde "5 2024" zu einem HbA1c von 5 %.
         OR b.text ~ '^\s*[<>]?\s*[0-9]+([.,][0-9]+)?\s*(%|[A-Za-z][A-Za-z0-9/%²]{0,11})?\s*$'
          )
),

hba1c_standard AS (
    SELECT
        patientid,
        laborbefundid,
        hba1c_datum,
        round(CASE WHEN system = 'prozent'  THEN wert_roh
                   ELSE wert_roh / 10.929 + 2.15 END, 2) AS hba1c_prozent,
        round(CASE WHEN system = 'mmol_mol' THEN wert_roh
                   ELSE (wert_roh - 2.15) * 10.929 END, 1) AS hba1c_mmol_mol,
        system,
        zensiert,
        testident,
        prio
    FROM (
        SELECT r.*,
            -- Reihenfolge: Bezeichnung > Einheit > Ident-Vorgabe > Wertebereich.
            -- Der Wertebereich entscheidet nur ausserhalb des Bandes 10-22.
            CASE
                WHEN r.testbezeichnung ~* 'ifcc'           THEN 'mmol_mol'
                WHEN r.testbezeichnung ~* 'ngsp|dcct|dccp' THEN 'prozent'
                WHEN r.einheit ~* 'mmol/\s*mol'            THEN 'mmol_mol'
                WHEN r.einheit ~  '%'                      THEN 'prozent'
                WHEN r.testbezeichnung ~ '%'               THEN 'prozent'
                WHEN r.system_soll IS NOT NULL             THEN r.system_soll
                WHEN r.wert_roh >= 22                      THEN 'mmol_mol'
                WHEN r.wert_roh <= 10                      THEN 'prozent'
            END AS system
        FROM hba1c_roh r
    ) x
    WHERE wert_roh IS NOT NULL
      AND system IS NOT NULL                                   -- Band 10-22 ohne Systemhinweis
      -- Plausibilitaetsfenster
      AND (   (system = 'prozent'  AND wert_roh BETWEEN 2.5 AND 20)
           OR (system = 'mmol_mol' AND wert_roh BETWEEN 5   AND 200) )
),

gfr_roh AS (
    SELECT
        b.patientid, b.laborbefundid,
        b.befund_datum AS gfr_datum,
        b.testident, b.einheit,
        i.basis, i.formel, i.prio,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN b.ergebniswert
            ELSE replace(substring(b.text from '^\s*[<>]?\s*([0-9]+(?:[.,][0-9]+)?)'),
                         ',', '.')::numeric
        END AS wert_roh,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN NULL
            WHEN b.text ~ '^\s*<'           THEN 'unten'
            WHEN b.text ~ '^\s*>'           THEN 'oben'     -- ">90" ist bei der GFR der Normalfall
        END AS zensiert
    FROM lab_basis b
    JOIN gfr_idents i ON b.testident = i.testident
    WHERE (
            b.ergebniswert IS NOT NULL
         OR b.text ~ '^\s*[<>]?\s*[0-9]+([.,][0-9]+)?\s*(%|[A-Za-z][A-Za-z0-9/%²]{0,11})?\s*$'
          )
),

gfr_standard AS (
    SELECT
        patientid, laborbefundid, gfr_datum,
        -- ml/min und ml/min/1,73m² gepoolt (Annahme, keine Umrechnung); ml/s * 60
        round(wert_roh * CASE WHEN einheit ~* 'ml/\s*s' THEN 60 ELSE 1 END, 1) AS gfr_wert,
        zensiert, basis, formel, testident, prio
    FROM gfr_roh
    WHERE wert_roh IS NOT NULL
      AND wert_roh * CASE WHEN einheit ~* 'ml/\s*s' THEN 60 ELSE 1 END BETWEEN 1 AND 200
),


albuminurie_roh AS (
    SELECT
        b.patientid, b.laborbefundid,
        b.befund_datum AS albuminurie_datum,
        b.testident, b.einheit,
        i.einheit_soll, i.verfahren, i.prio,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN b.ergebniswert
            ELSE replace(substring(b.text from '^\s*[<>]?\s*([0-9]+(?:[.,][0-9]+)?)'),
                         ',', '.')::numeric
        END AS wert_roh,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN NULL
            WHEN b.text ~ '^\s*<'           THEN 'unten'
            WHEN b.text ~ '^\s*>'           THEN 'oben'
        END AS zensiert
    FROM lab_basis b
    JOIN albuminurie_idents i ON b.testident = i.testident
    WHERE (
            b.ergebniswert IS NOT NULL
         OR b.text ~ '^\s*[<>]?\s*[0-9]+([.,][0-9]+)?\s*(%|[A-Za-z][A-Za-z0-9/%²]{0,11})?\s*$'
          )
),

albuminurie_standard AS (
    SELECT
        patientid, laborbefundid, albuminurie_datum,
        round(wert_roh * CASE WHEN einheit_soll = 'mg/dl' THEN 10 ELSE 1 END, 1) AS albuminurie_wert,
        zensiert, verfahren, testident, prio
    FROM albuminurie_roh
    WHERE wert_roh IS NOT NULL
      AND wert_roh * CASE WHEN einheit_soll = 'mg/dl' THEN 10 ELSE 1 END BETWEEN 0.1 AND 20000
),


uacr_roh AS (
    SELECT
        b.patientid, b.laborbefundid,
        b.befund_datum AS uacr_datum,
        b.testident, b.einheit,
        i.einheit_soll, i.prio,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN b.ergebniswert
            ELSE replace(substring(b.text from '^\s*[<>]?\s*([0-9]+(?:[.,][0-9]+)?)'),
                         ',', '.')::numeric
        END AS wert_roh,
        CASE
            WHEN b.ergebniswert IS NOT NULL THEN NULL
            WHEN b.text ~ '^\s*<'           THEN 'unten'
            WHEN b.text ~ '^\s*>'           THEN 'oben'
        END AS zensiert
    FROM lab_basis b
    JOIN uacr_idents i ON b.testident = i.testident
    WHERE (
            b.ergebniswert IS NOT NULL
         OR b.text ~ '^\s*[<>]?\s*[0-9]+([.,][0-9]+)?\s*(%|[A-Za-z][A-Za-z0-9/%²]{0,11})?\s*$'
          )
),

uacr_standard AS (
    SELECT
        patientid, laborbefundid, uacr_datum,
        round(wert_roh * CASE WHEN einheit_soll = 'mg/mmol Krea' THEN 8.84 ELSE 1 END, 1) AS uacr_wert,
        CASE
            WHEN wert_roh * CASE WHEN einheit_soll = 'mg/mmol Krea' THEN 8.84 ELSE 1 END <   30 THEN 'A1'
            WHEN wert_roh * CASE WHEN einheit_soll = 'mg/mmol Krea' THEN 8.84 ELSE 1 END <= 300 THEN 'A2'
            ELSE 'A3'
        END AS kdigo_kategorie,
        zensiert, testident, prio
    FROM uacr_roh
    WHERE wert_roh IS NOT NULL
      AND wert_roh * CASE WHEN einheit_soll = 'mg/mmol Krea' THEN 8.84 ELSE 1 END BETWEEN 0.1 AND 20000
),

lab_long AS (
    -- Geschlechtsabgleich jetzt gegen die Patiententabelle statt gegen die Kohorte,
    -- damit diese Stufe nicht vom Stichtag abhaengt
    SELECT g.patientid,
           'gfr'::text AS parameter,
           g.gfr_datum AS datum,
           g.gfr_wert::numeric AS wert,
           (g.zensiert IS NOT NULL) AS zensiert, g.prio
    FROM gfr_standard g
    JOIN t_patient p ON p.patientid = g.patientid
    WHERE g.basis = 'kreatinin'
      AND g.formel <> 'Schwartz'
      AND g.testident <> 'DMPD'
      AND (   g.formel <> 'stratifiziert'
           OR substring(upper(g.testident) from '^E?GFRE?([MW])') = p.geschlecht)
    UNION ALL
    SELECT patientid,
           'uacr_log',
           uacr_datum,
           ln(greatest(uacr_wert, 0.1)),
           (zensiert IS NOT NULL), prio
    FROM uacr_standard
    UNION ALL
    SELECT patientid, 'hba1c', hba1c_datum, hba1c_prozent,
           (zensiert IS NOT NULL), prio
    FROM hba1c_standard
    UNION ALL
    SELECT patientid,
           'ualb_log',
           albuminurie_datum,
           ln(greatest(albuminurie_wert, 0.1)),
           (zensiert IS NOT NULL), prio
    FROM albuminurie_standard
    WHERE verfahren = 'quantitativ'
)
SELECT DISTINCT ON (patientid, parameter, datum)
       patientid,
       parameter,
       datum,
       wert,
       zensiert
FROM lab_long
ORDER BY patientid, parameter, datum, prio, wert;

CREATE INDEX ON t_lab_tag (parameter, patientid, datum);
ANALYZE t_lab_tag;


-- ============================================================
--  4) Diagnosen: nur benoetigte Spalten
-- ============================================================
DROP TABLE IF EXISTS t_diag;
CREATE TEMP TABLE t_diag AS
SELECT patientid,
       icd_code,
       diagnosedatum::date AS diagnosedatum,
       abgesetzt,
       ist_dauerdiagnose,
       ist_anamnestische_dauerdiagnose
FROM diagnosen
WHERE typ = 'G' AND patientid IS NOT NULL --hier noch "abgesetzte" Diagnosen ausschließen;

CREATE INDEX ON t_diag (patientid, diagnosedatum);
ANALYZE t_diag;


-- ============================================================
--  5) Sterbedatum: exakt, sonst letzte Aktivitaet (Kontakt oder Labor)
--     als Untergrenze. Diagnosen bewusst nicht (Nacherfassung nach Tod).
-- ============================================================
UPDATE t_patient p
SET sterbedatum_eff        = COALESCE(p.sterbedatum, a.letzte_aktivitaet),
    sterbedatum_geschaetzt = (p.sterbedatum IS NULL)
FROM (SELECT patientid, max(datum) AS letzte_aktivitaet
      FROM (SELECT patientid, datum FROM t_kontakt
            UNION ALL
            SELECT patientid, datum FROM t_lab_tag) x
      GROUP BY patientid) a
WHERE a.patientid = p.patientid
  AND p.verstorben;
ANALYZE t_patient;



