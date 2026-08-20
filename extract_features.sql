-- ============================================================
-- CKD LANDMARK DATAFRAME
-- Docport Forschungspipeline — ML Feature Extraction
-- Flower.ai Federated Learning Baseline
--
-- Stichtag:  CURRENT_DATE - 365 Tage (rollierend, kein hardcoded Datum)
-- Zielfenster: [t0, t0 + 365 Tage) — Inzidenz im Folgejahr
--
-- Einschluss:
--   - Lebende Patienten, 18–95 Jahre, Geschlecht dokumentiert
--   - Mindestens 1 Kontakt ≥ 52 Wochen vor Stichtag
--   - Keine bekannte CKD (N18.*) vor Stichtag
--   - Keine Testpatienten
--
-- Noch fehlende Exklusionskriterien (TODO):
--   - Dialyse-Patienten (OPS 8-854.* oder ICD Z99.2)
--   - Nierentransplantierte (ICD Z94.0)
--   - Vertretungsscheine (Scheinart-Filter gegen kv_scheine.scheinart)
--
-- Label (Inzidenz, KDIGO-angelehnt):
--   ckd_incident = 1 wenn im Zielfenster:
--     (A) gesicherte ICD-Diagnose N18.* (typ='G', nicht anamnestisch) ODER
--     (B) eGFR-Kriterium: aktueller eGFR ≤ 60 UND Vorgänger ≤ 60
--         (Abstand ≥ 90 Tage) UND alle Werte im 90-Tage-Fenster ≤ 60
--   Hinweis: Kriterium (B) approximiert KDIGO-Persistenzkriterium
--
-- Features:
--   alter_jahre                  Alter zum Stichtag
--   geschlecht                   1=M, 0=W
--   dm / aht / cvd               Vordiagnosen-Flags (gesichert, vor t0)
--   tage_seit_*_diagnose         Zeit seit Erstdiagnose (NULL wenn kein Flag)
--   egfr_letzter                 Letzter eGFR-Wert vor t0
--   egfr_mittelwert_3            Mittelwert letzte 3 eGFR-Werte (wenn >= 2 vorhanden)
--   hba1c_letzter                Letzter HbA1c vor t0
--   hba1c_mittelwert_3           Mittelwert letzte 3 HbA1c-Werte (wenn >= 2 vorhanden)
--
-- CVD-Abdeckung (aktuell):
--   I20-I25 (KHK/ACS), I50 (Herzinsuffizienz), I63-I66 (Schlaganfall/TIA)
--   Noch fehlend: I47-I49 (Arrhythmien), I60-I62 (Hirnblutung),
--                 I70 (Atherosklerose), I73.9 (PAVK)
--
-- Diabetes-Abdeckung:
--   E10 (Typ 1), E11 (Typ 2), E13 (sonstig spezifiziert), E14 (nicht naeher bezeichnet)
--
-- Export pro Praxis via psql:
--   psql -d tomedo -U postgres \
--     -c "\COPY (<query>) TO '/tmp/ckd_landmark_<praxis_id>.csv' WITH CSV HEADER"
-- ============================================================

WITH

-- ── 0. Stichtag ────────────────────────────────────────────────────────────
-- Rollierend: immer 365 Tage vor heute.
-- Zielfenster: [stichtag, stichtag + 365 Tage)
-- Fuer reproduzierbare Runs: CURRENT_DATE durch festes Datum ersetzen,
-- z.B. '2024-01-01'::TIMESTAMP
stichtag AS (
    SELECT (CURRENT_DATE - INTERVAL '365 days')::TIMESTAMP AS val
),

-- ── 1. Basispopulation ─────────────────────────────────────────────────────
-- Erstkontakt wird ueber drei Quellen ermittelt (KV-Scheine, HZV-Teilnahmen,
-- Privatpatienten) und als fruehestes Datum zusammengefuehrt.
-- HAVING-Klausel stellt sicher: Kontakt muss >= 52 Wochen vor t0 liegen
-- (Mindest-Beobachtungszeit fuer Feature-Stabilitaet).
basis_patienten AS (
    SELECT
        lp.patientid,
        lp.geburtsdatum,
        lp.geschlecht,
        EXTRACT(YEAR FROM AGE(
            (SELECT val FROM stichtag),
            lp.geburtsdatum
        ))::INT AS alter_jahre,
        MIN(e.datum) AS erstkontakt_datum
    FROM lebende_patienten lp
    LEFT JOIN (
        SELECT patientid, MIN(datum)::TIMESTAMP AS datum
        FROM (
            -- KV-Scheine: nur Jahresangabe verfuegbar → ersten Tag des Jahres
            SELECT patientid, MIN(TO_TIMESTAMP(jahr::TEXT, 'YYYY'))::TIMESTAMP AS datum
            FROM kv_scheine
            GROUP BY patientid

            UNION ALL

            -- HZV-Teilnahmen: Status 2 = aktiv, 3 = beendet
            SELECT patientid, MIN(beginnteilnahme) AS datum
            FROM hzv_teilnahmen
            WHERE statusteilnahme IN (2, 3)
            GROUP BY patientid

            UNION ALL

            -- Privatpatienten: letztes Abrechnungsdatum als Proxy
            SELECT patientid, MIN(datum_der_letzten_abrechnung) AS datum
            FROM privatpatienten
            GROUP BY patientid
        ) src
        GROUP BY patientid
    ) e ON e.patientid = lp.patientid
    WHERE
        -- Testpatienten ausschliessen
        NOT EXISTS (
            SELECT 1 FROM test_patienten tp WHERE tp.patientid = lp.patientid
        )
        AND lp.geburtsdatum IS NOT NULL
        AND lp.geschlecht IS NOT NULL
        -- Altersrange: 18-95 Jahre zum Stichtag
        AND EXTRACT(YEAR FROM AGE(
            (SELECT val FROM stichtag), lp.geburtsdatum
        )) BETWEEN 18 AND 95
    GROUP BY lp.patientid, lp.geburtsdatum, lp.geschlecht
    -- Mindest-Beobachtungszeit: Erstkontakt vor >= 52 Wochen
    HAVING
        MIN(e.datum) IS NOT NULL
        AND MIN(e.datum) <= (SELECT val FROM stichtag) - INTERVAL '52 weeks'
),

-- ── 2. Exklusion: vorbekannte CKD ─────────────────────────────────────────
-- Nur gesicherte Diagnosen (typ='G'), keine anamnestischen Dauerdiagnosen.
-- Alle N18.*-Subkodes werden ausgeschlossen (Stadium 1-5, G und D).
-- TODO: Zusaetzlich ausschliessen:
--   - Z99.2 / OPS 8-854.* (Dialyse)
--   - Z94.0 (Nierentransplantation)
--   - Vertretungsscheine (kv_scheine.scheinart = Vertretungsschein-Code)
vorbekannte_ckd AS (
    SELECT DISTINCT d.patientid
    FROM diagnosen d
    WHERE d.icd_code ^@ 'N18'
      AND d.typ = 'G'
      AND NOT d.ist_anamnestische_dauerdiagnose
      AND d.diagnosedatum < (SELECT val FROM stichtag)
),

eingeschlossene_patienten AS (
    SELECT bp.*
    FROM basis_patienten bp
    WHERE NOT EXISTS (
        SELECT 1 FROM vorbekannte_ckd ckd WHERE ckd.patientid = bp.patientid
    )
),

-- ── 3. Label A: ICD-Inzidenz im Zielfenster ───────────────────────────────
-- Gesicherte N18.*-Erstdiagnose im Jahr nach Stichtag.
-- Nur gesicherte Diagnosen (typ='G'), keine Anamnese.
-- Patienten ohne N18 vor t0 wurden bereits in Schritt 2 selektiert.
label_icd AS (
    SELECT DISTINCT d.patientid
    FROM diagnosen d
    WHERE d.icd_code ^@ 'N18'
      AND d.typ = 'G'
      AND NOT d.ist_anamnestische_dauerdiagnose
      AND d.diagnosedatum >= (SELECT val FROM stichtag)
      AND d.diagnosedatum <  (SELECT val FROM stichtag) + INTERVAL '365 days'
      AND EXISTS (
          SELECT 1 FROM eingeschlossene_patienten ep WHERE ep.patientid = d.patientid
      )
),

-- ── 4. Label B: eGFR-Kriterium ────────────────────────────────────────────
-- Breite testident-Liste deckt Laborgerate-Varianz ueber alle 12 Praxen ab.
-- Prafix-Matching (^@) fuer Gerate-spezifische Suffixe.
-- Zeitfenster: alle Messungen bis Ende Zielfenster (t0 + 365 Tage).
egfr AS (
    SELECT
        l.patientid,
        l.berichtsdatum AS egfr_datum,
        l.ergebniswert  AS egfr_wert
    FROM laborwerte l
    WHERE l.ergebniswert IS NOT NULL
      AND (
          l.testident = ANY (ARRAY[
              'CKDEPI', 'GFRCKD', 'GFR-CKD', 'CKD-EP',
              'GFRK', 'GFR2', 'GFRA', 'GFR-M', 'GFRBIS1',
              'GFR1', 'GFR (MDRD-Formel)', 'GFR (BIS1-Formel)',
              'GFRCYC', 'GFRCY', 'MDRD',
              'eGFRWH', 'eGFRMH', 'eGFRWL', 'eGFRML',
              'GFRM', 'GFRW', 'GFMDRD',
              'GFR', 'GFRT', 'GFRWX', 'GFRMX', 'CGFR'
          ])
          OR l.testident ^@ ANY (ARRAY['GFR', 'GLOMFI', 'gfr', 'EGFR', 'eGFR'])
          OR l.testbezeichnung ^@ 'GLOM FILT R'
      )
      AND l.berichtsdatum < (SELECT val FROM stichtag) + INTERVAL '365 days'
      AND EXISTS (
          SELECT 1 FROM eingeschlossene_patienten ep WHERE ep.patientid = l.patientid
      )
),

-- Nur Messungen im Zielfenster [t0, t0+365)
egfr_im_fenster AS (
    SELECT e.patientid, egfr_datum, egfr_wert
    FROM egfr e
    WHERE egfr_datum >= (SELECT val FROM stichtag)
),

-- Fenster-Indikatoren pro Messung:
--   egfr_wert_vor_min_3_monaten: letzter Wert >= 90 Tage vor aktueller Messung
--     KDIGO-Persistenzkriterium: CKD nur wenn ueber >= 3 Monate persistiert.
--     JOIN gegen egfr (nicht egfr_im_fenster), damit auch Messungen vor t0
--     als Vorgaenger herangezogen werden koennen.
--   egfr_unter_60_im_fenster: Anzahl Werte <= 60 im rollierenden 90-Tage-Fenster.
--     Verglichen mit anzahl_im_fenster: kein einziger Wert > 60 im Fenster.
egfr_werte_mit_fenster_indikatoren AS (
    SELECT
        gfr.patientid,
        gfr.egfr_datum,
        gfr.egfr_wert,
        prev.egfr_wert AS egfr_wert_vor_min_3_monaten,
        COUNT(*) FILTER (WHERE gfr.egfr_wert <= 60) OVER w AS egfr_unter_60_im_fenster,
        COUNT(*)                                     OVER w AS anzahl_im_fenster
    FROM egfr_im_fenster gfr
    LEFT JOIN LATERAL (
        SELECT gfr2.egfr_wert
        FROM egfr gfr2
        WHERE gfr2.patientid = gfr.patientid
          AND gfr2.egfr_datum <= gfr.egfr_datum - INTERVAL '90 days'
        ORDER BY gfr2.egfr_datum DESC
        LIMIT 1
    ) prev ON TRUE
    WINDOW w AS (
        PARTITION BY gfr.patientid
        ORDER BY gfr.egfr_datum
        RANGE BETWEEN INTERVAL '90 days' PRECEDING AND CURRENT ROW
    )
),

-- Ein Patient gilt als eGFR-positiv wenn alle drei Bedingungen erfuellt:
--   (1) aktuelle Messung <= 60
--   (2) Vorgaenger >= 90 Tage davor ebenfalls <= 60 (KDIGO-Persistenz)
--   (3) alle Werte im 90-Tage-Fenster <= 60 (kein spontaner Ausreisser)
label_gfr AS (
    SELECT DISTINCT patientid
    FROM egfr_werte_mit_fenster_indikatoren
    WHERE egfr_wert <= 60
      AND COALESCE(egfr_wert_vor_min_3_monaten <= 60, FALSE)
      AND egfr_unter_60_im_fenster = anzahl_im_fenster
),

-- ── 5. Vordiagnosen-Features ───────────────────────────────────────────────
-- Alle gesicherten Dauerdiagnosen (typ='G') vor Stichtag.
-- Erstdiagnose-Datum fuer Zeitfeature (tage_seit_*).
-- CVD-Abdeckung: KHK (I20-I25), Herzinsuffizienz (I50), Schlaganfall (I63-I66)
-- TODO ergaenzen: I47-I49 (Arrhythmien), I60-I62 (Hirnblutung),
--               I70 (generalisierte Atherosklerose), I73.9 (PAVK)
diagnose_features AS (
    SELECT
        ep.patientid,

        -- Diabetes mellitus: E10 (Typ 1), E11 (Typ 2),
        --   E13 (sonst. spez.), E14 (nicht naeher bezeichnet)
        BOOL_OR(d.icd_code ^@ ANY (ARRAY['E10','E11','E13','E14'])) AS dm,
        MIN(CASE WHEN d.icd_code ^@ ANY (ARRAY['E10','E11','E13','E14'])
            THEN d.diagnosedatum END) AS dm_erstes_datum,

        -- Arterielle Hypertonie: I10 (essenziell) + hypertensive Organschaeden
        BOOL_OR(d.icd_code ^@ ANY (ARRAY['I10','I11','I12','I13','I15'])) AS aht,
        MIN(CASE WHEN d.icd_code ^@ ANY (ARRAY['I10','I11','I12','I13','I15'])
            THEN d.diagnosedatum END) AS aht_erstes_datum,

        -- Kardiovaskulaere Erkrankungen (CVD) — aktuell unvollstaendig (s.o. TODO)
        BOOL_OR(d.icd_code ^@ ANY (
            ARRAY['I20','I21','I22','I23','I24','I25',  -- KHK / ACS
                  'I50',                                 -- Herzinsuffizienz
                  'I63','I64','I65','I66']               -- Schlaganfall / TIA
        )) AS cvd,
        MIN(CASE WHEN d.icd_code ^@ ANY (
            ARRAY['I20','I21','I22','I23','I24','I25','I50','I63','I64','I65','I66']
        ) THEN d.diagnosedatum END) AS cvd_erstes_datum

    FROM eingeschlossene_patienten ep
    LEFT JOIN diagnosen d
        ON  d.patientid = ep.patientid
        AND d.typ = 'G'
        AND NOT d.ist_anamnestische_dauerdiagnose
        AND d.diagnosedatum < (SELECT val FROM stichtag)
    GROUP BY ep.patientid
),

-- ── 6. Labor-Features: eGFR vor Stichtag ──────────────────────────────────
-- ROW_NUMBER DESC: rn=1 ist letzter Wert vor t0.
-- egfr_mittelwert_3: Mittelwert der letzten 3 Messungen als Stabilitaetsindikator
--   (NULL wenn < 2 Werte vorhanden — im Python-Client mit Median imputieren).
egfr_vor_stichtag AS (
    SELECT
        e.patientid,
        egfr_wert,
        ROW_NUMBER() OVER (PARTITION BY e.patientid ORDER BY egfr_datum DESC) AS rn
    FROM egfr e
    WHERE egfr_datum < (SELECT val FROM stichtag)
),

-- ── 7. Labor-Features: HbA1c vor Stichtag ─────────────────────────────────
-- Analoges Pattern wie eGFR.
-- Prafix-Liste deckt praxisuebliche Kuerzel (HBA1C, HBAKAP) ab.
-- Fuer Nicht-Diabetiker typischerweise NULL → kein Fehler, sondern Information.
hba1c_vor_stichtag AS (
    SELECT
        l.patientid,
        l.ergebniswert AS hba1c_wert,
        ROW_NUMBER() OVER (PARTITION BY l.patientid ORDER BY l.berichtsdatum DESC) AS rn
    FROM laborwerte l
    WHERE l.ergebniswert IS NOT NULL
      AND l.testident ^@ ANY (ARRAY['HBA1C', 'HBAKAP'])
      AND l.berichtsdatum < (SELECT val FROM stichtag)
      AND EXISTS (
          SELECT 1 FROM eingeschlossene_patienten ep WHERE ep.patientid = l.patientid
      )
),

-- ── 8. Labor-Feature Assembly ──────────────────────────────────────────────
-- Aggregation auf Patientenebene.
-- NULL-Werte signalisieren: kein Labor in Tomedo dokumentiert.
-- Empfehlung fuer Client-seitiges Preprocessing: SimpleImputer(strategy='median')
-- oder Missing-Indicator als separates Binary-Feature.
labor_features AS (
    SELECT
        ep.patientid,
        MAX(egfr.egfr_wert)  FILTER (WHERE egfr.rn = 1)          AS egfr_letzter,
        CASE WHEN COUNT(egfr.egfr_wert) FILTER (WHERE egfr.rn <= 3) >= 2
            THEN AVG(egfr.egfr_wert) FILTER (WHERE egfr.rn <= 3)
        END                                                        AS egfr_mittelwert_3,
        MAX(hba1c.hba1c_wert) FILTER (WHERE hba1c.rn = 1)         AS hba1c_letzter,
        CASE WHEN COUNT(hba1c.hba1c_wert) FILTER (WHERE hba1c.rn <= 3) >= 2
            THEN AVG(hba1c.hba1c_wert) FILTER (WHERE hba1c.rn <= 3)
        END                                                        AS hba1c_mittelwert_3
    FROM eingeschlossene_patienten ep
    LEFT JOIN egfr_vor_stichtag  egfr  ON egfr.patientid  = ep.patientid
    LEFT JOIN hba1c_vor_stichtag hba1c ON hba1c.patientid = ep.patientid
    GROUP BY ep.patientid
)

-- ── 9. Finaler Export ─────────────────────────────────────────────────────
-- Eine Zeile pro eingeschlossenem Patienten.
-- t0: Stichtag dieser Extraktion (fuer Reproducibility im CSV mitfuehren).
-- ckd_incident: zusammengesetztes Label (ICD ODER eGFR-Kriterium).
-- tage_seit_*: NULL wenn Vorkrankung nicht vorhanden (nicht 0!).
--   Im Modell: 0-Imputation ODER separates Flag dx_* bereits vorhanden.
SELECT
    (SELECT val FROM stichtag)                                     AS t0,
    -- ep.patientid ENTFERNT (Datenschutz): ein direkter Identifikator im Export macht die
    -- Praxis-CSV zu personenbezogenen Daten statt zu einem pseudonymen Extrakt. Nichts
    -- stromabwaerts nutzt ihn - data/loader.py selektiert nur FEATURE_COLS. Falls T4.2 je
    -- einen stabilen Join-Key ueber Extraktionen hinweg braucht: praxis-spezifischer
    -- gesalzener Hash, bewusst entschieden - nicht die rohe ID zurueckholen.
    ep.alter_jahre,
    CASE ep.geschlecht WHEN 'M' THEN 1 WHEN 'W' THEN 0 ELSE NULL END AS geschlecht,

    -- Label: 1 wenn CKD-Inzidenz im Zielfenster via ICD oder eGFR
    CASE WHEN label_icd.patientid IS NOT NULL
           OR label_gfr.patientid IS NOT NULL
         THEN 1 ELSE 0
    END AS ckd_incident,

    -- Vordiagnosen-Flags (0/1)
    COALESCE(df.dm,  FALSE)::INT AS dm,
    COALESCE(df.aht, FALSE)::INT AS aht,
    COALESCE(df.cvd, FALSE)::INT AS cvd,

    -- Zeit seit Erstdiagnose in Tagen (NULL = Diagnose nicht vorhanden)
    CASE WHEN df.dm  THEN EXTRACT(DAY FROM (SELECT val FROM stichtag) - df.dm_erstes_datum)::INT  END AS tage_seit_dm_diagnose,
    CASE WHEN df.aht THEN EXTRACT(DAY FROM (SELECT val FROM stichtag) - df.aht_erstes_datum)::INT END AS tage_seit_aht_diagnose,
    CASE WHEN df.cvd THEN EXTRACT(DAY FROM (SELECT val FROM stichtag) - df.cvd_erstes_datum)::INT END AS tage_seit_cvd_diagnose,

    -- Labor-Features (NULL = kein Wert in Tomedo dokumentiert)
    lf.egfr_letzter,
    lf.egfr_mittelwert_3,
    lf.hba1c_letzter,
    lf.hba1c_mittelwert_3

FROM eingeschlossene_patienten ep
LEFT JOIN label_icd         ON label_icd.patientid = ep.patientid
LEFT JOIN label_gfr         ON label_gfr.patientid = ep.patientid
LEFT JOIN diagnose_features df ON df.patientid     = ep.patientid
LEFT JOIN labor_features    lf ON lf.patientid     = ep.patientid
-- Sortierung: nach patientid, damit Extrakte reproduzierbar sind. Die Spalte selbst wird NICHT
-- exportiert (s.o.); die Zeilenreihenfolge folgt ihr aber weiterhin. Wenn patientid chronologisch
-- vergeben wird, ist die Zeilenposition ein schwacher Kanal fuer die relative Aufnahmereihenfolge.
-- Downstream daher die Zeilenposition nie als Information behandeln; wer das ausschliessen will,
-- sortiert stattdessen zufaellig (ORDER BY random()) und verliert dafuer die Reproduzierbarkeit.
ORDER BY ep.patientid;