-- Stage 2 — landmark training dataframe (one query, params bound at runtime).
-- Generated from practice_1/CKD_risk_model_dataframe.sql (CheCKD-Trainingsdatensatz, N. Conze).
-- Runtime copy for the Flower ClientApp (data/sql_loader.py); edit the practice_1 original and
-- re-sync this file. The params CTE literals are psycopg named binds (%(name)s).

/*
 Hier beginn die Abfrage für den eigentlichen Dataframe (CheCKD - Trainingsdatensatz)
 Voraussetzung: vorliegende temporary tables aus Stufe 1 dieser query
 (t_lab_tag, t_diag, t_kontakt, t_patient existieren).
 Je Horizont einmal ausfuehren: horizont_jahre = 1, 2, 5.
 */

WITH
-- ============================================================
--  Parameter
-- ============================================================
params AS (
    -- Values are bound at runtime by data/sql_loader.py (psycopg named binds):
    -- praxis from the SuperNode's node_config, the rest from the Flower run config so every
    -- practice in one run uses the IDENTICAL landmark, horizon, and feature window.
    SELECT
        %(praxis)s::text        AS praxis,          -- practice identifier, e.g. 'D001'
        %(stichtag)s::date      AS stichtag,        -- end of outcome window (landmark + horizon)
        %(horizont_jahre)s::int AS horizont_jahre,  -- outcome horizon: 1 / 2 / 5 years
        %(feature_jahre)s::int  AS feature_jahre    -- retrospective window for lab features
),

zeit AS (
    SELECT
        praxis, stichtag, horizont_jahre, feature_jahre,
        (stichtag - make_interval(years => horizont_jahre))::date                 AS cutoff,
        (stichtag - make_interval(years => horizont_jahre + feature_jahre))::date AS feature_start
    FROM params
),

imputation AS (
    SELECT 90.0::numeric   AS gfr,
           ln(10)::numeric AS uacr_log,
           5.4::numeric    AS hba1c,
           ln(10)::numeric AS ualb_log
),

-- ============================================================
--  Kontakte und Kohorte am cutoff
-- ============================================================
kontakt_cutoff AS (
    SELECT
        k.patientid,
        MIN(k.datum)                                      AS erstkontakt,
        MAX(k.datum)                                      AS letztkontakt,
        COUNT(*) FILTER (WHERE k.datum > z.cutoff - 365)  AS kontakte_12m,
        COUNT(*) FILTER (WHERE k.datum > z.feature_start) AS kontakte_feature
    FROM t_kontakt k
    CROSS JOIN zeit z
    WHERE k.datum <= z.cutoff
    GROUP BY k.patientid
),

kohorte AS (
    SELECT
        p.patientid,
        p.geschlecht,
        EXTRACT(YEAR FROM AGE(z.cutoff, p.geburtsdatum))::int AS alter_cutoff,
        p.verstorben,
        p.sterbedatum,
        p.sterbedatum_eff
    FROM t_patient p
    CROSS JOIN zeit z
    JOIN kontakt_cutoff kc ON kc.patientid = p.patientid
    WHERE EXTRACT(YEAR FROM AGE(z.cutoff, p.geburtsdatum)) BETWEEN 18 AND 95
      AND kc.erstkontakt  <= z.cutoff - 365          -- seit >= 1 Jahr in Behandlung
      AND kc.letztkontakt >  z.cutoff - 730          -- in den 2 Jahren vor cutoff gesehen
      -- Direkt gegen die indexierte Tabelle, NICHT gegen eine CTE
      AND NOT EXISTS (SELECT 1 FROM t_diag d
                      WHERE d.patientid = p.patientid
                        AND d.diagnosedatum <= z.cutoff
                        AND (NOT p.verstorben OR p.sterbedatum_eff > z.cutoff)
                        AND (d.abgesetzt IS NULL OR d.abgesetzt IN (0, -100))
                        AND (   d.icd_code IN ('Z49.1', 'Z49.2')
                             OR d.icd_code ^@ ANY (ARRAY['Z99.2', 'Z94.0', 'T86.1'])))
),

-- ============================================================
--  Diagnosen bis cutoff, nur Kohortenpatienten
-- ============================================================
diagnosen_cutoff AS (
    SELECT d.*
    FROM t_diag d
    CROSS JOIN zeit z
    WHERE (d.abgesetzt IS NULL OR d.abgesetzt IN (0, -100))   -- Achtung: heutiger Status
      AND d.diagnosedatum <= z.cutoff
      AND EXISTS (SELECT 1 FROM kohorte ko WHERE ko.patientid = d.patientid)
),

risk_diagnosen AS (
    SELECT
        patientid,
        icd_code,
        MIN(diagnosedatum)::date                                AS erstdatum,
        icd_code ^@ ANY (ARRAY['E10','E11'])                    AS ist_dm,
        icd_code ^@ ANY (ARRAY['I10','I11','I12','I13','I15'])  AS ist_aht
    FROM diagnosen_cutoff
    WHERE icd_code ^@ ANY (ARRAY[
        'E10','E11','I10','I11','I12','I13','I15',
        'I20','I21','I22','I23','I24','I25','I47','I48','I49','I50',
        'I60','I61','I62','I63','I64','I65','I66','I67','I68','I69',
        'I70','I73.9'])
    GROUP BY patientid, icd_code
    HAVING
        NOT (icd_code ^@ ANY (ARRAY[
            'E10','E11','I10','I11','I12','I13','I15','I20','I25',
            'I47','I48','I49','I50','I62.02','I65','I66','I67.2','I70','I73.9']))
        OR (    NOT COALESCE(BOOL_OR(ist_anamnestische_dauerdiagnose), FALSE)
            AND (   COALESCE(BOOL_OR(ist_dauerdiagnose), FALSE)
                 OR icd_code ^@ ANY (ARRAY[
                     'E10','E11','I10','I11','I12','I13','I25',
                     'I62.02','I65','I66','I67.2','I70','I73.9'])))
),

risk_features AS (
    SELECT
        patientid,
        MIN(erstdatum) FILTER (WHERE ist_dm)                     AS dm_erst,
        MIN(erstdatum) FILTER (WHERE ist_aht)                    AS aht_erst,
        MIN(erstdatum) FILTER (WHERE NOT ist_dm AND NOT ist_aht) AS cvd_erst,
        MIN(erstdatum) FILTER (WHERE icd_code ^@ ANY (ARRAY['I20','I21','I22','I23','I24','I25'])) AS khk_erst,
        MIN(erstdatum) FILTER (WHERE icd_code ^@ 'I50')          AS hi_erst,
        MIN(erstdatum) FILTER (WHERE icd_code ^@ 'I48')          AS vhf_erst,
        MIN(erstdatum) FILTER (WHERE icd_code ^@ ANY (ARRAY['I60','I61','I62','I63','I64',
                                                            'I65','I66','I67','I68','I69'])) AS zerebro_erst,
        MIN(erstdatum) FILTER (WHERE icd_code ^@ ANY (ARRAY['I70','I73.9']))                  AS pavk_erst
    FROM risk_diagnosen
    GROUP BY patientid
),

komorbiditaet AS (
    SELECT
        patientid,
        COUNT(DISTINCT left(icd_code, 3))                                        AS n_icd_dreisteller,
        BOOL_OR(icd_code ^@ 'E66')                                               AS adipositas,
        BOOL_OR(icd_code ^@ 'E78')                                               AS dyslipidaemie,
        BOOL_OR(icd_code ^@ ANY (ARRAY['E79','M10']))                            AS hyperurikaemie_gicht,
        BOOL_OR(icd_code ^@ ANY (ARRAY['F17','Z72.0']))                          AS raucher,
        BOOL_OR(icd_code ^@ ANY (ARRAY['D50','D51','D52','D53','D63','D64']))    AS anaemie,
        BOOL_OR(icd_code ^@ 'J44')                                               AS copd,
        BOOL_OR(icd_code ^@ 'N17')                                               AS akutes_nierenversagen,
        BOOL_OR(icd_code ^@ ANY (ARRAY['N20','N21','N22','N23']))                AS urolithiasis,
        BOOL_OR(icd_code ^@ 'N39.0')                                             AS harnwegsinfekt,
        BOOL_OR(icd_code ^@ ANY (ARRAY['K70','K71','K72','K73','K74','K75','K76','K77'])) AS hepatopathie,
        BOOL_OR(icd_code ^@ ANY (ARRAY['M05','M06','M32','M35']))                AS autoimmun_rheuma,
        BOOL_OR(icd_code ^@ ANY (ARRAY['N18.1','N18.2']))                        AS ckd_frueh_icd,
        BOOL_OR(icd_code ^@ ANY (ARRAY['N18.3','N18.4','N18.5']))                AS ckd_g3plus_icd,
        BOOL_OR(icd_code ^@ ANY (ARRAY['N18.4','N18.5']))                        AS ckd_g4plus_icd
    FROM diagnosen_cutoff
    GROUP BY patientid
),

-- ============================================================
--  Laborfeatures im Feature-Fenster
-- ============================================================
lab_features AS (
    SELECT
        l.patientid,
        l.parameter,
        count(*)                                          AS n,
        min(l.wert)                                       AS min_wert,
        max(l.wert)                                       AS max_wert,
        avg(l.wert)                                       AS mittel,
        COALESCE(stddev_samp(l.wert), 0)                  AS sd,
        (array_agg(l.wert ORDER BY l.datum DESC))[1]      AS letzter_wert,
        z.cutoff - max(l.datum)                           AS tage_seit_letzter,
        max(l.datum) - min(l.datum)                       AS spanne_tage,
        avg(l.zensiert::int)                              AS anteil_zensiert,
        regr_slope(l.wert, (l.datum - z.cutoff) / 365.25) AS slope_roh
    FROM t_lab_tag l
    CROSS JOIN zeit z
    WHERE l.datum >  z.feature_start
      AND l.datum <= z.cutoff
      AND EXISTS (SELECT 1 FROM kohorte ko WHERE ko.patientid = l.patientid)
    GROUP BY l.patientid, l.parameter, z.cutoff
),

-- ============================================================
--  Outcome: bestaetigte GFR-Unterschreitung im Fenster (cutoff, stichtag]
-- ============================================================
schwellen AS (
    SELECT unnest(ARRAY[60, 45, 30]) AS schwelle
),

gfr_serie AS (
    SELECT
        l.patientid, s.schwelle, l.datum, l.wert,
        CASE WHEN l.datum <= z.cutoff THEN 'baseline' ELSE 'outcome' END AS fenster,
        (l.wert < s.schwelle) AS unter
    FROM t_lab_tag l
    CROSS JOIN zeit z
    CROSS JOIN schwellen s
    WHERE l.parameter = 'gfr'
      AND l.datum >  z.feature_start
      AND l.datum <= z.stichtag
      AND EXISTS (SELECT 1 FROM kohorte ko WHERE ko.patientid = l.patientid)
),

gfr_inseln AS (
    SELECT *,
           row_number() OVER (PARTITION BY patientid, schwelle, fenster ORDER BY datum)
         - row_number() OVER (PARTITION BY patientid, schwelle, fenster, unter ORDER BY datum) AS insel
    FROM gfr_serie
),

gfr_bestaetigt AS (
    SELECT DISTINCT patientid, schwelle, fenster
    FROM gfr_inseln
    WHERE unter
    GROUP BY patientid, schwelle, fenster, insel
    HAVING max(datum) - min(datum) >= 90
),

gfr_outcome_fenster AS (
    SELECT patientid, schwelle,
           (array_agg(wert ORDER BY datum DESC))[1] AS letzter_wert
    FROM gfr_serie
    WHERE fenster = 'outcome'
    GROUP BY patientid, schwelle
),

nierenersatz_outcome AS (
    SELECT DISTINCT d.patientid
    FROM t_diag d
    CROSS JOIN zeit z
    WHERE d.diagnosedatum >  z.cutoff
      AND d.diagnosedatum <= z.stichtag
      AND (   d.icd_code IN ('Z49.1', 'Z49.2')
           OR d.icd_code ^@ ANY (ARRAY['Z99.2', 'Z94.0', 'N18.5']))
),

outcome AS (
    SELECT
        k.patientid,
        s.schwelle,
        CASE
            WHEN b.patientid IS NOT NULL                                THEN 'praevalent'
            WHEN s.schwelle = 60 AND COALESCE(km.ckd_g3plus_icd, FALSE) THEN 'praevalent'
            WHEN s.schwelle < 60 AND COALESCE(km.ckd_g4plus_icd, FALSE) THEN 'praevalent'
            WHEN ne.patientid IS NOT NULL                               THEN 'positiv'
            WHEN k.verstorben AND k.sterbedatum_eff <= z.stichtag THEN 'verstorben'
            WHEN o.patientid IS NULL                                    THEN 'nicht_bestimmbar'
            WHEN p.patientid IS NOT NULL                                THEN 'positiv'
            WHEN o.letzter_wert >= s.schwelle                           THEN 'negativ'
            ELSE 'unklar'
        END AS status
    FROM kohorte k
    CROSS JOIN zeit z
    CROSS JOIN schwellen s
    LEFT JOIN gfr_bestaetigt b
           ON b.patientid = k.patientid AND b.schwelle = s.schwelle AND b.fenster = 'baseline'
    LEFT JOIN gfr_bestaetigt p
           ON p.patientid = k.patientid AND p.schwelle = s.schwelle AND p.fenster = 'outcome'
    LEFT JOIN gfr_outcome_fenster o
           ON o.patientid = k.patientid AND o.schwelle = s.schwelle
    LEFT JOIN komorbiditaet km        ON km.patientid = k.patientid
    LEFT JOIN nierenersatz_outcome ne ON ne.patientid = k.patientid
),

outcome_wide AS (
    SELECT patientid,
           max(status) FILTER (WHERE schwelle = 60) AS status_g3a,
           max(status) FILTER (WHERE schwelle = 45) AS status_g3b,
           max(status) FILTER (WHERE schwelle = 30) AS status_g4
    FROM outcome
    GROUP BY patientid
)

-- ============================================================
--  Trainingsdatensatz
-- ============================================================
SELECT
    -- Identifikation, NICHT als Feature verwenden
    z.praxis,
    k.patientid,
    z.stichtag,
    z.cutoff,
    z.horizont_jahre,

    -- Demografie
    k.alter_cutoff,
    (k.geschlecht = 'M')::int                                              AS ist_maennlich,

    -- Versorgung
    z.cutoff - kc.erstkontakt                                              AS beobachtung_tage,
    kc.kontakte_12m,
    round(kc.kontakte_feature::numeric / z.feature_jahre, 2)               AS kontakte_pro_jahr,

    -- Risikodiagnosen
    (rf.dm_erst IS NOT NULL)::int                                          AS dm,
    COALESCE(z.cutoff - rf.dm_erst, 0)                                     AS dm_dauer_tage,
    COALESCE((abs(rf.dm_erst - kc.erstkontakt) <= 90)::int, 0)             AS dm_datum_bei_aufnahme,
    (rf.aht_erst IS NOT NULL)::int                                         AS aht,
    COALESCE(z.cutoff - rf.aht_erst, 0)                                    AS aht_dauer_tage,
    COALESCE((abs(rf.aht_erst - kc.erstkontakt) <= 90)::int, 0)            AS aht_datum_bei_aufnahme,
    (rf.cvd_erst IS NOT NULL)::int                                         AS cvd,
    COALESCE(z.cutoff - rf.cvd_erst, 0)                                    AS cvd_dauer_tage,
    COALESCE((abs(rf.cvd_erst - kc.erstkontakt) <= 90)::int, 0)            AS cvd_datum_bei_aufnahme,
    (rf.khk_erst     IS NOT NULL)::int                                     AS khk,
    (rf.hi_erst      IS NOT NULL)::int                                     AS herzinsuffizienz,
    (rf.vhf_erst     IS NOT NULL)::int                                     AS vorhofflimmern,
    (rf.zerebro_erst IS NOT NULL)::int                                     AS zerebrovaskulaer,
    (rf.pavk_erst    IS NOT NULL)::int                                     AS pavk_atherosklerose,

    -- Komorbiditaet
    COALESCE(km.n_icd_dreisteller, 0)                                      AS n_icd_dreisteller,
    COALESCE(km.adipositas, FALSE)::int                                    AS adipositas,
    COALESCE(km.dyslipidaemie, FALSE)::int                                 AS dyslipidaemie,
    COALESCE(km.hyperurikaemie_gicht, FALSE)::int                          AS hyperurikaemie_gicht,
    COALESCE(km.raucher, FALSE)::int                                       AS raucher,
    COALESCE(km.anaemie, FALSE)::int                                       AS anaemie,
    COALESCE(km.copd, FALSE)::int                                          AS copd,
    COALESCE(km.akutes_nierenversagen, FALSE)::int                         AS akutes_nierenversagen,
    COALESCE(km.urolithiasis, FALSE)::int                                  AS urolithiasis,
    COALESCE(km.harnwegsinfekt, FALSE)::int                                AS harnwegsinfekt,
    COALESCE(km.hepatopathie, FALSE)::int                                  AS hepatopathie,
    COALESCE(km.autoimmun_rheuma, FALSE)::int                              AS autoimmun_rheuma,
    COALESCE(km.ckd_frueh_icd, FALSE)::int                                 AS ckd_frueh_icd,
    COALESCE(km.ckd_g3plus_icd, FALSE)::int                                AS ckd_g3plus_icd,

    -- GFR (ml/min)
    (fg.patientid IS NOT NULL)::int                                        AS gfr_hat_messung,
    COALESCE(fg.n, 0)                                                      AS gfr_n,
    round(COALESCE(fg.n, 0)::numeric / z.feature_jahre, 2)                 AS gfr_n_pro_jahr,
    round(COALESCE(fg.letzter_wert, i.gfr), 1)                             AS gfr_letzter,
    round(COALESCE(fg.min_wert,     i.gfr), 1)                             AS gfr_min,
    round(COALESCE(fg.max_wert,     i.gfr), 1)                             AS gfr_max,
    round(COALESCE(fg.mittel,       i.gfr), 1)                             AS gfr_mittel,
    round(COALESCE(fg.sd, 0), 2)                                           AS gfr_sd,
    COALESCE(fg.tage_seit_letzter, z.feature_jahre * 365)                  AS gfr_tage_seit_letzter,
    COALESCE(fg.spanne_tage, 0)                                            AS gfr_spanne_tage,
    round(COALESCE(fg.anteil_zensiert, 0), 2)                              AS gfr_anteil_zensiert,
    COALESCE((fg.n >= 3 AND fg.spanne_tage >= 365)::int, 0)                AS gfr_slope_valide,
    round((CASE WHEN fg.n >= 3 AND fg.spanne_tage >= 365
                THEN fg.slope_roh ELSE 0 END)::numeric, 3)                 AS gfr_slope_pro_jahr,

    -- UACR (ln mg/g)
    (fu.patientid IS NOT NULL)::int                                        AS uacr_hat_messung,
    COALESCE(fu.n, 0)                                                      AS uacr_n,
    round(COALESCE(fu.n, 0)::numeric / z.feature_jahre, 2)                 AS uacr_n_pro_jahr,
    round(COALESCE(fu.letzter_wert, i.uacr_log), 3)                        AS uacr_log_letzter,
    round(COALESCE(fu.min_wert,     i.uacr_log), 3)                        AS uacr_log_min,
    round(COALESCE(fu.max_wert,     i.uacr_log), 3)                        AS uacr_log_max,
    round(COALESCE(fu.mittel,       i.uacr_log), 3)                        AS uacr_log_mittel,
    round(COALESCE(fu.sd, 0), 3)                                           AS uacr_log_sd,
    COALESCE(fu.tage_seit_letzter, z.feature_jahre * 365)                  AS uacr_tage_seit_letzter,
    COALESCE(fu.spanne_tage, 0)                                            AS uacr_spanne_tage,
    round(COALESCE(fu.anteil_zensiert, 0), 2)                              AS uacr_anteil_zensiert,
    COALESCE((fu.n >= 3 AND fu.spanne_tage >= 365)::int, 0)                AS uacr_slope_valide,
    round((CASE WHEN fu.n >= 3 AND fu.spanne_tage >= 365
                THEN fu.slope_roh ELSE 0 END)::numeric, 3)                 AS uacr_log_slope_pro_jahr,

    -- HbA1c (%)
    (fh.patientid IS NOT NULL)::int                                        AS hba1c_hat_messung,
    COALESCE(fh.n, 0)                                                      AS hba1c_n,
    round(COALESCE(fh.n, 0)::numeric / z.feature_jahre, 2)                 AS hba1c_n_pro_jahr,
    round(COALESCE(fh.letzter_wert, i.hba1c), 2)                           AS hba1c_letzter,
    round(COALESCE(fh.min_wert,     i.hba1c), 2)                           AS hba1c_min,
    round(COALESCE(fh.max_wert,     i.hba1c), 2)                           AS hba1c_max,
    round(COALESCE(fh.mittel,       i.hba1c), 2)                           AS hba1c_mittel,
    round(COALESCE(fh.sd, 0), 3)                                           AS hba1c_sd,
    COALESCE(fh.tage_seit_letzter, z.feature_jahre * 365)                  AS hba1c_tage_seit_letzter,
    COALESCE(fh.spanne_tage, 0)                                            AS hba1c_spanne_tage,
    COALESCE((fh.n >= 3 AND fh.spanne_tage >= 365)::int, 0)                AS hba1c_slope_valide,
    round((CASE WHEN fh.n >= 3 AND fh.spanne_tage >= 365
                THEN fh.slope_roh ELSE 0 END)::numeric, 3)                 AS hba1c_slope_pro_jahr,

    -- Urin-Albumin ohne Quotient (ln mg/l)
    (fa.patientid IS NOT NULL)::int                                        AS ualb_hat_messung,
    COALESCE(fa.n, 0)                                                      AS ualb_n,
    round(COALESCE(fa.letzter_wert, i.ualb_log), 3)                        AS ualb_log_letzter,
    round(COALESCE(fa.mittel,       i.ualb_log), 3)                        AS ualb_log_mittel,
    round(COALESCE(fa.max_wert,     i.ualb_log), 3)                        AS ualb_log_max,
    COALESCE(fa.tage_seit_letzter, z.feature_jahre * 365)                  AS ualb_tage_seit_letzter,

    -- Outcome, NICHT als Feature verwenden. Label NULL = nicht trainierbar.
    ow.status_g3a,
    CASE ow.status_g3a WHEN 'positiv' THEN 1 WHEN 'negativ' THEN 0 END     AS label_g3a,
    ow.status_g3b,
    CASE ow.status_g3b WHEN 'positiv' THEN 1 WHEN 'negativ' THEN 0 END     AS label_g3b,
    ow.status_g4,
    CASE ow.status_g4  WHEN 'positiv' THEN 1 WHEN 'negativ' THEN 0 END     AS label_g4

FROM kohorte k
CROSS JOIN zeit z
CROSS JOIN imputation i
JOIN      kontakt_cutoff kc ON kc.patientid = k.patientid
LEFT JOIN risk_features  rf ON rf.patientid = k.patientid
LEFT JOIN komorbiditaet  km ON km.patientid = k.patientid
LEFT JOIN lab_features   fg ON fg.patientid = k.patientid AND fg.parameter = 'gfr'
LEFT JOIN lab_features   fu ON fu.patientid = k.patientid AND fu.parameter = 'uacr_log'
LEFT JOIN lab_features   fh ON fh.patientid = k.patientid AND fh.parameter = 'hba1c'
LEFT JOIN lab_features   fa ON fa.patientid = k.patientid AND fa.parameter = 'ualb_log'
LEFT JOIN outcome_wide   ow ON ow.patientid = k.patientid;


