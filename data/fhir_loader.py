"""Read one practice's cohort from its local HL7 FHIR server (R4).

This is the production data path (Projektantrag AP1/T1.2: each practice runs its own FHIR
database; `extract_features.sql` is the Tomedo→PostgreSQL equivalent for practices not yet on
FHIR). The SuperNode runs this on the practice's own machine — the resulting rows are used to fit
the local model and are **never transmitted**; only weights leave (immutable rule 3).

The frame it builds follows the **canonical contract of `extract_features.sql`, not the synthetic
10-feature schema** (CLAUDE.md §3b): German column names, eGFR/HbA1c labs, `geschlecht`, and a CKD
**incidence** label under a rolling landmark — `Stichtag` `t0 = today - 365d`, outcome window
`[t0, t0 + 365d)`. Where FHIR has no 1:1 equivalent for a Tomedo concept, the mapping chosen is
documented inline and mirrors the SQL as written (including its open exclusion TODOs, kept in
parity rather than silently diverging between the two extraction paths).

De-identification (privacy layer L0, PRIVACY.md §2) is structural, not a filter the caller can
forget: only `birthDate`, `gender`, condition codes/dates, lab values/dates and encounter dates are
ever read — names, addresses, telecom and identifiers are never touched. The FHIR patient `id` is
used purely in memory to join resources, never enters a row, and rows are emitted **ordered by
SHA-256(id)**, which closes the ordering channel PRIVACY.md §2 documents for the SQL export
(`ORDER BY patientid`) while staying deterministic per extraction. If T4.2 ever needs a stable
join key across extractions, `pseudonym_salt` enables a per-practice HMAC-SHA256 pseudonym column
— the salted-hash replacement PRIVACY.md §2 names, a deliberate opt-in, never the raw id.

Resources consumed (FHIR R4):
- `Patient`      -> demographics, inclusion (alive, 18-95 at t0, sex documented), test-tag screen
- `Encounter`    -> first contact; inclusion needs one ending >= 52 weeks before t0
- `Condition`    -> comorbidity flags (dm/aht/cvd), `tage_seit_*`, prior-CKD exclusion, label A
- `Observation`  -> eGFR and HbA1c (feature aggregation + label B, the KDIGO approximation)

Stdlib-only HTTP (no `requests` dependency) to keep the client footprint small, since this runs
inside every practice's SuperNode.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

import pandas as pd

from .loader import (
    CANONICAL_BINARY_FLAG_COLS,
    CANONICAL_FEATURE_COLS,
    CANONICAL_LABEL_COL,
    CANONICAL_LAB_COLS,
    CANONICAL_META_COLS,
    CANONICAL_TAGE_COLS,
)

# ── Code systems — mirroring extract_features.sql exactly ────────────────────

# Comorb flags use the SQL's ICD-10-GM prefix sets, so the SQL and FHIR paths classify
# identically (SQL §5; CVD coverage deliberately keeps the SQL's TODO gaps — I47-I49, I60-I62,
# I70, I73.9 stay out until the contract is amended there).
ICD_PREFIXES: dict[str, tuple[str, ...]] = {
    "dm": ("E10", "E11", "E13", "E14"),
    "aht": ("I10", "I11", "I12", "I13", "I15"),
    "cvd": ("I20", "I21", "I22", "I23", "I24", "I25", "I50", "I63", "I64", "I65", "I66"),
}

# flag -> its paired tage_seit_ column (SQL §5).
_TAGE_COL = {
    "dm": "tage_seit_dm_diagnose",
    "aht": "tage_seit_aht_diagnose",
    "cvd": "tage_seit_cvd_diagnose",
}

# Any N18.* confirmed before t0 excludes (SQL §2); confirmed N18.* in [t0, t0+365d) is label A
# (SQL §3). Dialysis (Z99.2/OPS 8-854.*), transplant (Z94.0) and Vertretungsscheine are open SQL
# TODOs and are NOT applied here either — the two extraction paths must not diverge silently.
CKD_ICD_PREFIX = "N18"

# Labs: primary match is a LOINC coding (system http://loinc.org); the prefix lists replicate the
# SQL's broad testident fallback for exports whose lab codings are local-only. Units are taken as
# recorded (mL/min/1.73m^2 / %) — harmonised units are the IKIM/SHIP platform's job, and the same
# assumption the SQL makes about Tomedo. Tune here if a practice export proves otherwise.
EGFR_LOINC: tuple[str, ...] = (
    "62238-1",  # eGFR, CKD-EPI 2021 (race-free)
    "98979-8",  # eGFR, CKD-EPI
    "48642-3",  # eGFR, MDRD (non-black)
    "48643-1",  # eGFR, MDRD (black)
    "50384-7",  # eGFR, CKD-EPI (black)
)
EGFR_LOCAL_PREFIXES: tuple[str, ...] = ("GFR", "GLOMFI", "EGFR", "eGFR")
HBA1C_LOINC: tuple[str, ...] = (
    "4548-4",  # HbA1c %
    "17856-6",  # HbA1c % by HPLC
    "59261-8",  # HbA1c %
)
HBA1C_LOCAL_PREFIXES: tuple[str, ...] = ("HBA1C", "HBAKAP")

# SQL: typ='G' (gesichert), nicht anamnestisch. FHIR mapping (pragmatic, reviewable by docport):
# verificationStatus must be `confirmed` — provisional/unconfirmed/differential/refuted and
# entered-in-error are dropped — and a clinicalStatus of inactive/resolved/remission screens out
# the anamnestic history states the SQL excludes via `ist_anamnestische_dauerdiagnose`.
_EXCLUDED_CLINICAL_STATUS = {"inactive", "resolved", "remission"}
_CONFIRMED = "confirmed"

# Testpatienten: Tomedo keeps a flag table; FHIR conveys this at most as meta tags (HAPI: HTEST).
_TEST_PATIENT_TAGS = {"HTEST"}

_OBS_USABLE_STATUS = {"final", "amended", "corrected", None}

_AGE_MIN, _AGE_MAX = 18, 95           # SQL §1: age range at Stichtag
_LANDMARK_DAYS = 365                  # SQL §0: Stichtag = CURRENT_DATE - 365 days
_OUTCOME_DAYS = 365                   # SQL §0: Zielfenster [t0, t0+365d)
_FIRST_CONTACT_WEEKS = 52             # SQL §1: Erstkontakt >= 52 weeks before t0
_KDIGO_PREDECESSOR_DAYS = 90          # SQL §4: persistence window


def _get(url: str, timeout: float) -> dict:
    # Whole patient resources cross this socket, so the transport guard is in code, not docs:
    # plaintext HTTP is loopback-only (the extraction runs on the practice machine; anything
    # off-box must be TLS).
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" and (parsed.hostname or "").lower() not in (
        "localhost", "127.0.0.1", "::1",
    ):
        raise ValueError(
            f"FHIR base URL {url!r} is neither https nor loopback — patient resources would "
            f"cross this host unencrypted"
        )
    req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - guard above
        return json.loads(resp.read().decode("utf-8"))


def _iter_bundle(base_url: str, resource: str, params: dict, timeout: float):
    """Yield every resource in a FHIR search Bundle, following `next` links."""
    url = f"{base_url.rstrip('/')}/{resource}?{urllib.parse.urlencode(params)}"
    while url:
        bundle = _get(url, timeout)
        for entry in bundle.get("entry", []) or []:
            if "resource" in entry:
                yield entry["resource"]
        url = next(
            (lk.get("url") for lk in bundle.get("link", []) or [] if lk.get("relation") == "next"),
            None,
        )


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None


def _codes(resource: dict) -> list[str]:
    return [
        str(c.get("code", ""))
        for c in (resource.get("code", {}).get("coding", []) or [])
        if c.get("code")
    ]


def _coding_has(resource: dict, *, system: str, codes: tuple[str, ...]) -> bool:
    return any(
        c.get("system") == system and c.get("code") in codes
        for c in (resource.get("code", {}).get("coding", []) or [])
    )


def _subject_id(resource: dict) -> str | None:
    ref = resource.get("subject", {}).get("reference", "")
    if ref.startswith("Patient/"):
        return ref.split("/", 1)[1]
    # Absolute references — some HAPI configurations emit "https://hapi/fhir/Patient/123":
    # take the trailing /Patient/<id> pair. Dropping these orphans the patient's Conditions,
    # Observations, and Encounters, silently contaminating the exclusion window and labels.
    parts = ref.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-2] == "Patient" and parts[-1]:
        return parts[-1]
    return None


def _coding_code(resource: dict, field: str) -> str | None:
    for c in (resource.get(field, {}).get("coding", []) or []):
        if c.get("code"):
            return str(c["code"])
    return None


def _is_confirmed_current(condition: dict) -> bool:
    """SQL `typ = 'G' AND NOT ist_anamnestische_dauerdiagnose` mapped to FHIR statuses."""
    if _coding_code(condition, "verificationStatus") != _CONFIRMED:
        return False
    return _coding_code(condition, "clinicalStatus") not in _EXCLUDED_CLINICAL_STATUS


def _condition_date(condition: dict) -> date | None:
    return _parse_date(condition.get("onsetDateTime")) or _parse_date(condition.get("recordedDate"))


def _lab_match(observation: dict, loinc: tuple[str, ...], local_prefixes: tuple[str, ...]) -> bool:
    if _coding_has(observation, system="http://loinc.org", codes=loinc):
        return True
    return any(
        code.startswith(local_prefixes) for code in _codes(observation)
    )


def _lab_value(observation: dict) -> float | None:
    value = (observation.get("valueQuantity") or {}).get("value")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lab_series(
    observations: list[dict],
    loinc: tuple[str, ...],
    local_prefixes: tuple[str, ...],
    before: date,
    stats: dict | None = None,
) -> dict[str, list[tuple[date, float]]]:
    """patient id -> ascending (date, value) list of usable measurements strictly before `before`."""
    series: dict[str, list[tuple[date, float]]] = {}
    for obs in observations:
        if obs.get("status") not in _OBS_USABLE_STATUS:
            continue
        pid = _subject_id(obs)
        if pid is None:
            if stats is not None:
                stats["unlinked_references"] += 1
            continue
        if not _lab_match(obs, loinc, local_prefixes):
            continue
        when = _parse_date(obs.get("effectiveDateTime"))
        value = _lab_value(obs)
        if when is None or value is None or not when < before:
            continue
        series.setdefault(pid, []).append((when, value))
    for values in series.values():
        values.sort()
    return series


def _lab_features(series: list[tuple[date, float]] | None) -> tuple[float | None, float | None]:
    """SQL §8: last value; mean of last 3 only when >= 2 exist (else None — informative NaN)."""
    if not series:
        return None, None
    last3 = [v for _, v in series[-3:]]
    mean3 = sum(last3) / len(last3) if len(last3) >= 2 else None
    return series[-1][1], mean3


def _egfr_persistent(series: list[tuple[date, float]] | None, window_start: date) -> bool:
    """SQL §4 label B: a measurement in [window_start, ...] <= 60 whose predecessor >= 90 days
    earlier is also <= 60, with **every** value in the trailing 90-day window <= 60."""
    if not series:
        return False
    for when, value in series:
        if when < window_start or value > 60:
            continue
        predecessors = [(d, v) for d, v in series if d <= when - timedelta(days=_KDIGO_PREDECESSOR_DAYS)]
        if not predecessors or predecessors[-1][1] > 60:
            continue
        in_window = [v for d, v in series if when - timedelta(days=_KDIGO_PREDECESSOR_DAYS) <= d <= when]
        if all(v <= 60 for v in in_window):
            return True
    return False


def _age_completed_years(birth: date, at: date) -> int:
    """SQL §1: EXTRACT(YEAR FROM AGE(t0, geburtsdatum)) — completed years, not fractional."""
    return at.year - birth.year - ((at.month, at.day) < (birth.month, birth.day))


def _extraction_stats() -> dict:
    return {
        "patients_total": 0,
        "excluded_test": 0,
        "unlinked_references": 0,
        "excluded_deceased": 0,
        "excluded_demographics": 0,
        "excluded_age": 0,
        "excluded_no_early_contact": 0,
        "excluded_prior_ckd": 0,
        "included": 0,
        "label_icd": 0,
        "label_egfr": 0,
    }


def _extract(
    base_url: str,
    *,
    page_size: int,
    timeout: float,
    stichtag: date,
    pseudonym_salt: str | None,
) -> tuple[pd.DataFrame, dict]:
    """Full extraction + exclusion accounting. `load_practice_frame` is the quiet wrapper."""
    t0 = stichtag
    window_end = t0 + timedelta(days=_OUTCOME_DAYS)
    stats = _extraction_stats()

    try:
        patients = list(_iter_bundle(base_url, "Patient", {"_count": page_size}, timeout))
        encounters = list(_iter_bundle(base_url, "Encounter", {"_count": page_size}, timeout))
        conditions = list(_iter_bundle(base_url, "Condition", {"_count": page_size}, timeout))
        observations = list(
            _iter_bundle(
                base_url, "Observation", {"_count": page_size, "category": "laboratory"}, timeout
            )
        )
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ConnectionError(
            f"Could not reach the FHIR server at {base_url!r}: {exc}. "
            "Check fhir-base-url in --node-config (docs/DEPLOYMENT.md), or run with "
            "data-source=csv."
        ) from exc

    if not patients:
        raise ValueError(f"FHIR server at {base_url!r} returned no Patient resources.")
    stats["patients_total"] = len(patients)

    # ── Earliest contact per patient (SQL §1's billing-source union -> Encounter dates) ──
    first_contact: dict[str, date] = {}
    for enc in encounters:
        pid = _subject_id(enc)
        if pid is None:
            stats["unlinked_references"] += 1
            continue
        period = enc.get("period") or {}
        when = (
            _parse_date(period.get("start"))
            or _parse_date(period.get("end"))
            or _parse_date(enc.get("meta", {}).get("lastUpdated"))
        )
        if when is not None and (pid not in first_contact or when < first_contact[pid]):
            first_contact[pid] = when

    # ── Confirmed current conditions, grouped once (SQL §2/§3/§5 all read this population) ──
    by_patient: dict[str, list[tuple[list[str], date | None]]] = {}
    for cond in conditions:
        pid = _subject_id(cond)
        if pid is None:
            stats["unlinked_references"] += 1
            continue
        if not _is_confirmed_current(cond):
            continue
        by_patient.setdefault(pid, []).append((_codes(cond), _condition_date(cond)))

    # Labs: feature series end at t0 (SQL §6/§7); the label series runs to end of window (§4).
    # stats counts unlinked observations once — the same list feeds all three _lab_series calls, so
    # only the first pass tallies them.
    egfr_pre = _lab_series(observations, EGFR_LOINC, EGFR_LOCAL_PREFIXES, t0, stats)
    hba1c_pre = _lab_series(observations, HBA1C_LOINC, HBA1C_LOCAL_PREFIXES, t0)
    egfr_window = _lab_series(observations, EGFR_LOINC, EGFR_LOCAL_PREFIXES, window_end)

    earliest_contact_allowed = t0 - timedelta(weeks=_FIRST_CONTACT_WEEKS)

    rows: dict[str, dict] = {}
    for patient in patients:
        pid = patient.get("id")
        if not pid:
            continue

        # Testpatienten (SQL §1 via test_patienten).
        tags = {
            str(t.get("code"))
            for t in (patient.get("meta", {}).get("tag", []) or [])
            if t.get("code")
        }
        if tags & _TEST_PATIENT_TAGS:
            stats["excluded_test"] += 1
            continue

        # Lebende Patienten.
        if patient.get("deceasedBoolean") or patient.get("deceasedDateTime"):
            stats["excluded_deceased"] += 1
            continue

        # Geburtsdatum + Geschlecht dokumentiert (SQL CASE maps M->1, W->0; anything else is not
        # part of the contract, mirroring `ELSE NULL` + the IS NOT NULL inclusion).
        birth = _parse_date(patient.get("birthDate"))
        gender = patient.get("gender")
        if birth is None or gender not in ("male", "female"):
            stats["excluded_demographics"] += 1
            continue

        alter = _age_completed_years(birth, t0)
        if not (_AGE_MIN <= alter <= _AGE_MAX):
            stats["excluded_age"] += 1
            continue

        # Erstkontakt >= 52 Wochen vor t0.
        if first_contact.get(pid) is None or first_contact[pid] > earliest_contact_allowed:
            stats["excluded_no_early_contact"] += 1
            continue

        conds = by_patient.get(pid, [])
        codes_with_dates = [(code, when) for codes, when in conds for code in codes]

        # SQL §2: vorbekannte CKD — any confirmed N18.* before t0 excludes. An undated confirmed
        # N18 cannot be proven post-t0 and is conservatively treated as prior.
        prior_ckd = any(
            code.startswith(CKD_ICD_PREFIX) and (when is None or when < t0)
            for code, when in codes_with_dates
        )
        if prior_ckd:
            stats["excluded_prior_ckd"] += 1
            continue

        row: dict[str, object] = {
            "t0": t0.isoformat(),
            "alter_jahre": alter,
            "geschlecht": 1 if gender == "male" else 0,
        }

        # SQL §3, label A: confirmed N18.* whose date falls inside the outcome window.
        row[CANONICAL_LABEL_COL] = int(
            any(
                code.startswith(CKD_ICD_PREFIX) and when is not None and t0 <= when < window_end
                for code, when in codes_with_dates
            )
        )
        stats["label_icd"] += row[CANONICAL_LABEL_COL]

        # SQL §4, label B: KDIGO persistence over the window (predecessor may predate t0).
        if not row[CANONICAL_LABEL_COL] and _egfr_persistent(egfr_window.get(pid), t0):
            row[CANONICAL_LABEL_COL] = 1
            stats["label_egfr"] += 1

        # SQL §5: Vordiagnosen — flag from confirmed codes before t0 (undated counts as present);
        # tage_seit_* from the earliest dated onset, NULL when the flag is absent.
        for flag, prefixes in ICD_PREFIXES.items():
            onsets = [
                when for code, when in codes_with_dates if code.startswith(prefixes)
            ]
            dated = [d for d in onsets if d is not None and d < t0]
            row[flag] = 1 if dated or any(d is None for d in onsets) else 0
            first_dx = min(dated, default=None)
            row[_TAGE_COL[flag]] = (t0 - first_dx).days if row[flag] and first_dx else None

        # SQL §6-§8: labs before t0; letzter / mittelwert_3 (>= 2 values), NULL informative.
        egfr_last, egfr_mean3 = _lab_features(egfr_pre.get(pid))
        hba1c_last, hba1c_mean3 = _lab_features(hba1c_pre.get(pid))
        row["egfr_letzter"], row["egfr_mittelwert_3"] = egfr_last, egfr_mean3
        row["hba1c_letzter"], row["hba1c_mittelwert_3"] = hba1c_last, hba1c_mean3

        rows[pid] = row
        stats["included"] += 1

    # L0: rows keyed by SHA-256(id) — deterministic per extraction, but unlike the SQL export's
    # `ORDER BY patientid` it carries no information about enrolment order.
    ordered_ids = sorted(rows, key=lambda pid: hashlib.sha256(pid.encode("utf-8")).hexdigest())
    out_rows = []
    for pid in ordered_ids:
        row = rows[pid]
        if pseudonym_salt is not None:
            row = {
                "patient_pseudonym": hmac.new(
                    pseudonym_salt.encode("utf-8"), pid.encode("utf-8"), hashlib.sha256
                ).hexdigest(),
                **row,
            }
        out_rows.append(row)

    df = pd.DataFrame(
        out_rows,
        columns=(
            (["patient_pseudonym"] if pseudonym_salt is not None else [])
            + [*CANONICAL_META_COLS, *CANONICAL_FEATURE_COLS, *CANONICAL_LAB_COLS, CANONICAL_LABEL_COL]
        ),
    )
    for col in (*CANONICAL_BINARY_FLAG_COLS, CANONICAL_LABEL_COL):
        df[col] = df[col].astype("int64")
    for col in (*CANONICAL_TAGE_COLS, *CANONICAL_LAB_COLS):
        df[col] = df[col].astype("float64")
    return df, stats


def load_practice_frame(
    base_url: str,
    *,
    page_size: int = 500,
    timeout: float = 30.0,
    stichtag: date | None = None,
    pseudonym_salt: str | None = None,
) -> pd.DataFrame:
    """Query one practice's FHIR server and return the canonical landmark frame.

    Args:
        base_url: FHIR base, e.g. ``http://localhost:8080/fhir``.
        page_size: `_count` per search page.
        timeout: per-request timeout in seconds.
        stichtag: the landmark t0 (fixed for reproducible runs); defaults to
            ``today - 365 days`` like the SQL's rolling Stichtag.
        pseudonym_salt: opt-in per-practice salt. When set, a `patient_pseudonym`
            column (HMAC-SHA256 over the FHIR id) is prepended — the named T4.2 join-key
            mechanism. Default None: no per-row key at all (strictest L0).

    The frame carries `t0` as a constant column (the SQL exports it for reproducibility). No
    patient identifier enters any feature row; ordering is by SHA-256 of the in-memory id, so row
    position is information-free (PRIVACY.md §2).
    """
    t0 = stichtag or (date.today() - timedelta(days=_LANDMARK_DAYS))
    df, stats = _extract(
        base_url, page_size=page_size, timeout=timeout, stichtag=t0, pseudonym_salt=pseudonym_salt
    )
    if not len(df):
        raise ValueError(
            f"FHIR server at {base_url!r}: 0 of {stats['patients_total']} patients met the "
            f"inclusion criteria of extract_features.sql "
            f"(excluded: test={stats['excluded_test']}, deceased={stats['excluded_deceased']}, "
            f"demographics={stats['excluded_demographics']}, age={stats['excluded_age']}, "
            f"no-early-contact={stats['excluded_no_early_contact']}, "
            f"prior-ckd={stats['excluded_prior_ckd']})."
        )
    return df


def main() -> None:
    """Practice-side rehearsal/extraction CLI: `uv run ckd-fhir-extract <base-url>`.

    Runs on the practice machine against its own server (same queries the SuperNode issues),
    prints the exclusion accounting and cohort summary, and writes the de-identified canonical CSV
    only when asked to. Nothing leaves the machine.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical CKD landmark table (extract_features.sql contract) from one "
            "practice's local FHIR server. De-identified: no identifiers are read or emitted."
        )
    )
    parser.add_argument("base_url", help="FHIR base URL, e.g. http://localhost:8080/fhir")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--t0",
        type=date.fromisoformat,
        default=None,
        help="Stichtag (YYYY-MM-DD), fixed for reproducibility; default: today - 365d",
    )
    parser.add_argument(
        "--pseudonym-salt",
        default=None,
        help="Enable the per-practice salted-HMAC join key (T4.2 mechanism; off by default)",
    )
    parser.add_argument(
        "--write-csv",
        default=None,
        help="Also write the de-identified extract here (default: print summary only)",
    )
    args = parser.parse_args()

    t0 = args.t0 or (date.today() - timedelta(days=_LANDMARK_DAYS))
    df, stats = _extract(
        args.base_url,
        page_size=args.page_size,
        timeout=args.timeout,
        stichtag=t0,
        pseudonym_salt=args.pseudonym_salt,
    )

    print(f"Stichtag t0 = {t0}  (Zielfenster [{t0}, {t0 + timedelta(days=_OUTCOME_DAYS)}))")
    for key, value in stats.items():
        print(f"  {key:<24} {value}")
    n = len(df)
    if n:
        print(f"included                    {n}")
        print(f"ckd_incident rate           {df[CANONICAL_LABEL_COL].mean():.3f}")
        for col in CANONICAL_LAB_COLS:
            print(f"{col:<27} missing in {df[col].isna().mean():.1%}")
    if args.write_csv:
        df.to_csv(args.write_csv, index=False)
        print(f"de-identified extract written to {args.write_csv}")


if __name__ == "__main__":
    main()
