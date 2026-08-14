"""Read one practice's cohort from its local HL7 FHIR server.

This is the production data path (Projektantrag AP1/T1.2: each practice runs its own FHIR database;
`extract_features.sql` is the Tomedo→PostgreSQL equivalent for practices not yet on FHIR). The
SuperNode calls this on the practice's own machine — the resulting rows are used to fit the local
model and are **never transmitted**; only weights leave (immutable rule 3).

Output schema is `data.loader.FEATURE_COLS + LABEL_COL`, so everything downstream (`to_xy`,
`build_client_from_frame`, every model) is unchanged whether rows came from CSV or FHIR.

Resources consumed:
- `Patient`      -> `age_years` (from `birthDate`)
- `Condition`    -> the `dx_*` flags and `years_since_*` (from `onsetDateTime`), and the CKD label

Stdlib-only HTTP (no `requests` dependency) to keep the client footprint small, since this runs
inside every practice's SuperNode.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime

import pandas as pd

from .loader import BINARY_FLAG_COLS, FEATURE_COLS, LABEL_COL, YEARS_SINCE_COLS

# ICD-10-GM prefixes per comorbidity, matching the synthetic schema documented in CLAUDE.md §3a.
ICD_PREFIXES: dict[str, tuple[str, ...]] = {
    "dx_hypertonie": ("I10",),
    "dx_diabetes": ("E10", "E11", "E13"),
    "dx_khk": ("I25",),
    "dx_adipositas": ("E66",),
    "dx_herzinsuffizienz": ("I50", "I11.0"),
    "dx_hyperurikaemie": ("M10",),
}

# CKD stage >= 3 (the synthetic label). N18.3-N18.6 only — N18.1/N18.2 are stages 1-2.
CKD_STAGE3PLUS_PREFIXES: tuple[str, ...] = ("N18.3", "N18.4", "N18.5", "N18.6")

# dx_ flag -> its paired years_since_ column.
_YEARS_COL = {
    "dx_hypertonie": "years_since_hypertonie_dx",
    "dx_diabetes": "years_since_diabetes_dx",
    "dx_khk": "years_since_khk_dx",
}


def _get(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - operator-supplied URL
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


def _years_between(start: date | None, end: date) -> float:
    if start is None:
        return 0.0
    return max(0.0, (end - start).days / 365.25)


def _codes(condition: dict) -> list[str]:
    return [
        c.get("code", "")
        for c in (condition.get("code", {}).get("coding", []) or [])
        if c.get("code")
    ]


def _patient_id(condition: dict) -> str | None:
    ref = condition.get("subject", {}).get("reference", "")
    return ref.split("/", 1)[1] if ref.startswith("Patient/") else None


def load_practice_frame(
    base_url: str,
    *,
    page_size: int = 500,
    timeout: float = 30.0,
    index_date: date | None = None,
) -> pd.DataFrame:
    """Query one practice's FHIR server and return a model-ready DataFrame.

    Args:
        base_url: FHIR base, e.g. ``http://localhost:8080/fhir``.
        page_size: `_count` per search page.
        timeout: per-request timeout in seconds.
        index_date: landmark for age and `years_since_*`; defaults to today.

    Note: no patient identifier is carried into the returned frame (privacy layer L0) — rows are
    keyed positionally only.
    """
    t0 = index_date or date.today()

    try:
        patients = list(_iter_bundle(base_url, "Patient", {"_count": page_size}, timeout))
        conditions = list(_iter_bundle(base_url, "Condition", {"_count": page_size}, timeout))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise ConnectionError(
            f"Could not reach the FHIR server at {base_url!r}: {exc}. "
            "Start one with ./scripts/fhir_dev_server.sh, or run with data-source=csv."
        ) from exc

    if not patients:
        raise ValueError(f"FHIR server at {base_url!r} returned no Patient resources.")

    # Group conditions by patient once, rather than re-scanning per patient.
    by_patient: dict[str, list[dict]] = {}
    for cond in conditions:
        pid = _patient_id(cond)
        if pid:
            by_patient.setdefault(pid, []).append(cond)

    rows = []
    for patient in patients:
        pid = patient.get("id")
        birth = _parse_date(patient.get("birthDate"))
        row: dict[str, float | int] = {"age_years": _years_between(birth, t0)}

        patient_conditions = by_patient.get(pid, [])
        codes_with_onset = [
            (code, _parse_date(c.get("onsetDateTime"))) for c in patient_conditions
            for code in _codes(c)
        ]

        for flag, prefixes in ICD_PREFIXES.items():
            onsets = [d for code, d in codes_with_onset if code.startswith(prefixes)]
            row[flag] = 1 if onsets else 0
            if flag in _YEARS_COL:
                earliest = min((d for d in onsets if d is not None), default=None)
                row[_YEARS_COL[flag]] = _years_between(earliest, t0) if row[flag] else 0.0

        row[LABEL_COL] = int(
            any(code.startswith(CKD_STAGE3PLUS_PREFIXES) for code, _ in codes_with_onset)
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # Guarantee the full contract even if a practice happens to document nothing in some category.
    for col in (*BINARY_FLAG_COLS, *YEARS_SINCE_COLS):
        if col not in df.columns:
            df[col] = 0
    return df[[*FEATURE_COLS, LABEL_COL]]
