"""The local Helios FHIR server — the FLIP-IT `data-source=fhir` host.

Immutable doctrine (CLAUDE.md §0.9): every practice's cohort is read from a **Helios FHIR
server** (`HeliosSoftware/hfs`, the consortium server). This module is the sandbox
incarnation of that server, and of the SQL-on-FHIR queries we run against it:

* `ensure_hfs()`    — download/extract the HFS release binary for this platform (or use
                      `$HELIOS_BIN` if you point at an existing binary).
* `serve()`         — start `hfs` on 127.0.0.1:8080 with a project-local SQLite store.
* `build_seed()`    — convert the committed synthetic cohort (`data/synthetic_ckd_data.csv`)
                      into deterministic FHIR R4 resources (Patient / Encounter / Condition /
                      Observation) that reproduce the canonical `extract_features.sql` contract.
* `seed_server()`   — PUT those resources into the running Helios server.
* `extract()`       — run `data/fhir_loader` (the full landmark-contract query) against Helios
                      and verify the round-trip against the seed source.
* `sql_features()`  — the SQL-on-FHIR queries: HL7 SQL-on-FHIR ViewDefinitions executed by
                      Helios' own `pysof` engine over the very resources Helios is hosting.

One command from the repo root:
    uv run ckd-helios                 # binary -> serve -> seed -> extract proof (keeps server up)
    uv run ckd-helios serve           # start the server only (blocking)
    uv run ckd-helios seed            # (re)build + PUT the synthetic FHIR resources
    uv run ckd-helios extract         # run the landmark extraction against the running server
    uv run ckd-helios sql             # run the SQL-on-FHIR ViewDefinitions (pysof) -> CSVs

Determinism (CLAUDE.md rule 6): one global seed (`SEED`) and the source row index drive every
synthetic value. The fields the synthetic CSV actually carries — age, the comorbidity flags,
years-since-first-diagnosis and, above all, the `ckd_stage3plus` label (emitted as a confirmed
N18.* condition with onset inside the outcome window) — are preserved verbatim. The fields the
schema does not carry (sex, eGFR/HbA1c values, encounter dates) are synthesised deterministically
so the canonical contract genuinely round-trips through `data/fhir_loader.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 42
HELIOS_VERSION = "0.2.1"
HOST = "127.0.0.1"
DEFAULT_PORT = 8080  # Helios owns 8080 now; the demo webapp moved to 9000.
LANDMARK_DAYS = 365  # Stichtag = today - 365d, mirroring data/fhir_loader._LANDMARK_DAYS

_REPO = Path(__file__).resolve().parent.parent
_HELIOS_DIR = _REPO / "data" / "helios"
_BIN_DIR = _HELIOS_DIR / "bin"
_DATA_DIR = _HELIOS_DIR / "data"
_SEED_DIR = _HELIOS_DIR / "seed"
_OUT_DIR = _HELIOS_DIR / "out"
_LOG_FILE = _HELIOS_DIR / "server.log"
_HFS_BIN = _BIN_DIR / "hfs"
_SEED_NDJSON = _SEED_DIR / "fhir_resources.ndjson"

DEFAULT_SOURCE = _REPO / "data" / "synthetic_ckd_data.csv"

# ICD-10-GM codes: one representative code per synthetic flag. Prefixes are what the contract
# matches (data/fhir_loader ICD_PREFIXES / extract_features.sql §5), so any code in range works.
_COND_CODE = {
    "dm": "E11.90",          # diabetes (dx_diabetes)
    "aht": "I10.90",         # hypertension (dx_hypertonie)
    "cvd_khk": "I25.10",     # KHK (dx_khk)
    "cvd_hi": "I50.01",      # heart failure (dx_herzinsuffizienz)
    "adipositas": "E66.90",  # not part of the canonical frame; hosted for completeness
    "hyperurikaemie": "M10.09",
}
_CKD_CODE = "N18.30"

_ICD_SYSTEM = "http://hl7.org/fhir/sid/icd-10-gm"
_LOINC_EGFR = "62238-1"  # eGFR, CKD-EPI 2021
_LOINC_HBA1C = "4548-4"  # HbA1c %

# Encounter placed 400 days before t0 — comfortably inside the ">= 52 weeks" inclusion window
# without colliding with t0 itself.
_EARLY_CONTACT_DAYS = 400
# Lab observation offsets (days before t0); three values so egfr/hba1c mean-of-3 is defined.
_LAB_OFFSETS = (400, 300, 180)
# N18 onset inside the outcome window [t0, t0+365d) so the label round-trips as ICD-incidence.
_WINDOW_ONSET_DAYS = 100


# ── Deterministic pseudo-random helpers -------------------------------------------------------

def _digest(*parts: object) -> bytes:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.digest()


def _rand(*parts: object, lo: float = 0.0, hi: float = 1.0) -> float:
    u = int.from_bytes(_digest(*parts)[:8], "big") / (1 << 64)
    return lo + u * (hi - lo)


def _birthdate(t0: date, age: int) -> str:
    # Fixed month/day avoids a leap-day birthday when t0 is in late February.
    return date(t0.year - age, 6, 15).isoformat()


# ── HFS binary ---------------------------------------------------------------------------------

def _target() -> str:
    syst = platform.system().lower()
    arch = platform.machine().lower()
    if syst == "darwin":
        if arch != "arm64":
            raise SystemExit(
                "Helios publishes no macOS Intel binary; run the ghcr.io/heliossoftware/hfs "
                "container instead, or set $HELIOS_BIN to a working hfs binary."
            )
        return "aarch64-apple-darwin"
    if syst == "linux":
        if arch == "x86_64":
            return "x86_64-unknown-linux-gnu"
        if arch in ("aarch64", "arm64"):
            return "aarch64-unknown-linux-gnu"
    raise SystemExit(
        f"No prebuilt Helios HFS binary for {platform.system()}/{platform.machine()}. "
        "Run the ghcr.io/heliossoftware/hfs container instead, or set $HELIOS_BIN."
    )


def ensure_hfs() -> Path:
    """Return the path to an hfs binary, downloading the release build if needed.

    Set $HELIOS_BIN to bypass the download (in CI or when the binary is installed elsewhere).
    """
    override = os.environ.get("HELIOS_BIN")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise SystemExit(f"$HELIOS_BIN points at a non-existent file: {path}")
        return path
    if _HFS_BIN.is_file() and os.access(_HFS_BIN, os.X_OK):
        return _HFS_BIN
    _BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = _target()
    url = (
        f"https://github.com/HeliosSoftware/hfs/releases/download/v{HELIOS_VERSION}/"
        f"hfs-{HELIOS_VERSION}-{target}.tar.gz"
    )
    archive = _BIN_DIR / f"hfs-{HELIOS_VERSION}-{target}.tar.gz"
    print(f"downloading Helios HFS {HELIOS_VERSION} ({target}) …")
    urllib.request.urlretrieve(url, archive)  # noqa: S310 - pinned upstream URL
    with tarfile.open(archive) as tf:
        tf.extractall(_BIN_DIR, filter="data")  # noqa: S202 - pinned upstream release
    archive.unlink()
    binaries = list(_BIN_DIR.rglob("hfs")) + list(_BIN_DIR.rglob("hfs.exe"))
    if not binaries:
        raise SystemExit(f"no hfs binary found inside {archive.name}")
    binary = binaries[0]
    if binary.resolve() != _HFS_BIN.resolve():
        shutil.copy(binary, _HFS_BIN)
    _HFS_BIN.chmod(_HFS_BIN.stat().st_mode | 0o100)
    # The release tarball bundles companion tools (sof-cli/server, fhirpath-*, hts, terminology…)
    # that this repo never invokes and that added ~2.2 GB per setup. Keep only hfs.
    for entry in _BIN_DIR.iterdir():
        if entry.name in ("hfs", "hfs.exe") or entry.suffix == ".wgt":
            continue
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    # Downloading via urllib sets no quarantine attribute, but clear it defensively on macOS.
    if platform.system().lower() == "darwin":
        subprocess.run(["xattr", "-cr", str(_HFS_BIN)], check=False)
    shutil.copy(binary, _HFS_BIN)
    print(f"Helios HFS binary ready at {_HFS_BIN}")
    return _HFS_BIN


# ── Server lifecycle ---------------------------------------------------------------------------

def _get_json(url: str, timeout: float = 5.0) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/fhir+json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - loopback-only callers
        return json.loads(resp.read().decode("utf-8"))


def _wait_ready(base: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            _get_json(f"{base}/metadata", timeout=3.0)
            return
        except Exception as exc:  # noqa: BLE001 - server still starting
            last = exc
            time.sleep(0.5)
    raise TimeoutError(f"Helios did not become ready at {base} (last error: {last})")


def _ensure_search_params() -> Path:
    """Provision HFS's R4 SearchParameter spec (the server vendor's own file).

    The shipped HFS binary falls back to a 9-parameter registry unless its
    `search-parameters-r4.json` config exists, which would break the loader's
    `Observation?category=laboratory` search. We fetch the same spec file the HFS
    release ships, pinned to the matching tag, so standard R4 search parameters
    (Observation.category/code/date, Condition.code, Encounter.subject, ...) work.
    """
    target = _DATA_DIR / "search-parameters-r4.json"
    if target.is_file():
        return target
    urls = [
        (
            f"https://raw.githubusercontent.com/HeliosSoftware/hfs/"
            f"v{HELIOS_VERSION}/crates/fhir-gen/resources/R4/search-parameters.json"
        ),
        (
            "https://raw.githubusercontent.com/HeliosSoftware/hfs/main/"
            "crates/fhir-gen/resources/R4/search-parameters.json"
        ),
    ]
    last: Exception | None = None
    for url in urls:
        try:
            urllib.request.urlretrieve(url, target)  # noqa: S310 - pinned upstream config
            print(f"provisioned HFS R4 SearchParameter spec -> {target}")
            return target
        except Exception as exc:  # noqa: BLE001
            last = exc
    raise SystemExit(f"could not fetch the HFS R4 search-parameters spec: {last}")


def serve(*, port: int = DEFAULT_PORT, reset: bool = False) -> subprocess.Popen:
    """Start the Helios HFS server on 127.0.0.1:<port> and wait until /metadata responds."""
    binary = ensure_hfs()
    if reset and _DATA_DIR.exists():
        shutil.rmtree(_DATA_DIR)
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_search_params()
    env = {
        **os.environ,
        "HFS_SERVER_HOST": HOST,
        "HFS_SERVER_PORT": str(port),
        "HFS_DATA_DIR": str(_DATA_DIR),
        # SQLite lives HERE, never at {cwd}/fhir.db (HFS's default → repo-root litter that is
        # trivially destroyed). Both files are under data/helios/ and gitignored as a unit.
        "HFS_DATABASE_URL": str(_DATA_DIR / "fhir.db"),
        "HFS_BASE_URL": f"http://{HOST}:{port}",
        "HFS_MAX_BODY_SIZE": "104857600",
        "HFS_COMPOSITE_SYNC_MODE": "synchronous",  # read-your-write search after each batch
        "HFS_LOG_LEVEL": "info",
    }
    _HELIOS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(_LOG_FILE, "a", encoding="utf-8")  # noqa: SIM115 - held for the server lifetime
    proc = subprocess.Popen(
        [str(binary)], cwd=str(_REPO), env=env, stdout=log, stderr=subprocess.STDOUT
    )
    base = f"http://{HOST}:{port}"
    print(f"starting Helios FHIR server: {base}  (data: {_DATA_DIR})")
    _wait_ready(base)
    print(f"Helios ready at {base}  [log: {_LOG_FILE}]")
    return proc


# ── Seed: synthetic CKD CSV -> FHIR R4 --------------------------------------------------------

def _years_delta(row: pd.Series, column: str, t0: date) -> str | None:
    """Onset date from a `years_since_*` column; 0/absent -> undated (SQL 'tage_seit NULL')."""
    years = float(row[column]) if column in row.index else 0.0
    if years > 0:
        return (t0 - timedelta(days=round(years * 365.25))).isoformat()
    return None


def _condition(rid: str, pid: str, code: str, onset: str | None) -> dict:
    cond: dict = {
        "resourceType": "Condition",
        "id": rid,
        "clinicalStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                        "code": "active"}],
        },
        "verificationStatus": {
            "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                        "code": "confirmed"}],
        },
        "code": {"coding": [{"system": _ICD_SYSTEM, "code": code}]},
        "subject": {"reference": f"Patient/{pid}"},
    }
    if onset is not None:  # undated = "gesichert, nicht datiert": flag set, tage_seit NULL (SQL §5)
        cond["onsetDateTime"] = onset
        cond["recordedDate"] = onset
    return cond


def _observation(rid: str, pid: str, loinc: str, value: float, unit: str, day: date) -> dict:
    return {
        "resourceType": "Observation",
        "id": rid,
        "status": "final",
        "category": [
            {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category",
                         "code": "laboratory"}]},
        ],
        "code": {"coding": [{"system": "http://loinc.org", "code": loinc}]},
        "subject": {"reference": f"Patient/{pid}"},
        "effectiveDateTime": day.isoformat(),
        "valueQuantity": {"value": round(value, 1), "unit": unit},
    }


def build_seed(source: Path, t0: date) -> list[dict]:
    """Convert the committed synthetic cohort into deterministic FHIR R4 resources."""
    df = pd.read_csv(source)
    resources: list[dict] = []

    for i, row in df.iterrows():
        pid = f"pat-{i + 1:05d}"
        age = int(row["age_years"])
        gender = "female" if _rand(("gender", i)) < 0.5 else "male"

        resources.append({
            "resourceType": "Patient",
            "id": pid,
            "gender": gender,
            "birthDate": _birthdate(t0, age),
        })

        resources.append({
            "resourceType": "Encounter",
            "id": f"enc-{i + 1:05d}",
            "status": "finished",
            "subject": {"reference": f"Patient/{pid}"},
            "period": {"start": (t0 - timedelta(days=_EARLY_CONTACT_DAYS)).isoformat()},
        })

        conditions: list[tuple[str, str | None]] = []
        if int(row["dx_diabetes"]):
            conditions.append((_COND_CODE["dm"], _years_delta(row, "years_since_diabetes_dx", t0)))
        if int(row["dx_hypertonie"]):
            conditions.append((_COND_CODE["aht"], _years_delta(row, "years_since_hypertonie_dx", t0)))
        if int(row["dx_khk"]):
            conditions.append((_COND_CODE["cvd_khk"], _years_delta(row, "years_since_khk_dx", t0)))
        if int(row["dx_herzinsuffizienz"]):
            # no dedicated years column in the schema -> deterministic default onset
            conditions.append((_COND_CODE["cvd_hi"], (t0 - timedelta(days=365)).isoformat()))
        if int(row["dx_adipositas"]):
            conditions.append((_COND_CODE["adipositas"], None))
        if int(row["dx_hyperurikaemie"]):
            conditions.append((_COND_CODE["hyperurikaemie"], None))
        if int(row["ckd_stage3plus"]):
            conditions.append((_CKD_CODE, (t0 + timedelta(days=_WINDOW_ONSET_DAYS)).isoformat()))
        for j, (code, onset) in enumerate(conditions):
            resources.append(_condition(f"cond-{i + 1:05d}-{j}", pid, code, onset))

        has_diabetes = int(row["dx_diabetes"])
        # eGFR is label-CORRELATED, not label-deterministic: incident-CKD cases get a wide band
        # whose upper tail reaches >60 (early-stage CKD can read above the threshold), while
        # controls are floored at 61. That keeps the two arms overlapping in the FEATURES (so the
        # task is learnable, not trivially 1.0) without ever letting a control satisfy the
        # label's eGFR-persistence clause (2 readings <= 60) — the label stays exactly the
        # N18-anchored seed rate. Identical seeds -> identical values (rule 6).
        is_case = int(row["ckd_stage3plus"]) == 1
        # cases: medium base + wide noise (upper tail crosses 60 — early-stage CKD can read above
        # the threshold); controls: floored at 61 so the label's eGFR-persistence clause (2 reads
        # <= 60) can never fire for a control. eGFR stays strongly predictive but not dominant —
        # the CSV synthetic data never had a feature that was a label lookup, and neither should the
        # FHIR path (eGFR-alone discrimination ~0.8, full model ~0.9, not 1.0).
        egfr_base, egfr_epread, egfr_floor = (58.0, 110.0, 15.0) if is_case else (82.0, 80.0, 61.0)
        for j, offset in enumerate(_LAB_OFFSETS):
            u = (_rand(("egfr", i, j)) + _rand(("egfr", i, j, "b"))) / 2  # ~triangular, mean 0.5
            egfr = max(egfr_floor, egfr_base + (u - 0.5) * egfr_epread)
            resources.append(_observation(
                f"obs-egfr-{i + 1:05d}-{j}", pid, _LOINC_EGFR, round(egfr, 1),
                "mL/min/1.73m^2", t0 - timedelta(days=offset),
            ))
        hba1c_base = 5.4 + (0.9 if has_diabetes else 0.0)
        for j, offset in enumerate(_LAB_OFFSETS):
            hba1c = hba1c_base + (_rand(("hba1c", i, j)) - 0.5) * 1.2
            resources.append(_observation(
                f"obs-hba1c-{i + 1:05d}-{j}", pid, _LOINC_HBA1C, hba1c,
                "%", t0 - timedelta(days=offset),
            ))

    _SEED_DIR.mkdir(parents=True, exist_ok=True)
    _SEED_NDJSON.write_text(
        "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in resources),
        encoding="utf-8",
    )
    kinds: dict[str, int] = {}
    for r in resources:
        kinds[r["resourceType"]] = kinds.get(r["resourceType"], 0) + 1
    print(f"seed built: {len(resources)} FHIR resources from {source.name} ({t0})")
    print("  " + ", ".join(f"{k}×{v}" for k, v in sorted(kinds.items())))
    return resources


def load_seed() -> list[dict]:
    if not _SEED_NDJSON.is_file():
        raise SystemExit(
            f"no seed at {_SEED_NDJSON} — run `ckd-helios seed` (or `ckd-helios up`) first"
        )
    return [json.loads(line) for line in _SEED_NDJSON.read_text(encoding="utf-8").splitlines()]


def seed_server(resources: list[dict], *, base: str) -> None:
    """Idempotently PUT every seeded resource (unconditional update: re-seeding replaces)."""
    failures: list[str] = []
    for idx, resource in enumerate(resources):
        rtype, rid = resource["resourceType"], resource["id"]
        url = f"{base}/{rtype}/{rid}"
        body = json.dumps(resource).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="PUT",
            headers={"Content-Type": "application/fhir+json",
                     "Accept": "application/fhir+json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60.0) as resp:  # noqa: S310 - loopback only
                _ = resp.status
        except urllib.error.HTTPError as exc:
            failures.append(f"{rtype}/{rid}: HTTP {exc.code} {exc.read()[:200]!r}")
        except urllib.error.URLError as exc:
            failures.append(f"{rtype}/{rid}: {exc}")
        if (idx + 1) % 2000 == 0:
            print(f"  seeded {idx + 1}/{len(resources)} resources …")
    print(f"seeded {len(resources) - len(failures)}/{len(resources)} resources into {base}")
    if failures:
        sample = failures[:5]
        raise SystemExit(f"{len(failures)} resources failed to write; first errors: {sample}")


# ── Extraction (canonical landmark contract) ---------------------------------------------------

def extract(*, base: str, t0: date, source: Path, write_csv: Path | None = None) -> pd.DataFrame:
    """Run data/fhir_loader against Helios and verify the seed round-trips."""
    from . import fhir_loader

    df, stats = fhir_loader._extract(  # noqa: SLF001 - intentional internal reuse (same package)
        base, page_size=500, timeout=60.0, stichtag=t0, pseudonym_salt=None
    )
    print(f"Stichtag t0 = {t0}  (Zielfenster [{t0}, {t0 + timedelta(days=365)}))")
    for key, value in stats.items():
        print(f"  {key:<24} {value}")
    print(f"included                    {len(df)}")

    src = pd.read_csv(source)
    expected_rate = float(src["ckd_stage3plus"].mean())
    observed_rate = float(df[fhir_loader.CANONICAL_LABEL_COL].mean())
    print("  ── round-trip vs synthetic seed ──")
    print(f"  included rows              {len(df)} (source rows {len(src)})")
    print(f"  ckd_incident rate          {observed_rate:.4f} (source stage3plus {expected_rate:.4f})")
    for col in fhir_loader.CANONICAL_LAB_COLS:
        print(f"  {col:<27} missing in {df[col].isna().mean():.1%}")
    for col in fhir_loader.CANONICAL_TAGE_COLS:
        print(f"  {col:<27} missing in {df[col].isna().mean():.1%}")

    if len(df) != len(src):
        raise SystemExit("ROUND-TRIP FAILED: extracted cohort size != seed cohort size")
    if abs(observed_rate - expected_rate) > 1e-9:
        raise SystemExit(
            f"ROUND-TRIP FAILED: ckd_incident rate {observed_rate:.4f} != "
            f"source ckd_stage3plus rate {expected_rate:.4f}"
        )
    print("  ROUND-TRIP OK: every source row extracted; labels match the seed exactly.")
    if write_csv:
        df.to_csv(write_csv, index=False)
        print(f"de-identified extract written to {write_csv}")
    return df


# ── SQL-on-FHIR: ViewDefinitions run by Helios pysof ------------------------------------------

def view_definitions() -> dict[str, dict]:
    """HL7 SQL-on-FHIR ViewDefinitions: the SQL-on-FHIR query form for the seeded resources."""
    return {
        "patient": {
            "resourceType": "ViewDefinition",
            "id": "ckd-patient",
            "name": "CkdPatient",
            "status": "active",
            "resource": "Patient",
            "select": [{"column": [
                {"name": "id", "path": "getResourceKey()"},
                {"name": "gender", "path": "gender"},
                {"name": "birth_date", "path": "birthDate"},
            ]}],
        },
        "condition": {
            "resourceType": "ViewDefinition",
            "id": "ckd-condition",
            "name": "CkdCondition",
            "status": "active",
            "resource": "Condition",
            "select": [{"column": [
                {"name": "id", "path": "getResourceKey()"},
                {"name": "patient_id", "path": "subject.reference"},
                {"name": "icd10_gm", "path": "code.coding.where(system='http://hl7.org/fhir/sid/icd-10-gm').code.first()"},
                {"name": "dx_date", "path": "recordedDate"},
            ]}],
        },
        "observation": {
            "resourceType": "ViewDefinition",
            "id": "ckd-observation",
            "name": "CkdObservation",
            "status": "active",
            "resource": "Observation",
            "select": [{"column": [
                {"name": "id", "path": "getResourceKey()"},
                {"name": "patient_id", "path": "subject.reference"},
                {"name": "loinc", "path": "code.coding.where(system='http://loinc.org').code.first()"},
                {"name": "value", "path": "valueQuantity.value"},
                {"name": "unit", "path": "valueQuantity.unit"},
                {"name": "effective_date", "path": "effectiveDateTime"},
            ]}],
        },
    }


def sql_features() -> int:
    """Run the SQL-on-FHIR ViewDefinitions over the seeded resources via Helios' pysof engine."""
    try:
        import pysof  # noqa: PLC0415 - optional dep, fail with guidance
    except ImportError:
        print(
            "pysof (Helios SQL-on-FHIR) is not installed. In a healthy checkout run "
            "`uv add pysof`; in this workspace (uv cannot resolve the repo path) run: "
            "uv pip install --python .venv/bin/python pysof==0.2.1"
        )
        return 1

    resources = load_seed()
    bundle: dict = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [{"resource": r} for r in resources],
    }
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, view in view_definitions().items():
        ok = pysof.validate_view_definition(view)
        if not ok:
            print(f"  view {name}: FAILED validation")
            continue
        rows = pysof.run_view_definition(view, bundle, "csv").decode("utf-8")
        path = _OUT_DIR / f"{name}.csv"
        path.write_text(rows, encoding="utf-8")
        print(f"  view {name}: {rows.count(chr(10))} rows -> {path.name}")
    print("SQL-on-FHIR views written to data/helios/out/")
    return 0


# ── CLI -----------------------------------------------------------------------------------------

def _base(port: int) -> str:
    return f"http://{HOST}:{port}"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ckd-helios",
        description=(
            "Helios FHIR server host + synthetic seed for the FLIP-IT sandbox (CLAUDE.md §0.9). "
            "builds from Helios, queries are SQL-on-FHIR."
        ),
    )
    parser.add_argument(
        "cmd", nargs="?", default="up",
        choices=("up", "serve", "seed", "extract", "sql"),
        help="up = ensure binary + serve + seed + extract (default); others are one stage each",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="HFS listen port")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="synthetic cohort CSV to seed (default: data/synthetic_ckd_data.csv)")
    parser.add_argument("--t0", type=date.fromisoformat, default=None,
                        help="Stichtag (YYYY-MM-DD); default today - 365d, like extract_features.sql")
    parser.add_argument("--reset", action="store_true",
                        help="wipe the Helios data dir before serving (fresh SQLite)")
    parser.add_argument("--write-csv", type=Path, default=None,
                        help="extract: also write the de-identified landmark CSV here")
    args = parser.parse_args()

    t0 = args.t0 or (date.today() - timedelta(days=LANDMARK_DAYS))
    base = _base(args.port)

    if args.cmd == "serve":
        proc = serve(port=args.port, reset=args.reset)
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return

    if args.cmd == "seed":
        resources = build_seed(args.source, t0)
        _wait_ready(base)
        seed_server(resources, base=base)
        return

    if args.cmd == "extract":
        _wait_ready(base)
        extract(base=base, t0=t0, source=args.source, write_csv=args.write_csv)
        return

    if args.cmd == "sql":
        sys.exit(sql_features())

    # up: full chain, then keep the server up until Ctrl-C.
    resources = build_seed(args.source, t0)
    proc = serve(port=args.port, reset=args.reset)
    try:
        seed_server(resources, base=base)
        extract(base=base, t0=t0, source=args.source, write_csv=args.write_csv)
        print("\nserver still running at " + base + " — Ctrl-C to stop")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
