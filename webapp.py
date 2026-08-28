"""Local physician demo webapp — input patient details, get the federated model's CKD risk score.

A deliberately small, stdlib-only surface (no web framework dependency) that demonstrates the
deployment story from docs/DEPLOYMENT.md §use: the global model trained across the federation is
exported once (``uv run ckd-export-model``) and served on the practice's own machine. No patient
identifier is ever requested or stored — the form holds features only, and nothing is persisted.

    uv run ckd-export-model          # trains via Flower FedAvg, writes models/global_model.json
    uv run ckd-web --port 8080       # serves http://127.0.0.1:8080

⚠️ Research demo on synthetic data — not a medical device, not for clinical decisions.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from model_artifact import ARTIFACT_PATH, Scorer

# Human label + ICD hint + input kind per feature, in schema order (data.loader.FEATURE_COLS).
_FIELDS: list[tuple[str, str, str, str]] = [
    ("age_years", "Age (years)", "number", ""),
    ("dx_hypertonie", "Hypertension", "checkbox", "ICD I10"),
    ("dx_diabetes", "Diabetes mellitus", "checkbox", "ICD E10/E11/E13"),
    ("dx_khk", "Coronary heart disease", "checkbox", "ICD I25"),
    ("dx_adipositas", "Obesity", "checkbox", "ICD E66"),
    ("dx_herzinsuffizienz", "Heart failure", "checkbox", "ICD I50/I11.0"),
    ("dx_hyperurikaemie", "Hyperuricemia / gout", "checkbox", "ICD M10"),
    ("years_since_hypertonie_dx", "Years since hypertension diagnosis", "number", "0 if absent"),
    ("years_since_diabetes_dx", "Years since diabetes diagnosis", "number", "0 if absent"),
    ("years_since_khk_dx", "Years since CHD diagnosis", "number", "0 if absent"),
]

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CKD Risk — FLIP-IT federated model (demo)</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 560px; margin: 2rem auto;
         padding: 0 1rem; color: #1a1a1a; }
  h1 { font-size: 1.3rem; }
  label { display: block; margin: 0.55rem 0; }
  input[type=number] { width: 6rem; }
  .hint { color: #666; font-size: 0.8rem; }
  button { margin-top: 1rem; padding: 0.5rem 1.2rem; font-size: 1rem; }
  .risk { font-size: 2.4rem; font-weight: 700; margin: 0.5rem 0; }
  .band-lower { color: #1a7f37; } .band-intermediate { color: #9a6700; }
  .band-elevated { color: #cf222e; }
  .note { background: #fff8c5; border: 1px solid #eac54f; padding: 0.6rem 0.8rem;
          font-size: 0.85rem; margin-top: 1.5rem; border-radius: 6px; }
  .err { background: #ffebe9; border-color: #ff818266; }
</style>
</head>
<body>
__BODY__
</body>
</html>
"""

_DISCLAIMER = (
    '<div class="note">Demo of the FLIP-IT federated CKD model, trained with Flower across '
    "simulated GP practices on <b>synthetic</b> data. Research prototype — not a medical device, "
    "not for clinical decisions.</div>"
)


def _band(risk: float) -> tuple[str, str]:
    """Illustrative triage band for the demo UI."""
    if risk < 0.20:
        return "lower", "band-lower"
    if risk < 0.50:
        return "intermediate", "band-intermediate"
    return "elevated", "band-elevated"


def _form_page() -> bytes:
    rows = []
    for name, label, kind, hint in _FIELDS:
        hint_html = f' <span class="hint">({hint})</span>' if hint else ""
        if kind == "checkbox":
            rows.append(f'<label><input type="checkbox" name="{name}" value="1"> {label}{hint_html}</label>')
        else:
            default = "0" if name.startswith("years_since") else ""
            rows.append(
                f'<label>{label}: <input type="number" step="any" min="0" max="120" '
                f'name="{name}" value="{default}" required>{hint_html}</label>'
            )
    body = f"""
<h1>CKD risk score — federated model</h1>
<p class="hint">Patient features only. No identifiers are collected or stored.
A years field is ignored when its diagnosis is not ticked.</p>
<form method="post" action="score">
  {''.join(rows)}
  <button type="submit">Score</button>
</form>
{_DISCLAIMER}"""
    return _PAGE.replace("__BODY__", body).encode()


def _result_page(risk: float, inputs: dict[str, float]) -> bytes:
    band, css = _band(risk)
    echo = ", ".join(
        f"{label}: {int(inputs[name])}" if kind == "checkbox" else f"{label}: {inputs[name]:g}"
        for name, label, kind, _ in _FIELDS
        if inputs.get(name)
    )
    body = f"""
<h1>CKD risk score</h1>
<p class="risk {css}">{risk * 100:.1f}%</p>
<p>Estimated P(CKD stage ≥ 3) — illustrative band: <b class="{css}">{band}</b>.</p>
<p class="hint">Inputs: {echo}</p>
<p><a href="./">← Score another patient</a></p>
{_DISCLAIMER}"""
    return _PAGE.replace("__BODY__", body).encode()


def _error_page(message: str) -> bytes:
    body = f"""
<h1>Invalid input</h1>
<div class="note err">{message}</div>
<p><a href="./">← Back</a></p>"""
    return _PAGE.replace("__BODY__", body).encode()


def _parse_form(params: dict[str, list[str]]) -> dict[str, float]:
    """Raw POST params → feature dict. Raises ValueError with a user-facing message."""
    values: dict[str, float] = {}
    for name, label, kind, _ in _FIELDS:
        if kind == "checkbox":
            values[name] = 1.0 if name in params else 0.0
            continue
        text = (params.get(name) or [""])[0].strip()
        if text == "" and name.startswith("years_since_"):
            values[name] = 0.0  # absent years_since encodes "diagnosis absent" (CLAUDE.md §3)
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f'"{label}" must be a number (got "{text}").') from None
        if not (0.0 <= value <= 120.0):
            raise ValueError(f'"{label}" must be between 0 and 120 (got {value:g}).')
        values[name] = value
    return values


def make_handler(scorer: Scorer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ckd-fl-demo"

        def _send(self, body: bytes, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib hook)
            if urlparse(self.path).path == "/":
                self._send(_form_page())
            else:
                self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802 (stdlib hook)
            if urlparse(self.path).path != "/score":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            params = parse_qs(self.rfile.read(length).decode())
            try:
                inputs = _parse_form(params)
            except ValueError as exc:
                self._send(_error_page(str(exc)), status=400)
                return
            self._send(_result_page(scorer.score_one(inputs), inputs))

        def log_message(self, fmt: str, *args) -> None:  # keep demo output quiet
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Local physician demo webapp for the federated CKD model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model", type=Path, default=ARTIFACT_PATH, help="exported model artifact")
    args = parser.parse_args()

    scorer = Scorer.from_file(args.model)  # exits loudly if `ckd-export-model` has not run
    server = ThreadingHTTPServer((args.host, args.port), make_handler(scorer))
    print(f"Serving CKD risk demo on http://{args.host}:{args.port}  (model: {args.model})")
    server.serve_forever()


if __name__ == "__main__":
    main()
