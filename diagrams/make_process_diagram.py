"""Regenerate the FLIP-IT end-to-end process diagram (drawio XML + PNG).

One layout spec drives both artifacts so they cannot drift apart:

    uv run --extra notebook python diagrams/make_process_diagram.py

Outputs (next to this file, alongside Flipit-privacy.xml / .png):
    Flipit-process.xml   — editable in diagrams.net (drawio)
    Flipit-process.png   — rendered copy for docs / slides

Visual language matches Flipit-privacy.xml:
    green  = inside the practice trust boundary      (fill #d5e8d4 / stroke #82b366)
    blue   = central consortium / SuperLink side     (fill #dae8fc / stroke #6c8ebf)
    yellow = review / decision gate                  (fill #fff2cc / stroke #d6b656)
    gray   = legend / notes                          (fill #f5f5f5 / stroke #666666)
    solid arrow  = model weights / model updates
    dashed arrow = planning metadata (keys, census N, config) — never patient data
"""

from __future__ import annotations

import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent

CANVAS_W = 1160
COL_L_X, COL_W = 40, 500            # left column: inside the practice
COL_R_X = COL_L_X + COL_W + 80      # right column: central consortium
GAP_X = COL_L_X + COL_W             # 540..620: inter-column gap for stage badges

GREEN = ("#d5e8d4", "#82b366")
BLUE = ("#dae8fc", "#6c8ebf")
YELLOW = ("#fff2cc", "#d6b656")
GRAY = ("#f5f5f5", "#666666")

WRAP = 62  # body chars per line before a manual wrap


def wrap_body(bullets: list[str]) -> list[str]:
    out: list[str] = []
    for b in bullets:
        lines = textwrap.wrap(b, WRAP)
        out.append(f"• {lines[0]}")
        out.extend(f"  {line}" for line in lines[1:])
    return out


class Node:
    def __init__(self, nid: str, x: int, y: int, w: int, title: str,
                 bullets: list[str], colors: tuple[str, str]):
        self.id, self.x, self.y, self.w = nid, x, y, w
        self.title = title
        self.body_lines = wrap_body(bullets)
        self.h = 40 + 17 * len(self.body_lines) + 10
        self.fill, self.stroke = colors
        # full-width bars (strip/legend) are plain rectangles
        self.rounded = colors != GRAY or w < 1000


class Edge:
    def __init__(self, eid: str, src: str, dst: str, label: str, dashed: bool,
                 exit: tuple[float, float], entry: tuple[float, float],
                 elbows: list[tuple[int, int]] | None = None):
        self.id, self.src, self.dst, self.label, self.dashed = eid, src, dst, label, dashed
        self.exit, self.entry = exit, entry
        self.elbows = elbows or []  # extra waypoints for the PNG renderer


# --------------------------------------------------------------------------
# Content — the whole FLIP-IT pipeline, clinician's-eye view
# --------------------------------------------------------------------------

TITLE = "FLIP-IT End-to-End Process — From Practice Onboarding to the Consultation"
SUBTITLE = ("The model travels to the data — patient records never leave the practice. "
            "One consortium, 25 GP practices, one shared CKD risk model.")
HEADER_L = "INSIDE THE PRACTICE — patient data never leaves"
HEADER_R = "CENTRAL CONSORTIUM — one SuperLink (docport / IKIM KITE)"

NODES: dict[str, Node] = {}


def add(nid: str, x: int, y: int, w: int, title: str, bullets: list[str],
        colors: tuple[str, str]) -> Node:
    NODES[nid] = Node(nid, x, y, w, title, bullets, colors)
    return NODES[nid]


# Row 1 — onboarding
L0 = add("L0", COL_L_X, 150, COL_W, "ONBOARDING — one time, with practice IT", [
    "Practice generates a P-384 key pair — the private key never leaves the machine",
    "Practice IT installs one Flower SuperNode and points it at the practice's own Helios FHIR server",
    "Cohort validated locally: ckd-fhir-extract prints exclusion accounting; no rows leave",
], GREEN)
R0 = add("R0", COL_R_X, 150, COL_W, "CONSORTIUM ADMIN", [
    "Runs the SuperLink on central infrastructure (TLS + key-based admission)",
    "Registers each practice's public key (flwr supernode register)",
    "Only registered practices can ever join a run — anyone can be withdrawn",
], BLUE)

# Row 2 — data + round zero
y = L0.y + L0.h + 40
L1 = add("L1", COL_L_X, y, COL_W, "DATA PIPELINE — runs inside the practice", [
    "Tomedo practice software → nightly export → the practice's PostgreSQL / FHIR server",
    "extract_features.sql builds the canonical 16-column feature contract",
    "Pseudonymised extract: no names, no identifiers, no patientid column",
], GREEN)
R1 = add("R1", COL_R_X, y, COL_W, "ROUND ZERO — rule-based server agent", [
    "Census: each practice reports only its patient count N — never patient data",
    "Enforces the agreed privacy policy: one global target (ε, δ) for everyone",
    "Inverts the privacy accountant per clinic → its own (batch size, noise σ) plan",
    "Plan dispatched to each SuperNode inside the ConfigRecord",
], BLUE)

# Row 3 — federated rounds
y = L1.y + max(L1.h, R1.h) + 40
L2 = add("L2", COL_L_X, y, COL_W, "FEDERATED TRAINING — repeats ≈ 10 rounds", [
    "SuperNode receives the current global model",
    "Trains locally on its own cohort (DP-SGD with its own (batch, σ) plan)",
    "Returns a SecAgg+-masked model update — never raw weights",
], GREEN)
R2 = add("R2", COL_R_X, y, COL_W, "SUPERLINK AGGREGATION — per round", [
    "SecAgg+ unmasking opens only the weighted SUM — individual updates stay hidden",
    "FedAvg over the sum → the new global model for the next round",
    "Every round logs global AND worst-practice AUROC / sensitivity — a weak practice cannot hide",
], BLUE)

# Row 4 — gate (centred)
y = L2.y + max(L2.h, R2.h) + 40
G = add("G", 180, y, 800, "GOVERNANCE GATE — nothing ships until this passes", [
    "Global and worst-practice AUROC / sensitivity clear the agreed floor",
    "Membership-inference leakage audit re-run on the final model",
    "Consortium clinical review (docport / IKIM / RUB) + legal review (Jorzig & Partner) → release decision",
], YELLOW)

# Row 5 — back to the practice (+ oversight note)
y = G.y + G.h + 40
L3 = add("L3", COL_L_X, y, COL_W, "MODEL BACK IN THE PRACTICE", [
    "Approved model exported as a JSON artifact, paired with the practice-local scaler",
    "Delivered onto the practice machine — scoring runs fully on-site",
    "No server connection needed at consultation time",
], GREEN)
R4 = add("R4", COL_R_X, y, COL_W, "CONSORTIUM OVERSIGHT — continuous", [
    "Every run logged at the SuperLink: rounds, ε spend, dual-level metrics",
    "A practice can pause or leave at any time — its key is unregistered and no data was ever held centrally",
    "Later rounds refresh the model as cohorts evolve",
], BLUE)

# Row 6 — point of care
y = L3.y + max(L3.h, R4.h) + 40
L4 = add("L4", COL_L_X, y, COL_W, "AT THE POINT OF CARE", [
    "Clinician opens the local form, enters routine features — no identifiers collected or stored",
    "Gets P(CKD stage ≥ 3) with the practice's calibrated flag",
    "High-risk patients offered early diagnostics and guideline follow-up",
    "The score informs the consultation — the doctor decides",
], GREEN)

# Bottom strips
y = L4.y + L4.h + 40
STRIP = add("S", 40, y, 1080, "WHAT THE CLINICIAN ACTUALLY DOES", [
    "One-time: agree to join, hand IT the install sheet. During training: nothing — no data entry, no uploads, no change to daily work.",
    "After go-live: the CKD risk score shows up as routine decision support in the consultation.",
], GREEN)

y = STRIP.y + STRIP.h + 30
LEGEND = add("X", 40, y, 1080, "LEGEND", [
    "Solid arrow: model weights / model updates  ·  Dashed arrow: planning metadata (keys, census N, config) — never patient data",
    "Green: inside the practice trust boundary  ·  Blue: central consortium  ·  Yellow: review / decision gate",
], GRAY)

CANVAS_H = LEGEND.y + LEGEND.h + 30

EDGES = [
    Edge("e0", "L0", "R0", "public key only (P-384)", True, (1, 0.5), (0, 0.5)),
    Edge("e1", "L0", "L1", "", True, (0.5, 1), (0.5, 0)),
    Edge("e2", "R0", "R1", "practice admitted", True, (0.5, 1), (0.5, 0)),
    Edge("e3", "L1", "R1", "census: headcount N only", True, (1, 0.5), (0, 0.5)),
    Edge("e4", "R1", "L2", "per-clinic (batch, σ) plan in ConfigRecord", True,
         (0, 0.9), (0.75, 0), elbows=[(COL_R_X, 0), (415, 0)]),  # y=0 → src bottom + 20
    Edge("e5", "L2", "R2", "SecAgg+ masked update", False, (1, 0.3), (0, 0.3)),
    Edge("e6", "R2", "L2", "new global weights", False, (0, 0.7), (1, 0.7)),
    Edge("e7", "R2", "G", "candidate final model", False, (0.5, 1), (0.6, 0),
         elbows=[(870, 0), (660, 0)]),  # y=0 → src bottom + 20
    Edge("e8", "G", "L3", "approved model artifact", False, (0, 0.5), (0.6, 0),
         elbows=[(140, int(G.y + G.h / 2)), (340, int(G.y + G.h / 2))]),
    Edge("e9", "L3", "L4", "", False, (0.5, 1), (0.5, 0)),
    Edge("e10", "G", "R4", "run record logged", True, (0.8, 1), (0.5, 0)),
]

STAGES = [
    ("Stage 1 — join", "L0"), ("Stage 2 — data & census", "L1"),
    ("Stage 3 — federated rounds", "L2"), ("Stage 4 — approval", "G"),
    ("Stage 5 — deploy", "L3"), ("Stage 6 — consult", "L4"),
]

# --------------------------------------------------------------------------
# Sanity checks — fail loudly instead of shipping an overlapping layout
# --------------------------------------------------------------------------


def check_layout() -> None:
    boxes = list(NODES.values())
    for b in boxes:
        assert 0 <= b.x and b.x + b.w <= CANVAS_W, f"{b.id} outside x bounds"
        assert 0 <= b.y and b.y + b.h <= CANVAS_H, f"{b.id} outside y bounds"
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            overlap = (a.x < b.x + b.w and b.x < a.x + a.w
                       and a.y < b.y + b.h and b.y < a.y + a.h)
            assert not overlap, f"{a.id} overlaps {b.id}"
    for e in EDGES:
        assert e.src in NODES and e.dst in NODES, f"{e.id} unknown endpoint"


# --------------------------------------------------------------------------
# drawio XML
# --------------------------------------------------------------------------


def emit_drawio() -> Path:
    root_model = ET.Element("mxfile", host="app.diagrams.net")
    diag = ET.SubElement(root_model, "diagram", name="FLIP-IT Process", id="flipit-process")
    model = ET.SubElement(diag, "mxGraphModel", {
        "dx": "2092", "dy": "1155", "grid": "1", "gridSize": "10", "guides": "1",
        "tooltips": "1", "connect": "1", "arrows": "1", "fold": "1", "page": "1",
        "pageScale": "1", "pageWidth": "850", "pageHeight": "1100",
        "math": "0", "shadow": "0",
    })
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    def text_cell(tid: str, x: int, y: int, w: int, h: int, value: str,
                  size: int = 11, style_flags: int = 0, color: str = "#333333") -> None:
        style = (f"text;html=1;whiteSpace=wrap;strokeColor=none;fillColor=none;"
                 f"align=center;verticalAlign=middle;fontSize={size};fontColor={color};"
                 f"fontStyle={style_flags};rounded=0;")
        cell = ET.SubElement(root, "mxCell", id=tid, parent="1", vertex="1",
                             style=style, value=value)
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w),
                      height=str(h), **{"as": "geometry"})

    text_cell("t0", 20, 20, 1120, 34, TITLE, size=14, style_flags=1, color="#000000")
    text_cell("t1", 20, 56, 1120, 26, SUBTITLE, size=11, style_flags=2)
    text_cell("t2", COL_L_X, 100, COL_W, 28, HEADER_L, size=12, style_flags=1, color="#2f5b32")
    text_cell("t3", COL_R_X, 100, COL_W, 28, HEADER_R, size=12, style_flags=1, color="#36476b")

    for label, ref in STAGES:
        n = NODES[ref]
        text_cell(f"sg-{ref}", GAP_X - 30, n.y - 24, 140, 20, label, size=10,
                  style_flags=2, color="#666666")

    for n in NODES.values():
        value = f"<b>{n.title}</b><br>" + \
                "<br>".join(line for line in n.body_lines)
        style = ("rounded=1;whiteSpace=wrap;html=1;align=center;verticalAlign=top;"
                 f"spacingTop=6;fontSize=11;fillColor={n.fill};strokeColor={n.stroke};")
        cell = ET.SubElement(root, "mxCell", id=n.id, parent="1", vertex="1",
                             style=style, value=value)
        ET.SubElement(cell, "mxGeometry", x=str(n.x), y=str(n.y), width=str(n.w),
                      height=str(n.h), **{"as": "geometry"})

    for e in EDGES:
        style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
                 f"html=1;fontSize=9;labelBackgroundColor=#FFFFFF;"
                 f"exitX={e.exit[0]};exitY={e.exit[1]};exitDx=0;exitDy=0;"
                 f"entryX={e.entry[0]};entryY={e.entry[1]};entryDx=0;entryDy=0;")
        if e.dashed:
            style += "dashed=1;strokeColor=#007FFF;"
        else:
            style += "strokeColor=#000000;"
        cell = ET.SubElement(root, "mxCell", id=e.id, parent="1", edge="1",
                             source=e.src, target=e.dst,
                             style=style, value=e.label)
        ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})

    ET.indent(root_model, space="  ")
    out = HERE / "Flipit-process.xml"
    ET.ElementTree(root_model).write(out, encoding="utf-8", xml_declaration=True)
    return out


# --------------------------------------------------------------------------
# PNG (matplotlib)
# --------------------------------------------------------------------------


def emit_png() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(CANVAS_W / 100, CANVAS_H / 100), dpi=150)
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(0, CANVAS_H)
    ax.axis("off")
    top = CANVAS_H  # drawio y-down → matplotlib y-up conversion

    def ty(y: float) -> float:
        return top - y

    ax.text(580, ty(37), TITLE, ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(580, ty(69), SUBTITLE, ha="center", va="center", fontsize=9.5, style="italic",
            color="#333333")
    ax.text(COL_L_X + COL_W / 2, ty(114), HEADER_L, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#2f5b32")
    ax.text(COL_R_X + COL_W / 2, ty(114), HEADER_R, ha="center", va="center",
            fontsize=11, fontweight="bold", color="#36476b")

    for label, ref in STAGES:
        n = NODES[ref]
        ax.text(GAP_X + 40, ty(n.y - 14), label, ha="center", va="center", fontsize=8.5,
                style="italic", color="#555555")

    for n in NODES.values():
        patch = FancyBboxPatch(
            (n.x + 1.5, ty(n.y + n.h) + 1.5), n.w - 3, n.h - 3,
            boxstyle="round,pad=0,rounding_size=10" if n.rounded else "round,pad=0,rounding_size=3",
            linewidth=1.4, edgecolor=n.stroke, facecolor=n.fill, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(n.x + n.w / 2, ty(n.y + 14), n.title, ha="center", va="center",
                fontsize=9.6, fontweight="bold", zorder=3)
        body = "\n".join(n.body_lines)
        ax.text(n.x + 14, ty(n.y + 30), body, ha="left", va="top",
                fontsize=8.2, zorder=3, linespacing=1.45)

    def port(n: Node, rel: tuple[float, float]) -> tuple[float, float]:
        return n.x + rel[0] * n.w, ty(n.y + rel[1] * n.h)

    for e in EDGES:
        src, dst = NODES[e.src], NODES[e.dst]
        (x0, y0), (x1, y1) = port(src, e.exit), port(dst, e.entry)
        elbows: list[tuple[float, float]] = []
        for ex, ey in e.elbows:
            elbows.append((ex, ty(ey) if ey else ty(src.y + src.h + 20)))
        if not elbows and abs(y0 - y1) > 1 and abs(x0 - x1) > 1:
            # generic orthogonal L-route into the entry side
            if e.exit[0] in (0, 1) and e.entry[1] in (0, 1):
                elbows = [(x1, y0)]
            else:
                elbows = [(x0, y1)]
        if elbows and (abs(elbows[-1][0] - x1) <= 1 and abs(elbows[-1][1] - y1) <= 1):
            elbows = elbows[:-1]
        pts = [(x0, y0), *elbows, (x1, y1)]
        color = "#007FFF" if e.dashed else "#000000"
        ls = (0, (5, 3)) if e.dashed else "solid"
        for (ax_, ay_), (bx_, by_) in zip(pts, pts[1:]):
            is_last = (bx_, by_) == pts[-1]
            ax.annotate("", xy=(bx_, by_), xytext=(ax_, ay_), zorder=1,
                        arrowprops=dict(arrowstyle="-|>" if is_last else "-",
                                        color=color, linestyle=ls, lw=1.5, shrinkA=0, shrinkB=0))
        if e.label:
            midseg = pts[len(pts) // 2 - 1], pts[len(pts) // 2]
            lx, ly = (midseg[0][0] + midseg[1][0]) / 2, (midseg[0][1] + midseg[1][1]) / 2
            ax.text(lx, ly + 9, e.label, ha="center", va="center", fontsize=7.6, zorder=4,
                    color="#222222",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

    out = HERE / "Flipit-process.png"
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


if __name__ == "__main__":
    check_layout()
    xml_path = emit_drawio()
    # re-parse — the file that ships must be well-formed
    ET.parse(xml_path)
    png_path = emit_png()
    print(f"wrote {xml_path}")
    print(f"wrote {png_path}")
    print(f"canvas {CANVAS_W}x{CANVAS_H}, nodes={len(NODES)}, edges={len(EDGES)}")
