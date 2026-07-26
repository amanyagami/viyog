"""Viyog vs pytorch-ood --- interactive OOD-vs-ADV leaderboard (Gradio).

A small, fully static-data app: it serves the tidy CSVs produced by
``export_leaderboard.py`` (a few tens of KB) and renders an interactive
leaderboard + plots comparing Viyog against pytorch-ood baseline detectors on
the OOD-vs-ADV separation task across architectures and datasets.

No GPU, no model weights, no large data --- runs on a free HuggingFace Spaces
CPU box (or locally: ``uv run --frozen python demo/app.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DATA = Path(__file__).resolve().parent / "data"

# ---- Okabe-Ito colourblind-safe palette (matches the paper) -----------------
C_VIYOG = "#009E73"   # green   - Viyog (ours)
C_LOGIT = "#0072B2"   # blue    - logit detectors
C_DIST = "#D55E00"    # orange  - distance/feature detectors
C_GREY = "#999999"    # grey    - raw-norm baseline
FAMILY_COLOR = {
    "Viyog (first-conv)": C_VIYOG,
    "logit": C_LOGIT,
    "distance (feature)": C_DIST,
}

LB = pd.read_csv(DATA / "leaderboard.csv")
PM = pd.read_csv(DATA / "permodel_t3.csv")
OD = pd.read_csv(DATA / "ood_difficulty.csv")
META = json.loads((DATA / "meta.json").read_text())
DATASETS = META["datasets"]

_PLOT_KW = dict(template="plotly_white", font=dict(size=13))


def _fam_color(family: str) -> str:
    return FAMILY_COLOR.get(family, C_GREY)


# ============================================================ Leaderboard tab
def leaderboard_table(dataset: str):
    d = LB[LB.dataset == dataset].copy()
    d = d.sort_values("T3_OOD_ADV", ascending=False)
    d["rank"] = range(1, len(d) + 1)
    show = d[["rank", "detector", "family", "T3_OOD_ADV", "T2_ID_ADV",
              "T1_ID_OOD", "state_mem_KB"]].rename(columns={
        "detector": "Detector", "family": "Family",
        "T3_OOD_ADV": "T3 OOD-vs-ADV ▲", "T2_ID_ADV": "T2 ID-vs-ADV",
        "T1_ID_OOD": "T1 ID-vs-OOD", "state_mem_KB": "State (KB)",
        "rank": "#",
    })
    for c in ["T3 OOD-vs-ADV ▲", "T2 ID-vs-ADV", "T1 ID-vs-OOD"]:
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    show["State (KB)"] = show["State (KB)"].map(lambda v: f"{v:,.2f}")
    return show


def leaderboard_bar(dataset: str):
    d = LB[LB.dataset == dataset].sort_values("T3_OOD_ADV")
    fig = go.Figure(go.Bar(
        x=d["T3_OOD_ADV"], y=d["detector"], orientation="h",
        marker_color=[_fam_color(f) for f in d["family"]],
        text=[f"{v:.3f}" for v in d["T3_OOD_ADV"]], textposition="outside",
    ))
    fig.add_vline(x=0.5, line=dict(color=C_GREY, dash="dot"),
                  annotation_text="chance", annotation_position="top")
    fig.update_layout(
        title=f"OOD-vs-ADV separation (T3 AUROC) — {dataset}",
        xaxis_title="T3 AUROC (1.0 perfect · 0.5 chance)", yaxis_title="",
        xaxis_range=[0.45, 1.0], height=460, margin=dict(l=10, r=30, t=50, b=40),
        **_PLOT_KW)
    return fig


# ======================================================= Cost-vs-accuracy tab
def cost_scatter(dataset: str):
    d = LB[LB.dataset == dataset].copy()
    d["mem"] = d["state_mem_KB"].clip(lower=0.003)
    fig = go.Figure()
    for fam, sub in d.groupby("family"):
        fig.add_trace(go.Scatter(
            x=sub["mem"], y=sub["T3_OOD_ADV"], mode="markers+text",
            name=fam, text=sub["detector"], textposition="top center",
            textfont=dict(size=10),
            marker=dict(size=[20 if v else 12 for v in sub["is_viyog"]],
                        color=_fam_color(fam), line=dict(width=1, color="white"),
                        symbol=["star" if v else "circle" for v in sub["is_viyog"]]),
        ))
    fig.add_hline(y=0.5, line=dict(color=C_GREY, dash="dot"))
    fig.update_layout(
        title=f"Detector cost vs accuracy — {dataset}  (top-left = cheap & accurate)",
        xaxis_title="Persistent state per model (KB, log scale)",
        yaxis_title="T3 OOD-vs-ADV AUROC", xaxis_type="log",
        height=480, margin=dict(l=10, r=10, t=50, b=40), **_PLOT_KW)
    return fig


# ========================================================= Per-architecture tab
def permodel_bar(dataset: str, baseline: str):
    d = PM[PM.dataset == dataset]
    viyog = d[d.detector == "Viyog-D (TV)"][["model", "T3"]].rename(columns={"T3": "Viyog-D (TV)"})
    base = d[d.detector == baseline][["model", "T3"]].rename(columns={"T3": baseline})
    m = viyog.merge(base, on="model").sort_values("Viyog-D (TV)", ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=m["model"], y=m["Viyog-D (TV)"], name="Viyog-D (TV)",
                         marker_color=C_VIYOG))
    fig.add_trace(go.Bar(x=m["model"], y=m[baseline], name=baseline,
                         marker_color=C_LOGIT))
    fig.add_hline(y=0.5, line=dict(color=C_GREY, dash="dot"),
                  annotation_text="chance")
    fig.update_layout(
        title=f"Per-architecture T3 (OOD-vs-ADV) — Viyog-D vs {baseline} — {dataset}",
        barmode="group", xaxis_title="", yaxis_title="T3 AUROC",
        yaxis_range=[0.4, 1.0], height=480, xaxis_tickangle=-40,
        margin=dict(l=10, r=10, t=50, b=110), **_PLOT_KW)
    return fig


def baseline_choices(dataset: str):
    opts = sorted(PM[(PM.dataset == dataset) & (~PM.is_viyog)]["detector"].unique())
    default = "Energy" if "Energy" in opts else (opts[0] if opts else None)
    return gr.update(choices=opts, value=default)


# ============================================================ OOD-difficulty tab
def ood_difficulty_bar(dataset: str):
    d = OD[OD.dataset == dataset]
    # mean over models per (detector, kind)
    agg = d.groupby(["detector", "kind"], as_index=False)["T3"].mean()
    order = ["Far", "Near", "Texture"]
    agg["kind"] = pd.Categorical(agg["kind"], categories=order, ordered=True)
    fig = px.bar(agg.sort_values("kind"), x="kind", y="T3", color="detector",
                 barmode="group", text_auto=".2f",
                 color_discrete_map={"Viyog-D (TV)": C_VIYOG, "Viyog-HF": C_LOGIT,
                                     "TV (mean)": "#56B4E9", "raw L∞": C_GREY})
    fig.add_hline(y=0.5, line=dict(color=C_GREY, dash="dot"),
                  annotation_text="chance")
    fig.update_layout(
        title=f"OOD-vs-ADV by OOD difficulty (mean over models) — {dataset}",
        xaxis_title="OOD difficulty", yaxis_title="T3 AUROC",
        yaxis_range=[0.4, 1.05], height=460, margin=dict(l=10, r=10, t=50, b=40),
        **_PLOT_KW)
    return fig


# ===================================================================== layout
ABOUT = f"""
## Viyog — separating adversarial inputs from out-of-distribution inputs

A deployed model needs to tell **two very different anomalies apart**: a benign
**out-of-distribution (OOD)** input (a novel-but-safe environment → *abstain*)
versus an **adversarial (ADV)** attack (→ *reject / alarm*). Most detectors
collapse both into one anomaly score. **Viyog** reads a tiny *roughness* statistic
off the **first convolutional layer's dormant band** — training-free, gradient-free,
no extra forward pass, **{LB[LB.is_viyog].state_mem_KB.min():.2f}–0.3 KB of state**.

### The three tasks
| Metric | Question | Who it favours |
|---|---|---|
| **T1** ID-vs-OOD | is this input in-distribution? | classic OOD detectors |
| **T2** ID-vs-ADV | is this input adversarial? | **Viyog (logit detectors are blind here)** |
| **T3** OOD-vs-ADV | given a non-ID input, OOD or ADV? | **Viyog — the headline task** |

AUROC: **1.0 = perfect, 0.5 = chance**. Numbers are directionless re-evaluations
across **{META['n_models']} architectures** and **{len(DATASETS)} datasets**
({', '.join(DATASETS)}). Baselines are the standard
[`pytorch-ood`](https://pytorch-ood.readthedocs.io) detectors (Energy, MSP,
MaxLogit, Entropy, KL-Matching, GEN, Mahalanobis, KNN, ViM).

*Data: `results/analysis/*.csv` — Viyog, CODES+ISSS 2026 (paper #215).*
"""

with gr.Blocks(title="Viyog OOD-vs-ADV Leaderboard",
               theme=gr.themes.Soft(primary_hue="green")) as demo:
    gr.Markdown("# 🛡️ Viyog vs `pytorch-ood` — OOD-vs-ADV Leaderboard")
    gr.Markdown("Separating **adversarial** from **out-of-distribution** inputs at "
                "the first conv layer — *training-free, gradient-free, sub-KB.*")

    with gr.Tab("🏆 Leaderboard"):
        ds1 = gr.Radio(DATASETS, value=DATASETS[0], label="Dataset (in-distribution)")
        gr.Markdown("Ranked by **T3 (OOD-vs-ADV)**. Viyog is green, logit detectors "
                    "blue, distance detectors orange.")
        tbl = gr.Dataframe(value=leaderboard_table(DATASETS[0]), interactive=False,
                           wrap=True)
        bar = gr.Plot(value=leaderboard_bar(DATASETS[0]))
        ds1.change(leaderboard_table, ds1, tbl)
        ds1.change(leaderboard_bar, ds1, bar)

    with gr.Tab("⚖️ Cost vs accuracy"):
        ds2 = gr.Radio(DATASETS, value=DATASETS[0], label="Dataset")
        gr.Markdown("Distance detectors (orange) buy accuracy with **7–25 MB** of "
                    "state; Viyog (green ★) sits **top-left**: highest T3 at "
                    "**sub-KB** cost.")
        sca = gr.Plot(value=cost_scatter(DATASETS[0]))
        ds2.change(cost_scatter, ds2, sca)

    with gr.Tab("🧩 Per-architecture"):
        with gr.Row():
            ds3 = gr.Radio(DATASETS, value=DATASETS[0], label="Dataset")
            base = gr.Dropdown(
                sorted(PM[(PM.dataset == DATASETS[0]) & (~PM.is_viyog)]["detector"].unique()),
                value="Energy", label="Baseline to compare against")
        pmb = gr.Plot(value=permodel_bar(DATASETS[0], "Energy"))
        ds3.change(baseline_choices, ds3, base).then(permodel_bar, [ds3, base], pmb)
        base.change(permodel_bar, [ds3, base], pmb)

    with gr.Tab("🎯 OOD difficulty"):
        ds4 = gr.Radio(DATASETS, value=DATASETS[0], label="Dataset")
        gr.Markdown("The total-variation read (**Viyog-D**) holds far/near OOD; its "
                    "high-frequency complement (**Viyog-HF**) rescues **texture-OOD**.")
        odb = gr.Plot(value=ood_difficulty_bar(DATASETS[0]))
        ds4.change(ood_difficulty_bar, ds4, odb)

    with gr.Tab("ℹ️ About"):
        gr.Markdown(ABOUT)


if __name__ == "__main__":
    demo.launch()
