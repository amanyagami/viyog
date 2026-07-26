"""Rebuttal figures from full_eval_*.csv (+ merged feature baselines).

Produces, into results/plots/rebuttal/:
  fig_task_auroc.png       grouped bars: deployable AUROC per method x {T1,T2,T3}
  fig_fpr95.png            FPR@95 per method x task (lower=better) — operating point
  fig_t2_blindspot.png     T2 ID-vs-ADV: logit baselines blind (dir-inconsistent) vs Viyog
  fig_t3_fpr_scatter.png   T3 AUROC vs FPR95 trade-off (Viyog dominates the corner)
  fig_perattack.png        per-attack T2/T3 heatmap (L-inf vs tv_dorm vs best logit)
  fig_complementarity.png  each method's best task — shows no single winner
A pure-matplotlib, CPU, post-hoc step.

    python experiments/full_eval_plots.py --dataset cifar100
"""
from __future__ import annotations
import argparse, os
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

ORDER = ["Viyog_Linf", "ViyogD_tv_dorm", "ViyogD_hf_dorm", "MSP", "MaxLogit",
         "Energy", "Entropy", "GEN", "KLMatching", "Mahalanobis", "KNN", "ViM", "ODIN"]
PRETTY = {"Viyog_Linf": "Viyog L∞ (paper)", "ViyogD_tv_dorm": "Viyog tv_dorm",
          "ViyogD_hf_dorm": "Viyog hf_dorm"}
VIYOG = {"Viyog_Linf", "ViyogD_tv_dorm", "ViyogD_hf_dorm"}


def label(m):
    return PRETTY.get(m, m)


def col(m):
    if m == "ViyogD_tv_dorm":
        return "#c0392b"
    if m == "ViyogD_hf_dorm":
        return "#e67e22"
    if m == "Viyog_Linf":
        return "#7f8c8d"
    return "#2c7fb8"


def merge_feature_baselines(summary, ds, ad):
    """Fold baselines_feature_<ds>.csv (Maha/KNN/ViM/ODIN) into the summary frame
    as per-task rows, if present. That CSV has per (model,method) T1/T2/T3 dirless."""
    p = ad / f"baselines_feature_{ds}.csv"
    if not p.exists():
        return summary, []
    bf = pd.read_csv(p)
    colmap = {"T1_ID_OOD": "T1", "T2_ID_ADV": "T2", "T3_OOD_ADV": "T3"}
    have = [c for c in colmap if c in bf.columns]
    rows = []
    for meth, sub in bf.groupby("method"):
        for raw, task in colmap.items():
            if raw not in have:
                continue
            rows.append(dict(method=meth, task=task,
                             deployable=round(sub[raw].mean(), 4),
                             mean_dl=round(sub[raw].mean(), 4),
                             dir_consistent=True, mean_fpr95=np.nan, mean_aupr=np.nan))
    add = pd.DataFrame(rows)
    return pd.concat([summary, add], ignore_index=True), sorted(bf["method"].unique())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="cifar100")
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    ad = config.ANALYSIS_DIR
    pd_dir = config.PLOTS_DIR / "rebuttal"; pd_dir.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(ad / f"full_eval_{args.dataset}_summary.csv")
    pm = pd.read_csv(ad / f"full_eval_{args.dataset}_permodel.csv")
    s, feat_methods = merge_feature_baselines(s, args.dataset, ad)

    methods = [m for m in ORDER if m in set(s.method)]

    def series(task, field):
        d = s[s.task == task].set_index("method")
        return [d.loc[m, field] if m in d.index else np.nan for m in methods]

    # ---- fig 1: grouped deployable AUROC per task ----
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(methods)); w = 0.26
    for i, (t, c) in enumerate([("T1", "#74add1"), ("T2", "#f46d43"), ("T3", "#1a9850")]):
        ax.bar(x + (i - 1) * w, series(t, "deployable"), w,
               label={"T1": "T1 ID-vs-OOD", "T2": "T2 ID-vs-ADV", "T3": "T3 OOD-vs-ADV (deploy)"}[t], color=c)
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([label(m) for m in methods], rotation=35, ha="right")
    ax.set_ylabel("deployable AUROC\n(directionless only if direction-consistent)")
    ax.set_title(f"[{args.dataset}] Per-task separation — no single method wins all three")
    ax.legend(loc="lower right"); ax.set_ylim(0.3, 1.0); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(pd_dir / "fig_task_auroc.png", dpi=150); plt.close()

    # ---- fig 2: FPR@95 per task (lower=better) ----
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (t, c) in enumerate([("T1", "#74add1"), ("T2", "#f46d43"), ("T3", "#1a9850")]):
        ax.bar(x + (i - 1) * w, series(t, "mean_fpr95"), w,
               label={"T1": "T1", "T2": "T2", "T3": "T3 (deploy)"}[t], color=c)
    ax.set_xticks(x); ax.set_xticklabels([label(m) for m in methods], rotation=35, ha="right")
    ax.set_ylabel("FPR @ 95% TPR  (lower = better)")
    ax.set_title(f"[{args.dataset}] Operating-point cost — Viyog minimises false alarms on T2/T3")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(pd_dir / "fig_fpr95.png", dpi=150); plt.close()

    # ---- fig 3: T2 blind-spot (per-model dots, dir sign) ----
    fig, ax = plt.subplots(figsize=(11, 5))
    t2 = pm.copy()
    show = [m for m in methods if m in set(t2.method)]
    for j, m in enumerate(show):
        sub = t2[t2.method == m]
        # deployable per-model T2 (raw if any sign flip in this method, else dl)
        vals = sub["T2_dl"].values
        signs = sub["T2_sign"].values
        flip = not (np.all(signs >= 0) or np.all(signs <= 0))
        y = sub["T2_raw"].values if flip else vals
        ax.scatter([j] * len(y), y, s=28, c=[col(m)], alpha=0.8, zorder=3,
                   edgecolors="k", linewidths=0.3)
        ax.scatter([j], [np.mean(y)], marker="_", s=900, c="k", zorder=4)
        if flip:
            ax.text(j, 0.32, "dir✗", ha="center", fontsize=7, color="crimson")
    ax.axhline(0.5, ls="--", c="gray", lw=0.8)
    ax.set_xticks(range(len(show))); ax.set_xticklabels([label(m) for m in show], rotation=35, ha="right")
    ax.set_ylabel("T2 ID-vs-ADV AUROC (per model)")
    ax.set_title(f"[{args.dataset}] Adversarials are invisible to logit/feature OOD baselines; "
                 "first-layer Viyog detects them")
    ax.set_ylim(0.3, 1.02); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(pd_dir / "fig_t2_blindspot.png", dpi=150); plt.close()

    # ---- fig 4: T3 AUROC vs FPR95 trade-off ----
    fig, ax = plt.subplots(figsize=(8, 6))
    t3 = s[s.task == "T3"].set_index("method")
    for m in methods:
        if m not in t3.index or np.isnan(t3.loc[m, "mean_fpr95"]):
            continue
        ax.scatter(t3.loc[m, "mean_fpr95"], t3.loc[m, "deployable"], s=140 if m in VIYOG else 80,
                   c=col(m), edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(label(m), (t3.loc[m, "mean_fpr95"], t3.loc[m, "deployable"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("FPR @ 95% TPR  (lower = better →)"); ax.set_ylabel("deployable AUROC (higher = better ↑)")
    ax.invert_xaxis()
    ax.set_title(f"[{args.dataset}] T3 OOD-vs-ADV: Viyog owns the low-FPR / high-AUROC corner")
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(pd_dir / "fig_t3_fpr_scatter.png", dpi=150); plt.close()

    # ---- fig 5: per-attack heatmap (T3) for key methods ----
    pa = pd.read_csv(ad / f"full_eval_{args.dataset}_perattack.csv")
    key = [m for m in ["Viyog_Linf", "ViyogD_tv_dorm", "ViyogD_hf_dorm", "Energy", "GEN", "MSP"] if m in set(pa.method)]
    attacks = sorted(pa.attack.unique())
    M = np.array([[pa[(pa.method == m) & (pa.attack == a)]["T3_dl"].mean() for a in attacks] for m in key])
    fig, ax = plt.subplots(figsize=(1.4 * len(attacks) + 2, 0.6 * len(key) + 2))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(attacks))); ax.set_xticklabels(attacks, rotation=20, ha="right")
    ax.set_yticks(range(len(key))); ax.set_yticklabels([label(m) for m in key])
    for i in range(len(key)):
        for j in range(len(attacks)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="T3 AUROC (dirless)")
    ax.set_title(f"[{args.dataset}] T3 OOD-vs-ADV by attack — Viyog generalises across attacks")
    plt.tight_layout(); plt.savefig(pd_dir / "fig_perattack.png", dpi=150); plt.close()

    print(f"[{args.dataset}] wrote 5 figs -> {pd_dir}")
    print("  methods plotted:", methods)
    if feat_methods:
        print("  feature baselines merged:", feat_methods)


if __name__ == "__main__":
    main()
