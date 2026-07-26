"""Generate all rebuttal figures from the analysis CSVs into figures/rebuttal/.

Each figure is gated on its source CSV existing, so this can be run incrementally
as the campaign produces outputs. Figures:
  1. new_vs_old.png        — Viyog vs L∞ AUROC per model (viyog_score_compare_final.csv)
  2. far_vs_near.png       — L∞ vs Viyog on far- vs near-OOD (recomputed from featfull)
  3. adaptive_frontier.png — attack-success & detector AUROC vs λ (adaptive_*_*.csv)
  4. systems.png           — first-conv MAC ratio & latency per model (systems_*.csv)
  5. cascade.png           — end-to-end 3-way recall, dorm vs linf (cascade_*.csv)

    python experiments/make_figs.py --dataset cifar100
"""
from __future__ import annotations
import argparse, glob, os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

EPS = 1e-8
NEAR = {"cifar10", "gtsrb", "dtd"}
FAR = {"mnist", "svhn", "fashionmnist", "flowers102", "food101", "stl10", "eurosat"}


def _auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]; s = np.r_[neg, pos]
    if len(np.unique(s)) < 2:
        return 0.5
    a = roc_auc_score(y, s); return max(a, 1 - a)


def fig_new_vs_old(outdir):
    import pandas as pd
    p = config.ANALYSIS_DIR / "viyog_score_compare_final.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    piv = df.pivot_table(index="model", columns="score", values="T3_OOD_ADV")
    piv = piv.sort_values("viyog_dorm")
    y = np.arange(len(piv))
    plt.figure(figsize=(7, 6))
    plt.barh(y - 0.2, piv["viyog_linf"], height=0.4, label="L∞ (paper)", color="#bbb")
    plt.barh(y + 0.2, piv["viyog_dorm"], height=0.4, label="Viyog (dorm)", color="#1f77b4")
    plt.yticks(y, piv.index, fontsize=8); plt.axvline(0.5, ls=":", c="k", lw=0.8)
    plt.xlabel("OOD-vs-ADV AUROC (pooled 10-OOD)"); plt.legend(); plt.title("New vs old Viyog score (T3)")
    plt.tight_layout(); plt.savefig(outdir / "new_vs_old.png", dpi=140); plt.close()
    print("  ✓ new_vs_old.png")


def fig_far_vs_near(outdir, models):
    import h5py
    FD = config.FEATURES_DIR
    def load(p, key):
        with h5py.File(p, "r") as f: return f[key][:].astype(np.float64)
    rows = []
    for m in models:
        idp = FD / f"featfull_{m}_id.h5"
        if not idp.exists():
            continue
        idm = load(idp, "filter_means"); dorm = np.argsort(idm.mean(0))[:max(1, int(0.10 * idm.shape[1]))]
        def st(p):
            fm = load(p, "filter_means")
            return {"linf": load(p, "inf_norms"), "dorm": fm[:, dorm].sum(1) / (fm.sum(1) + EPS)}
        advs = [st(p) for p in glob.glob(str(FD / f"featfull_{m}_adv_*.h5"))]
        oods = {os.path.basename(p).split("_ood_")[1][:-3]: st(p) for p in glob.glob(str(FD / f"featfull_{m}_ood_*.h5"))}
        if not advs or not oods:
            continue
        A = {k: np.concatenate([a[k] for a in advs]) for k in advs[0]}
        for k in ["linf", "dorm"]:
            far = np.mean([_auroc(o[k], A[k]) for n, o in oods.items() if n in FAR])
            near = np.mean([_auroc(o[k], A[k]) for n, o in oods.items() if n in NEAR])
            rows.append((m, k, far, near))
    if not rows:
        return
    import pandas as pd
    df = pd.DataFrame(rows, columns=["model", "stat", "far", "near"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for j, (k, c) in enumerate([("linf", "#888"), ("dorm", "#1f77b4")]):
        s = df[df.stat == k]
        ax[0].scatter(s["model"], s["far"], label=k, color=c)
        ax[1].scatter(s["model"], s["near"], label=k, color=c)
    for a, t in zip(ax, ["FAR-OOD", "NEAR-OOD"]):
        a.set_title(t); a.axhline(0.5, ls=":", c="k", lw=0.8); a.tick_params(axis="x", rotation=90, labelsize=6)
        a.set_ylim(0.45, 1.0)
    ax[0].set_ylabel("OOD-vs-ADV AUROC"); ax[0].legend()
    plt.suptitle("L∞ collapses on near-OOD; Viyog recovers it"); plt.tight_layout()
    plt.savefig(outdir / "far_vs_near.png", dpi=140); plt.close()
    print("  ✓ far_vs_near.png")


def fig_adaptive(outdir):
    import pandas as pd
    for p in glob.glob(str(config.ANALYSIS_DIR / "adaptive_*_*.csv")):
        df = pd.read_csv(p)
        model = os.path.basename(p).replace("adaptive_", "").replace(".csv", "")
        sub = df[df["mode"] == "normpresv"]
        if not len(sub):
            continue
        plt.figure(figsize=(6, 4))
        plt.plot(sub["lambda"], sub["auroc_linf"], "o-", label="L∞ detector")
        plt.plot(sub["lambda"], sub["auroc_dorm"], "s-", label="dorm detector")
        plt.plot(sub["lambda"], sub["auroc_hf"], "^-", label="HF detector")
        plt.plot(sub["lambda"], sub["attack_success"], "x--", c="r", label="attack success")
        plt.xscale("symlog"); plt.xlabel("norm-preservation penalty λ"); plt.ylabel("rate / AUROC")
        plt.title(f"Adaptive attack cost frontier — {model}"); plt.legend(fontsize=8); plt.ylim(0, 1.05)
        plt.tight_layout(); plt.savefig(outdir / f"adaptive_{model}.png", dpi=140); plt.close()
        print(f"  ✓ adaptive_{model}.png")


def fig_systems(outdir):
    import pandas as pd
    p = config.ANALYSIS_DIR / f"systems_{config.CURRENT_DATASET}.csv" \
        if hasattr(config, "CURRENT_DATASET") else None
    cands = glob.glob(str(config.ANALYSIS_DIR / "systems_*.csv"))
    if not cands:
        return
    df = pd.read_csv(cands[0]).sort_values("params_M")
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(df["model"], df["firstconv_mac_ratio_%"], color="#1f77b4", alpha=0.7)
    ax1.set_ylabel("first-conv MAC ratio (%)", color="#1f77b4"); ax1.tick_params(axis="x", rotation=90, labelsize=7)
    ax2 = ax1.twinx()
    ax2.plot(df["model"], df["lat_full_ms_per_img"], "ro-", label="full latency (ms/img)")
    ax2.set_ylabel("latency ms/img", color="r")
    plt.title("Systems: Viyog first-conv cost vs full forward"); plt.tight_layout()
    plt.savefig(outdir / "systems.png", dpi=140); plt.close()
    print("  ✓ systems.png")


def fig_cascade(outdir):
    import pandas as pd
    # the energy/msp cascade file carries stage1; exclude cascade_recall_full_*
    cands = [p for p in glob.glob(str(config.ANALYSIS_DIR / "cascade_*.csv"))
             if "recall_full" not in p]
    if not cands:
        return
    df = pd.read_csv(cands[0])
    if "stage1" not in df.columns:
        return
    g = df[df.stage1 == "energy"].groupby("stage2")[
        ["e2e_recall_OOD", "e2e_recall_ADV", "ID_FP_escalated_to_ADV"]].mean()
    g.plot(kind="bar", figsize=(7, 4))
    plt.title("End-to-end cascade (Energy stage-1): dorm vs L∞")
    plt.ylabel("rate"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(outdir / "cascade.png", dpi=140); plt.close()
    print("  ✓ cascade.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar100", choices=list(config.DATASET_SPECS))
    args = ap.parse_args()
    config.set_dataset(args.dataset)
    outdir = config.PLOTS_DIR / "rebuttal"
    outdir.mkdir(parents=True, exist_ok=True)
    models = [m for m in config.MODEL_ARCHS if (config.FEATURES_DIR / f"featfull_{m}_id.h5").exists()]
    print(f"=== make_figs [{args.dataset}] → {outdir} ===")
    fig_new_vs_old(outdir)
    fig_far_vs_near(outdir, models)
    fig_adaptive(outdir)
    fig_systems(outdir)
    fig_cascade(outdir)
    print(f"  figures in {outdir}")


if __name__ == "__main__":
    main()
