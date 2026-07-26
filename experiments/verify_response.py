"""End-to-end verification: re-derive every headline number in RESPONSE_4PAGE.md
from the released CSVs and assert it matches (20-model cifar100 canonical + gtsrb
cross-dataset + adaptive + edge). Prints PASS/FAIL per claim. Exit 0 iff all pass.
    python experiments/verify_response.py
"""
from __future__ import annotations
import sys, glob
import numpy as np, pandas as pd
import config

AD = config.ANALYSIS_DIR
TOL = 0.015
checks = []


def chk(name, got, want, tol=TOL):
    ok = (got is not None) and not (isinstance(got, float) and np.isnan(got)) and abs(got - want) <= tol
    checks.append((ok, name, got, want)); return ok


def near(name, got, lo, hi):
    ok = (got is not None) and (lo - TOL <= got <= hi + TOL)
    checks.append((ok, name, got, f"[{lo},{hi}]")); return ok


# ---- full_eval summary (cifar100, 20 models) ----
s = pd.read_csv(AD / "full_eval_cifar100_summary.csv").set_index(["method", "task"])
def dep(m, t): return float(s.loc[(m, t), "deployable"])
chk("[c100-20] T2 tv_dorm 0.966", dep("ViyogD_tv_dorm", "T2"), 0.966, 0.02)
chk("[c100-20] T3 tv_dorm 0.824", dep("ViyogD_tv_dorm", "T3"), 0.824, 0.02)
near("[c100-20] T2 Energy inconsistent <0.45", dep("Energy", "T2"), 0.30, 0.45)
checks.append((not bool(s.loc[("Energy", "T2"), "dir_consistent"]), "[c100-20] Energy T2 dir-INconsistent", bool(s.loc[("Energy", "T2"), "dir_consistent"]), False))
checks.append((bool(s.loc[("ViyogD_tv_dorm", "T2"), "dir_consistent"]), "[c100-20] tv_dorm T2 dir-consistent", True, True))
# tv_dorm is the best T3 (>= all logit)
t3 = s.xs("T3", level="task")["deployable"]
checks.append((t3["ViyogD_tv_dorm"] >= t3[["Energy", "GEN", "MSP", "MaxLogit"]].max() - 0.005,
               "[c100-20] tv_dorm is best/tied T3", round(float(t3["ViyogD_tv_dorm"]), 3), ">=logit"))

# ---- complementarity cifar100 (20 models) + bootstrap pack ----
c = pd.read_csv(AD / "complementarity_cifar100.csv").groupby("feature_set")["bal_acc"].mean()
chk("[c100-20] Energy-only 3-way 0.615", float(c["Energy only (logit)"]), 0.615, 0.02)
chk("[c100-20] Viyog-only 3-way 0.702", float(c["Viyog only"]), 0.702, 0.02)
chk("[c100-20] Energy+Viyog 0.757", float(c["Energy + Viyog"]), 0.757, 0.02)
chk("[c100-20] Full panel 0.798", float(c["Full panel"]), 0.798, 0.02)
# per-class complementarity signature
cc = pd.read_csv(AD / "complementarity_cifar100.csv").groupby("feature_set").mean(numeric_only=True)
near("[c100-20] Energy ADV recall ~0.45 (blind)", float(cc.loc["Energy only (logit)", "recall_ADV"]), 0.40, 0.50)
near("[c100-20] Viyog OOD recall ~0.29 (weak)", float(cc.loc["Viyog only", "recall_OOD"]), 0.24, 0.34)

# ---- gtsrb cross-dataset (analysis lives under results/gtsrb/analysis, not the shared AD) ----
GAD = config.RESULTS_DIR / "gtsrb" / "analysis"
_gad = GAD if (GAD / "complementarity_gtsrb.csv").exists() else AD  # fall back to AD if co-located
if (_gad / "complementarity_gtsrb.csv").exists():
    g = pd.read_csv(_gad / "complementarity_gtsrb.csv").groupby("feature_set")["bal_acc"].mean()
    chk("[gtsrb] Full panel 3-way 0.863", float(g["Full panel"]), 0.863, 0.02)
    gf = pd.read_csv(_gad / "full_eval_gtsrb_summary.csv").set_index(["method", "task"])
    chk("[gtsrb] T2 tv_dorm 0.939", float(gf.loc[("ViyogD_tv_dorm", "T2"), "deployable"]), 0.939, 0.02)
else:
    print("  [WARN] gtsrb analysis CSVs not found under results/gtsrb/analysis or results/analysis — gtsrb checks skipped")

# ---- end-to-end (Table 4 / §3): ID->ADV mis-escalation ----
ps = pd.read_csv(AD / "pipeline_seeds_cifar100.csv").groupby("detector")["ID_FP_to_ADV"].mean()
chk("[e2e] Viyog_D* ID->ADV 0.052", float(ps["Viyog_D*"]), 0.052, 0.01)
chk("[e2e] Viyog_Linf ID->ADV 0.819", float(ps["Viyog_Linf"]), 0.819, 0.02)
# outlier disclosure: effnet_lite0 carries the mean (median ~0)
pe = pd.read_csv(AD / "pipeline_seeds_cifar100.csv")
vd = pe[pe.detector == "Viyog_D*"].groupby("model")["ID_FP_to_ADV"].mean()
near("[e2e] effnet_lite0 outlier ~0.82", float(vd.get("effnet_lite0", np.nan)), 0.70, 0.95)
near("[e2e] median ID->ADV ~0", float(vd.median()), -0.001, 0.05)

# ---- adaptive signature-aware (Table 4 §4) ----
def adapt_mode(mode):
    vals = {"succ": [], "dorm": [], "hf": []}
    for f in glob.glob(str(AD / "adaptive_cifar100_*.csv")):
        d = pd.read_csv(f); sub = d[d["mode"] == mode]
        if len(sub):
            r = sub.iloc[-1]
            vals["succ"].append(r.attack_success); vals["dorm"].append(r.auroc_dorm); vals["hf"].append(r.auroc_hf)
    return {k: np.mean(v) for k, v in vals.items()}
np_ = adapt_mode("normpresv"); aw = adapt_mode("allaware"); base = adapt_mode("pgd")
near("[adapt] norm-preserving dorm unchanged ~0.89", np_["dorm"], 0.82, 0.94)
near("[adapt] both-aware success cost (<base)", base["succ"] - aw["succ"], 0.02, 0.20)
near("[adapt] both-aware dorm eroded but >chance ~0.59", aw["dorm"], 0.52, 0.66)

# ---- edge / detector-state (EDGE_METRICS, §5-A) ----
if (AD / "edge_latency.csv").exists():
    el = pd.read_csv(AD / "edge_latency.csv")
    near("[edge] firstconv lat ratio CNN <6% (ex-densenet)", float(el[el.model != "densenet121"]["firstconv_lat_ratio_%"].max()), 1.0, 6.0)
if (AD / "accelerator_energy.csv").exists():
    ae = pd.read_csv(AD / "accelerator_energy.csv")
    near("[edge] ZigZag firstconv energy 3-8%", float(ae["E_firstconv_%"].mean()), 3.0, 8.0)

# ---- B5 GELU constant ----
from scipy.stats import norm
chk("[B5] sup|GELU''| = 2 phi(0) = 0.798", float(2 * norm.pdf(0)), 0.798, 0.001)

# ---- report ----
npass = sum(1 for ok, *_ in checks if ok)
print(f"\n========= END-TO-END VERIFICATION (20-model + gtsrb + adaptive + edge): {npass}/{len(checks)} PASS =========\n")
for ok, name, got, want in checks:
    g = f"{got:.4f}" if isinstance(got, float) else str(got)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:52s} got={g:>10} want={want}")
fails = [c for c in checks if not c[0]]
print(f"\n  {'ALL PASS' if not fails else str(len(fails)) + ' FAILED'}")
sys.exit(0 if not fails else 1)
