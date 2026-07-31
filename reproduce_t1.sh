#!/usr/bin/env bash
#
#  Viyog — CODES+ISSS 2026 artifact — one-shot reproduction
#  =========================================================
#
#      bash reproduce_t1.sh --quick    # smoke test: does it work on my machine?
#      bash reproduce_t1.sh            # full Tier T1: reproduce the paper
#
#  Timing is dominated by FIRST-RUN DATASET DOWNLOADS, not compute. --quick
#  needs ~2 min of GPU work but must fetch ~350 MB of benchmark data the
#  first time; on a throttled link that can be 30-60 min. Once cached, a
#  re-run is minutes. The full tier measured 4 h 7 m end to end on one H200.
#
#  --quick runs the SAME pipeline end to end (real checkpoints, real data,
#  real attack) on one backbone / one attack / two OOD sets. It proves the
#  artifact functions on your hardware before you commit hours to it. Its
#  AUROCs are deliberately NOT comparable to the paper.
#
#  Hardware : one CUDA GPU of any size — batch sizes are scaled to your card.
#  Disk     : ~60 GB free recommended (datasets dominate).
#  Network  : HuggingFace (checkpoints) + torchvision mirrors (datasets).
#  Re-runs  : safe and fast — finished work is skipped, downloads are cached.
#
set -uo pipefail

# ------------------------------------------------------------------ ui
if [[ -t 1 ]]; then
    RED=$'\033[31m'; GRN=$'\033[32m'; YEL=$'\033[33m'; BLD=$'\033[1m'; RST=$'\033[0m'
else   # piped to a file or CI: keep the log free of escape codes
    RED=''; GRN=''; YEL=''; BLD=''; RST=''
fi
step() { printf '\n%s=== %s ===%s\n' "$BLD" "$*" "$RST"; }
ok()   { printf '%s  ok  %s %s\n' "$GRN" "$RST" "$*"; }
warn() { printf '%s  !!  %s %s\n' "$YEL" "$RST" "$*"; }
die()  { printf '\n%s  FAILED  %s %s\n' "$RED" "$RST" "$*" >&2; exit 1; }

usage() { sed -n '3,18p' "$0" | sed 's/^#\{1,2\} \{0,1\}//'; exit "${1:-0}"; }

# ------------------------------------------------------------------ args
QUICK=0
case "${1:-}" in
    --quick)        QUICK=1 ;;
    -h|--help|help) usage 0 ;;
    "")             ;;
    *)              printf '%sunknown argument: %s%s\n\n' "$RED" "$1" "$RST" >&2
                    usage 1 ;;        # never silently start a 4-hour run
esac

MODELS="convnextv2_base densenet121 mobilenetv3_l resnet50 swin_tiny vit_base"
ATTACKS="fgsm bim pgd apgd_ce"
OOD_ARG=""
N_OOD=10
if (( QUICK )); then
    MODELS="mobilenetv3_l"           # fastest core backbone
    ATTACKS="fgsm"                   # single-step attack
    OOD_ARG="--ood cifar10 svhn"     # two small, fast-downloading OOD sets
    N_OOD=2
fi
N_MODELS=$(wc -w <<<"$MODELS")
N_ATTACKS=$(wc -w <<<"$ATTACKS")

cd "$(dirname "$0")" || die "cannot cd to the artifact root"

# ------------------------------------------------------------------ preflight
if (( QUICK )); then
    step "0/6  Preflight   [QUICK smoke tier]"
    warn "QUICK: $N_MODELS model, $N_ATTACKS attack, $N_OOD OOD sets."
    warn "Validates that the pipeline runs. NOT the paper's numbers."
else
    step "0/6  Preflight   [full Tier T1]"
fi

command -v uv >/dev/null 2>&1 \
    || die "uv not found. Install it with:
       curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version 2>/dev/null | awk '{print $2}')"

command -v nvidia-smi >/dev/null 2>&1 \
    || die "no nvidia-smi — Tier T1 needs one CUDA GPU.

   Without a GPU you can still run the CPU-only sanity tier:
       uv run pytest tests/ -q
       uv run python examples/quickstart.py --smoke"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ok "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Size from FREE memory, not total: on a shared cluster GPU (a common reviewer
# situation) most of the card can already be in use by another process, and
# sizing from total capacity would OOM immediately.
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
FREE_MB=$(nvidia-smi --query-gpu=memory.free  --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
[[ -n "$TOTAL_MB" && -n "$FREE_MB" ]] || die "could not read GPU memory from nvidia-smi"
TOTAL_GB=$(( TOTAL_MB / 1024 )); VRAM_GB=$(( FREE_MB / 1024 ))
ok "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) (${TOTAL_GB} GB total, ${VRAM_GB} GB free)"
if (( TOTAL_GB - VRAM_GB > 8 )); then
    warn "$(( TOTAL_GB - VRAM_GB )) GB of this GPU is already in use by another process."
    warn "Batches are sized to the ${VRAM_GB} GB actually free. For the full run, prefer an idle GPU:"
    warn "    nvidia-smi --query-gpu=index,memory.free --format=csv"
    warn "    CUDA_VISIBLE_DEVICES=<idle-index> bash $(basename "$0")"
fi

# Committed config defaults are sized for the paper's 143 GB H200. Scale to
# this card: batch size affects throughput only, never results.
if   (( VRAM_GB >= 100 )); then ATK_B="";  FEAT_B="";    PROFILE="native (config defaults)"
elif (( VRAM_GB >= 60  )); then ATK_B=96;  FEAT_B=1024;  PROFILE="large"
elif (( VRAM_GB >= 36  )); then ATK_B=48;  FEAT_B=512;   PROFILE="medium"
elif (( VRAM_GB >= 20  )); then ATK_B=24;  FEAT_B=256;   PROFILE="small"
elif (( VRAM_GB >= 10  )); then ATK_B=12;  FEAT_B=128;   PROFILE="modest"
else                            ATK_B=6;   FEAT_B=64;    PROFILE="minimal"
fi
if [[ -n "$ATK_B" ]]; then
    ok "batch profile: $PROFILE (attack=$ATK_B, feature=$FEAT_B)"
    warn "smaller batches than the paper's GPU → slower, identical results"
else
    ok "batch profile: $PROFILE"
fi

FREE_GB=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$FREE_GB" ]]; then
    if (( FREE_GB >= 60 )); then ok "disk free: ${FREE_GB} GB"
    else warn "only ${FREE_GB} GB free; ~60 GB recommended"; fi
fi

# ------------------------------------------------------------------ pipeline
step "1/6  Environment"
uv sync --group experiments || die "uv sync failed"
ok "dependencies installed"

step "2/6  Checkpoints  (~250 MB, HuggingFace)"
uv run python experiments/01_download.py --core-only \
    || die "checkpoint download failed — check network access to huggingface.co"

step "3/6  Adversarial generation  ($N_MODELS model(s) x $N_ATTACKS attack(s))"
uv run python experiments/03_gen_adversarial.py \
        --models $MODELS --attacks $ATTACKS ${ATK_B:+--batch $ATK_B} \
    || die "adversarial generation failed.
   If this was a CUDA out-of-memory, retry with a smaller batch:
       uv run python experiments/03_gen_adversarial.py --models $MODELS --attacks $ATTACKS --batch 8"

step "4/6  Feature extraction  (ID + $N_OOD OOD + $N_ATTACKS ADV per model)"
uv run python experiments/06b_extract_full.py \
        --models $MODELS --attacks $ATTACKS $OOD_ARG ${FEAT_B:+--batch $FEAT_B} \
    || die "feature extraction failed.
   If this was a CUDA out-of-memory, retry with a smaller batch:
       uv run python experiments/06b_extract_full.py --models $MODELS --attacks $ATTACKS $OOD_ARG --batch 64"

# Extraction warns-and-continues when an OOD split fails, so exit status alone
# does not prove coverage. A missing split silently changes T3.
step "5/6  Verifying extraction coverage"
EXPECTED=$(( N_MODELS * (1 + N_OOD + N_ATTACKS) ))
GOT=$(ls results/features/featfull_*.h5 2>/dev/null | wc -l)
if (( GOT < EXPECTED )); then
    warn "expected $EXPECTED feature files, found $GOT — a split failed."
    warn "(GTSRB's upstream mirror has intermittent outages; usual cause.)"
    warn "T2 is unaffected; T3 would average over fewer OOD sets than the paper."
    warn "Re-running this script retries only the missing splits."
else
    ok "all $EXPECTED feature files present"
fi

step "6/6  Signatures, evaluation, comparison table"
uv run python experiments/09_signatures_full.py --models $MODELS \
    || die "signature recomputation failed"
# Step 9 gates models on a clean-accuracy file that the T1 path does not
# produce; without a gate bypass it analyses nothing yet still exits 0.
SIGS=$(ls results/analysis/signature_auroc_full_*.csv 2>/dev/null | wc -l)
if (( SIGS < N_MODELS )); then
    die "step 9 wrote $SIGS of $N_MODELS signature CSVs.
   It exited cleanly but analysed fewer models than requested -- check for a
   [gate] line in the output above."
fi
ok "$SIGS signature CSV(s) written"
uv run python experiments/full_eval.py --dataset cifar100 --models $MODELS \
    || die "full_eval failed"

# full_eval prints 'no complete features — skip' and exits 0 when its inputs
# are incomplete. Check the artifact it should have produced, not its status.
SUMMARY=results/analysis/full_eval_cifar100_summary.csv
[[ -s "$SUMMARY" ]] || die "$SUMMARY was not written.
   full_eval exited cleanly but produced nothing, meaning it found no complete
   feature set. Review step 5's coverage warnings above."
ok "$(basename "$SUMMARY") written"

uv run python experiments/exp_master_table.py --dataset cifar100 \
    || warn "master comparison table failed (secondary output; core results intact)"

# ------------------------------------------------------------------ verdict
step "Result"

if (( QUICK )); then
    cat <<EOF
  ${GRN}QUICK smoke tier completed successfully.${RST}

  The full pipeline ran on your machine: checkpoints fetched, adversarial
  examples generated, first-conv features extracted, signatures recomputed,
  evaluation written.

  ${YEL}These AUROCs are not comparable to the paper${RST} — one backbone, one
  attack, two OOD sets. To reproduce the paper's Tier T1 numbers:

      bash reproduce_t1.sh

  Output: $SUMMARY
EOF
    exit 0
fi

uv run python - "$SUMMARY" <<'PY'
import csv, sys, pathlib

TOL = 0.02                      # seed / hardware nondeterminism
REF = {"T2": 0.9804, "T3": 0.8441}

path = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader(path.open()))
hits = [r for r in rows
        if "viyogd" in " ".join(map(str, r.values())).lower()
        and "tv_dorm" in " ".join(map(str, r.values())).lower()]

print("  Reference (README/STATUS — 6-model core tier, 4 attacks):")
print(f"      T2 (ID-vs-ADV)   ~{REF['T2']}")
print(f"      T3 (OOD-vs-ADV)  ~{REF['T3']}")
print(f"  Tolerance: +/-{TOL} AUROC\n")

if not hits:
    print("  note  could not identify the ViyogD_tv_dorm row automatically.")
    print(f"        Open {path} and compare it against the table above.")
    sys.exit(0)

for r in hits:
    print("   ", {k: v for k, v in r.items() if v not in (None, "")})

print("\n  Within tolerance  -> reproduction PASSED.")
print("  Outside tolerance -> please report it with this script's full output.")
PY

cat <<EOF

${BLD}------------------------------------------------------------------${RST}
 Detailed numbers:
   results/analysis/full_eval_cifar100_summary.csv
   results/analysis/master_comparison_cifar100.csv

 The two efficiency headlines (~0.3 KB state, ~2.28% of a forward pass)
 are architecture-only — no GPU, weights or data needed:
   uv run python experiments/eval_detector_cost.py
   uv run python experiments/eval_systems.py
${BLD}------------------------------------------------------------------${RST}
EOF
