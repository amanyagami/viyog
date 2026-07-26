#!/usr/bin/env bash
# 3-GPU pipeline runner for GPUs 5, 6, 7.
#
# GPU assignment (optimised for H200 126 GB VRAM):
#   GPU 5 → efficientnetv2_l alone  (largest model; batches sized to fill VRAM)
#   GPU 6 → convnextv2_base alone   (medium model)
#   GPU 7 → vit_base + swin_tiny    (sequential; both small, comfortably fit)
#
# Steps 3 + 5 run fully in parallel: adversarial generation on 3 GPUs and
# OOD dataset download on CPU/network happen simultaneously.
# Steps 2, 4, 7 are serial (< 5 min each; shared result JSON avoids races).
# Step 6 (feature extraction) parallelises across 3 GPUs like Step 3.
#
# Each parallel job is redirected to logs/<name>.log so $! captures the
# Python PID directly (no tee pipe that would hide the true exit code).
# Monitor progress with: tail -f logs/adv_gpu5.log
#
# Usage:
#   cd /mnt/data1/asing725/viyog/Seperating_OOD_and_ADV
#   bash experiments/run_pipeline.sh                    # full pipeline
#   bash experiments/run_pipeline.sh 03_gen_adversarial # single step on GPU 5

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJ_DIR/logs"
mkdir -p "$LOG_DIR"

# GPU selection (verified 2026-06-08 via nvidia-smi):
#   GPU 5: 135 GB free  → EfficientNetV2-L alone
#   GPU 0: 121 GB free  → ConvNeXtV2-Base alone
#   GPU 4:  70 GB free  → ViT-Base + Swin-Tiny (sequential, combined <70 GB peak)
#   GPU 6/7: ~5 GB free → occupied, excluded
GPU_A=5   # EfficientNetV2-L
GPU_B=0   # ConvNeXtV2-Base
GPU_C=4   # ViT-Base + Swin-Tiny

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
run_serial() {
    # run_serial <step_name> <gpu_id> [extra python args...]
    local step="$1"; local gpu="$2"; shift 2
    echo ""
    echo "=========================================="
    echo "  Running: ${step}  (GPU ${gpu})"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=$gpu uv run --project "$PROJ_DIR" \
        python "$SCRIPT_DIR/${step}.py" "$@"
}

wait_pids() {
    # Wait for named background jobs; exit 1 if any failed.
    # Redirect-to-file (not pipe) ensures $PID = Python PID, not tee PID.
    local label="$1"; shift
    local rc=0
    for pid in "$@"; do
        wait "$pid" || { echo "ERROR: background job for '$label' failed (pid=$pid)"; rc=1; }
    done
    [[ $rc -eq 0 ]] || exit 1
}

# ---------------------------------------------------------------------------
# Single-step shortcut  (e.g. bash run_pipeline.sh 03_gen_adversarial)
# ---------------------------------------------------------------------------
if [[ $# -gt 0 ]]; then
    run_serial "$1" $GPU_A
    exit 0
fi

# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

# ── Step 1: download weights (network-bound, serial) ──────────────────────
run_serial 01_download $GPU_A

# ── Step 2: clean accuracy (fast, serial, writes shared JSON) ─────────────
run_serial 02_eval_clean $GPU_A

# ── Steps 3 + 5: adversarial gen (3 GPUs) + OOD download (CPU) ────────────
echo ""
echo "=========================================="
echo "  Running: 03_gen_adversarial (3 GPUs) + 05_prep_ood (CPU)"
echo "  GPU $GPU_A → efficientnetv2_l     log: logs/adv_gpu${GPU_A}.log"
echo "  GPU $GPU_B → convnextv2_base      log: logs/adv_gpu${GPU_B}.log"
echo "  GPU $GPU_C → vit_base swin_tiny   log: logs/adv_gpu${GPU_C}.log"
echo "  CPU       → OOD download          log: logs/ood_download.log"
echo "  Monitor:  tail -f logs/adv_gpu${GPU_A}.log"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_A uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/03_gen_adversarial.py" --models efficientnetv2_l \
    &> "$LOG_DIR/adv_gpu${GPU_A}.log" &
PID_ADV_A=$!

CUDA_VISIBLE_DEVICES=$GPU_B uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/03_gen_adversarial.py" --models convnextv2_base \
    &> "$LOG_DIR/adv_gpu${GPU_B}.log" &
PID_ADV_B=$!

CUDA_VISIBLE_DEVICES=$GPU_C uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/03_gen_adversarial.py" --models vit_base swin_tiny \
    &> "$LOG_DIR/adv_gpu${GPU_C}.log" &
PID_ADV_C=$!

uv run --project "$PROJ_DIR" python "$SCRIPT_DIR/05_prep_ood.py" \
    &> "$LOG_DIR/ood_download.log" &
PID_OOD=$!

wait_pids "03_gen_adversarial + 05_prep_ood" $PID_ADV_A $PID_ADV_B $PID_ADV_C $PID_OOD
echo "=== Steps 3 + 5 complete ==="

# ── Step 4: adversarial accuracy (serial — writes single shared JSON) ──────
run_serial 04_eval_adversarial $GPU_A

# ── Step 6: feature extraction (3 GPUs, same assignment as Step 3) ─────────
echo ""
echo "=========================================="
echo "  Running: 06_extract_features (3 GPUs)"
echo "  GPU $GPU_A → efficientnetv2_l     log: logs/feat_gpu${GPU_A}.log"
echo "  GPU $GPU_B → convnextv2_base      log: logs/feat_gpu${GPU_B}.log"
echo "  GPU $GPU_C → vit_base swin_tiny   log: logs/feat_gpu${GPU_C}.log"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU_A uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/06_extract_features.py" --models efficientnetv2_l \
    &> "$LOG_DIR/feat_gpu${GPU_A}.log" &
PID_FEAT_A=$!

CUDA_VISIBLE_DEVICES=$GPU_B uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/06_extract_features.py" --models convnextv2_base \
    &> "$LOG_DIR/feat_gpu${GPU_B}.log" &
PID_FEAT_B=$!

CUDA_VISIBLE_DEVICES=$GPU_C uv run --project "$PROJ_DIR" \
    python "$SCRIPT_DIR/06_extract_features.py" --models vit_base swin_tiny \
    &> "$LOG_DIR/feat_gpu${GPU_C}.log" &
PID_FEAT_C=$!

wait_pids "06_extract_features" $PID_FEAT_A $PID_FEAT_B $PID_FEAT_C
echo "=== Step 6 complete ==="

# ── Step 7: analysis + plots (CPU, serial) ────────────────────────────────
run_serial 07_analyze $GPU_A

echo ""
echo "=========================================="
echo "  Pipeline complete."
echo "  Results:  $PROJ_DIR/results/"
echo "  Logs:     $LOG_DIR/"
echo "=========================================="
