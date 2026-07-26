#!/bin/bash
# Autonomous driver for the remaining CIFAR-100 OOD/ADV pipeline.
# No GPU migration: pmon shows ConvNeXt=88% SM, ViT=99% SM — both compute-bound
# at the hardware limit, so moving them only loses in-progress work. Let them
# finish in place. EffNet/Swin low SM is DeepFool (CPU-bound per-sample), not placement.
# Then Step 4 (adv eval) + Step 6 (features) parallel by model; Step 7 analysis.
set -u
ROOT=/mnt/data1/asing725/viyog
EXP=$ROOT/Seperating_OOD_and_ADV
LOG=$ROOT/logs
ADV=$ROOT/data/adversarial
PY=$EXP/.venv/bin/python
cd "$EXP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ts(){ date +%H:%M:%S; }
running(){ pgrep -f "03_gen_adversarial.py --models $1" >/dev/null; }
say(){ echo "[$(ts)] $*"; }

say "ORCH start (no-migration; jobs are compute-bound at hw limit)"

# ---- Phase B: wait for all step-3 jobs to finish ----
for m in efficientnetv2_l swin_tiny convnextv2_base vit_base; do
  say "waiting for $m..."
  while running "$m"; do sleep 20; done
  say "$m DONE"
done

say "===== STEP 3 COMPLETE ====="
ls -la --block-size=M "$ADV"/*.h5 2>/dev/null | awk '{print $5, $9}'
n=$(ls "$ADV"/*.h5 2>/dev/null | wc -l)
say "$n/24 adversarial files present"

# ---- Step 4: adversarial accuracy eval (forward-only, parallel by model) ----
# Use the GPUs each model finished on (free by now): effnet/swin->5, vit->1, convnext->0
say "===== STEP 4: adversarial eval ====="
CUDA_VISIBLE_DEVICES=5 $PY experiments/04_eval_adversarial.py --models efficientnetv2_l > "$LOG/s4_effnet.log"   2>&1 &  A=$!
CUDA_VISIBLE_DEVICES=1 $PY experiments/04_eval_adversarial.py --models vit_base        > "$LOG/s4_vit.log"      2>&1 &  B=$!
CUDA_VISIBLE_DEVICES=5 $PY experiments/04_eval_adversarial.py --models swin_tiny       > "$LOG/s4_swin.log"     2>&1 &  C=$!
CUDA_VISIBLE_DEVICES=0 $PY experiments/04_eval_adversarial.py --models convnextv2_base > "$LOG/s4_convnext.log" 2>&1 &  D=$!
wait $A $B $C $D
say "STEP 4 done"

# ---- Step 6: first-layer feature extraction (parallel by model) ----
say "===== STEP 6: feature extraction ====="
CUDA_VISIBLE_DEVICES=5 $PY experiments/06_extract_features.py --models efficientnetv2_l > "$LOG/s6_effnet.log"   2>&1 &  E=$!
CUDA_VISIBLE_DEVICES=1 $PY experiments/06_extract_features.py --models vit_base        > "$LOG/s6_vit.log"      2>&1 &  F=$!
CUDA_VISIBLE_DEVICES=5 $PY experiments/06_extract_features.py --models swin_tiny       > "$LOG/s6_swin.log"     2>&1 &  G=$!
CUDA_VISIBLE_DEVICES=0 $PY experiments/06_extract_features.py --models convnextv2_base > "$LOG/s6_convnext.log" 2>&1 &  H=$!
wait $E $F $G $H
say "STEP 6 done"

# ---- Step 7: neuron analysis + plots (CPU) ----
say "===== STEP 7: analysis ====="
$PY experiments/07_analyze.py > "$LOG/s7_analyze.log" 2>&1
say "STEP 7 done"

# ---- Step 8: statistical signature battery + AUROC/3-way detector (CPU) ----
say "===== STEP 8: signature battery ====="
$PY experiments/08_signatures.py > "$LOG/s8_signatures.log" 2>&1
say "STEP 8 done"
say "===== PIPELINE COMPLETE ====="
