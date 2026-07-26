#!/usr/bin/env bash
# Autonomous completion + recovery + analysis for the Core-6 cifar10/gtsrb run.
# Waits for the two orchestrators to finish, recovers any model missing its
# featfull_id (re-run at 1/gpu, skip-finetune — weights intact, no OOM), verifies,
# then runs the cross-dataset analysis + baselines + figures.
set +e
EXP=/mnt/data1/asing725/viyog/Seperating_OOD_and_ADV/experiments
VENV=/mnt/data1/asing725/viyog/Seperating_OOD_and_ADV/.venv/bin/python
LOG=/mnt/data1/asing725/viyog/logs/finish_analyze.log
FD=/mnt/data1/asing725/viyog/results
ADV=/mnt/data1/asing725/viyog/data/adversarial
CORE6="resnet50 densenet121 convnextv2_base vit_base swin_tiny mobilenetv3_l"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HDF5_USE_FILE_LOCKING=FALSE
cd "$EXP" || exit 1

echo "=== finish_and_analyze START $(date '+%F %T') ===" >> "$LOG"

# 1) Wait for both run_matrix orchestrators to exit.
while ps -u "$USER" -o cmd | grep -E "[p]ython.*run_matrix" | grep -qv "bash -c"; do sleep 60; done
echo "[$(date '+%T')] orchestrators exited" >> "$LOG"
sleep 15

# 2) Recovery pass per dataset: any Core-6 model missing featfull_id → clean its
#    partial adv+feat, re-run chain at 1/gpu skip-finetune.
for d in cifar10 gtsrb; do
  miss=""
  for m in $CORE6; do
    [ -f "$FD/$d/features/featfull_${m}_id.h5" ] || miss="$miss $m"
  done
  if [ -n "$miss" ]; then
    echo "[$(date '+%T')] [$d] recovering:$miss" >> "$LOG"
    for m in $miss; do
      rm -f "$FD/$d/features/featfull_${m}_"*.h5 2>/dev/null
      rm -f "$ADV/$d/${m}_"*.h5 2>/dev/null
    done
    "$VENV" -u run_matrix.py --datasets "$d" --models $miss \
      --gpus 4 5 6 7 --jobs-per-gpu 1 --full --skip-ood-prep --skip-finetune --cleanup-adv \
      --attacks fgsm bim pgd apgd_ce >> "$LOG" 2>&1
  else
    echo "[$(date '+%T')] [$d] complete (6/6 featfull_id)" >> "$LOG"
  fi
done

# 3) Verify.
echo "=== VERIFY $(date '+%T') ===" >> "$LOG"
for d in cifar10 gtsrb; do
  n=$(ls "$FD/$d/features/featfull_"*_id.h5 2>/dev/null | wc -l)
  echo "  [$d] featfull_id = $n/6" >> "$LOG"
done

# 4) Analysis (GPUs now free).
echo "=== ANALYSIS $(date '+%T') ===" >> "$LOG"
# baselines Maha/KNN/ViM/ODIN vs Viyog on cifar100 Core-6 (adv kept)
CUDA_VISIBLE_DEVICES=4 "$VENV" -u baselines_feature.py --dataset cifar100 --models $CORE6 --n 2000 >> "$LOG" 2>&1
echo "[$(date '+%T')] baselines done" >> "$LOG"
# cross-dataset end-to-end recall + 3-way signature analysis
"$VENV" -u cascade_recall_full.py --datasets cifar10 gtsrb cifar100 >> "$LOG" 2>&1
for d in cifar10 gtsrb cifar100; do
  "$VENV" -u signature_3way_analysis.py --dataset "$d" >> "$LOG" 2>&1
  CUDA_VISIBLE_DEVICES="" "$VENV" -u audit_directions.py --dataset "$d" >> "$LOG" 2>&1
done
echo "[$(date '+%T')] cross-dataset analysis + direction audit done" >> "$LOG"
# figures
for d in cifar10 gtsrb cifar100; do
  CUDA_VISIBLE_DEVICES="" "$VENV" -u make_figs.py --dataset "$d" >> "$LOG" 2>&1
done
echo "=== finish_and_analyze COMPLETE $(date '+%F %T') ===" >> "$LOG"
