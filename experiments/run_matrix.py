"""Matrix orchestrator: finetune → attack → extract → analyse, across datasets.

Runs the full pipeline for a {dataset} × {model} matrix with optimal GPU use:

  Stage A (per model, GPU-parallel):  00 finetune → 02 eval → 03 attack → 06 extract
  Stage B (per dataset, after its models): 07 analyse → 08 signatures (aggregate)

One (dataset, model) job runs per GPU; a free GPU is refilled from the queue as
soon as a job finishes. Every underlying step is skip-if-exists, so the whole
matrix is resumable — re-run after a crash and completed work is skipped.

OOD raw data (shared across datasets) is downloaded once up front via step 05.

Examples:
    # plan only (no execution)
    python experiments/run_matrix.py --datasets cifar10 svhn gtsrb mnist --gpus 0 1 4 5 --dry-run

    # run the matrix, fast attacks first
    python experiments/run_matrix.py --datasets cifar10 svhn gtsrb mnist eurosat pets \
        --gpus 0 1 4 5 --attacks fgsm bim pgd apgd_ce

    # include the slow attacks too (deepfool/cw), and skip finetuning if weights exist
    python experiments/run_matrix.py --datasets cifar10 --gpus 0 1 --attacks fgsm bim pgd apgd_ce deepfool cw
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import config

HERE = Path(__file__).resolve().parent
# Use the venv launcher SYMLINK (not sys.executable, which resolves through the
# symlink to the bare uv interpreter and loses the venv's site-packages / timm).
_VENV_PY = HERE.parent / ".venv" / "bin" / "python"
PY = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
LOG_DIR = config.ROOT / "logs" / "matrix"
FAST_ATTACKS = ["fgsm", "bim", "pgd", "apgd_ce"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset×model pipeline orchestrator")
    p.add_argument("--datasets", nargs="+", required=True,
                   choices=list(config.DATASET_SPECS))
    p.add_argument("--models", nargs="+", default=list(config.MODEL_ARCHS),
                   choices=list(config.MODEL_ARCHS))
    p.add_argument("--gpus", nargs="+", type=int, required=True,
                   help="Physical GPU ids to schedule on")
    p.add_argument("--jobs-per-gpu", type=int, default=1,
                   help="Concurrent model-chains per physical GPU (pack to fill VRAM)")
    p.add_argument("--attacks", nargs="+", default=FAST_ATTACKS,
                   choices=list(config.ATTACKS))
    p.add_argument("--full", action="store_true",
                   help="Use the rich one-pass extractor (06b) + full battery (09) "
                        "instead of 06 + 07/08")
    p.add_argument("--cleanup-adv", action="store_true",
                   help="Delete each adversarial source h5 after 06b extracts its "
                        "signatures (bounds disk for multi-dataset runs; --full only)")
    p.add_argument("--skip-finetune", action="store_true",
                   help="Assume weights already exist; skip step 00")
    p.add_argument("--skip-ood-prep", action="store_true",
                   help="Skip step 05 (OOD download) — use if data is already present")
    p.add_argument("--seed", type=int, default=None,
                   help="finetune seed; namespaces outputs via VIYOG_* env (default: seed 0, canonical paths)")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit")
    return p.parse_args()


def _model_chain(dataset: str, model: str, attacks: list[str], skip_ft: bool,
                 full: bool, cleanup_adv: bool = False, seed: int | None = None) -> list[list[str]]:
    """The ordered Stage-A commands for one (dataset, model)."""
    extract = "06b_extract_full.py" if full else "06_extract_features.py"
    extract_cmd = [PY, extract, "--dataset", dataset, "--models", model, "--attacks", *attacks]
    if cleanup_adv and full:
        extract_cmd.append("--cleanup-adv")
    cmds = []
    if not skip_ft:
        ft = [PY, "00_finetune.py", "--dataset", dataset, "--models", model]
        if seed is not None:
            ft += ["--seed", str(seed)]
        cmds.append(ft)
    cmds += [
        [PY, "02_eval_clean.py", "--dataset", dataset, "--models", model],
        [PY, "03_gen_adversarial.py", "--dataset", dataset, "--models", model, "--attacks", *attacks],
        extract_cmd,
    ]
    return cmds


def _run_chain_bg(cmds: list[list[str]], gpu: int, log_path: Path) -> subprocess.Popen:
    """Launch a chain of commands (joined with &&) as one background process on `gpu`."""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    joined = " && ".join(subprocess.list2cmdline(c) for c in cmds)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_path, "w")
    fh.write(f"# GPU {gpu}\n# {joined}\n\n")
    fh.flush()
    return subprocess.Popen(joined, shell=True, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)


def _run_blocking(cmd: list[str], gpu: int, log_path: Path) -> int:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as fh:
        return subprocess.call(cmd, cwd=HERE, env=env, stdout=fh, stderr=subprocess.STDOUT)


def run_dataset(dataset: str, models: list[str], gpus: list[int],
                attacks: list[str], skip_ft: bool, full: bool,
                jobs_per_gpu: int = 1, cleanup_adv: bool = False, seed: int | None = None) -> None:
    """Stage A (model jobs across GPUs) then Stage B (aggregate) for one dataset.

    Scheduling is SLOT-based: each physical GPU exposes `jobs_per_gpu` slots, so
    several chains run concurrently on one GPU to fill its VRAM. A slot is
    (gpu, k); the chain still pins CUDA_VISIBLE_DEVICES to the physical gpu.
    """
    # Round-robin by GPU (outer loop = slot index) so models spread evenly across
    # GPUs first — [(g0,0),(g1,0),(g2,0),(g3,0),(g0,1),...] — instead of filling one
    # GPU's slots before the next (which starves the last GPUs).
    slots = [(g, k) for k in range(jobs_per_gpu) for g in gpus]
    print(f"\n##### dataset={dataset}  models={len(models)}  gpus={gpus} "
          f"x{jobs_per_gpu}/gpu = {len(slots)} slots #####")
    queue = list(models)
    running: dict[tuple[int, int], tuple[str, subprocess.Popen]] = {}  # slot -> (model, proc)
    free = list(slots)

    while queue or running:
        # Fill free slots from the queue.
        while queue and free:
            slot = free.pop(0)
            gpu = slot[0]
            model = queue.pop(0)
            log = LOG_DIR / (f"{dataset}_s{seed}_{model}.log" if seed is not None
                             else f"{dataset}_{model}.log")
            proc = _run_chain_bg(
                _model_chain(dataset, model, attacks, skip_ft, full, cleanup_adv, seed), gpu, log)
            running[slot] = (model, proc)
            print(f"  [launch] {dataset}/{model} on GPU {gpu} (slot {slot[1]})  → {log}")
        # Poll for finished jobs.
        for slot, (model, proc) in list(running.items()):
            rc = proc.poll()
            if rc is not None:
                tag = "ok" if rc == 0 else f"FAIL rc={rc}"
                print(f"  [done]   {dataset}/{model} on GPU {slot[0]}  ({tag})")
                del running[slot]
                free.append(slot)
        time.sleep(10)

    # Stage B — aggregate analysis over all models for this dataset (one GPU).
    agpu = gpus[0]
    steps = ("09_signatures_full.py",) if full else ("07_analyze.py", "08_signatures.py")
    print(f"  [aggregate] {', '.join(steps)} for {dataset} on GPU {agpu}")
    for step in steps:
        rc = _run_blocking([PY, step, "--dataset", dataset], agpu,
                           LOG_DIR / f"{dataset}_{step.split('_')[0]}.log")
        print(f"    {step} rc={rc}")


def main() -> None:
    args = _parse_args()

    print("=== Pipeline matrix plan ===")
    print(f"  datasets: {args.datasets}")
    print(f"  models:   {args.models}")
    print(f"  gpus:     {args.gpus}")
    print(f"  attacks:  {args.attacks}")
    print(f"  finetune: {'SKIP' if args.skip_finetune else 'yes'}")
    print(f"  jobs:     {len(args.datasets) * len(args.models)} (dataset×model) Stage-A chains")
    print(f"  logs:     {LOG_DIR}/")
    if args.dry_run:
        for d in args.datasets:
            for m in args.models:
                chain = _model_chain(d, m, args.attacks, args.skip_finetune, args.full,
                                     args.cleanup_adv, args.seed)
                print(f"\n  {d}/{m}:")
                for c in chain:
                    print("    " + subprocess.list2cmdline(c))
        return

    if not args.skip_ood_prep:
        print("\n  [step 05] downloading OOD universe (once)…")
        _run_blocking([PY, "05_prep_ood.py"], args.gpus[0], LOG_DIR / "_ood_prep.log")

    for dataset in args.datasets:
        run_dataset(dataset, args.models, args.gpus, args.attacks, args.skip_finetune,
                    args.full, args.jobs_per_gpu, args.cleanup_adv, args.seed)

    print("\n=== matrix complete ===")
    for d in args.datasets:
        sig = config.dataset_dirs(d)["analysis"] / "signatures.json"
        print(f"  {d}: {'✓' if sig.exists() else '—'} {sig}")


if __name__ == "__main__":
    main()
