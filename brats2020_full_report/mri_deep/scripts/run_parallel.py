"""
=============================================================================
Parallel Launcher — Run lambda_b training jobs concurrently on multiple GPUs
=============================================================================
Usage:
    # Run all 3 lambda_b values in parallel (each on a different GPU):
    python scripts/run_parallel.py --lambda_b 0.1 0.3 0.5

    # Run two specific values:
    python scripts/run_parallel.py --lambda_b 0.3 0.5 --gpus 0 1

    # Run with custom epochs:
    python scripts/run_parallel.py --lambda_b 0.1 0.5 --epochs 200

Requirements:
    - Multiple GPUs (one per lambda_b value)
    - Each job needs ~8-12 GB GPU memory (3D BraTS, batch_size=1)
    - Check GPU count: nvidia-smi
=============================================================================
"""

import os
import sys
import subprocess
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--lambda_b', type=float, nargs='+', required=True,
                    help='List of lambda_b values to train (e.g., 0.1 0.3 0.5)')
parser.add_argument('--gpus', type=int, nargs='+', default=None,
                    help='GPU IDs to use (e.g., 0 1 2). Default: auto-assign')
parser.add_argument('--epochs', type=int, default=200,
                    help='Max epochs per job (default: 200)')
args = parser.parse_args()

# Auto-detect GPUs
if args.gpus is None:
    try:
        nvidia_smi = subprocess.run(
            ['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        gpu_ids = [int(line.strip()) for line in nvidia_smi.stdout.strip().split('\n') if line.strip()]
        gpu_count = len(gpu_ids)
        args.gpus = list(range(min(len(args.lambda_b), gpu_count)))
    except:
        args.gpus = [0]  # fallback
        gpu_count = 1

if len(args.gpus) < len(args.lambda_b):
    print(f"[ERROR] Need {len(args.lambda_b)} GPUs, only {len(args.gpus)} available.")
    print(f"  Run: nvidia-smi  to check GPU count.")
    sys.exit(1)

print("=" * 60)
print("PARALLEL TRAINING LAUNCHER")
print("=" * 60)
print(f"lambda_b values: {args.lambda_b}")
print(f"GPU assignment:  {dict(zip(args.lambda_b, args.gpus))}")
print(f"Max epochs:      {args.epochs}")
print()

# Launch one process per lambda_b
processes = []
log_files = []

for lb, gpu_id in zip(args.lambda_b, args.gpus):
    script_path = os.path.join(os.path.dirname(__file__), 'train_enhanced.py')
    log_file = f'train_lb{lb}_gpu{gpu_id}.log'

    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable, script_path,
        '--lambda_b', str(lb),
        '--epochs', str(args.epochs),
    ]

    print(f"[LAUNCH] lambda_b={lb} -> GPU {gpu_id} | log: {log_file}")

    with open(log_file, 'w') as f:
        p = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    processes.append((lb, gpu_id, p))
    log_files.append(log_file)

print(f"\n{len(processes)} jobs running. Monitor with:")
for lb, gpu_id, _ in processes:
    print(f"  tail -f train_lb{lb}_gpu{gpu_id}.log")
print("\nWaiting for all jobs to complete...\n")

# Wait for all to finish
for lb, gpu_id, p in processes:
    p.wait()
    print(f"[DONE] lambda_b={lb} (GPU {gpu_id}) — exit code {p.returncode}")

print("\n" + "=" * 60)
print("ALL JOBS COMPLETE")
print("=" * 60)
for lb in args.lambda_b:
    ckpt_dir = f'/root/autodl-tmp/ResUNet_Enhanced_lb{lb}_model'
    print(f"  lambda_b={lb}: {ckpt_dir}")
print("\nReady for evaluation: notebooks/experiment_lambda_results.ipynb")
