#!/usr/bin/env python3

import os
import subprocess
import glob

# Directory containing the checkpoints
checkpoint_dir = os.path.expanduser("~/robotics/Latent_space_planning/dreamer3_results/door_find_10k")
#/home/elksnis/robotics/Latent_space_planning/dreamer3_results/door_find_10k
base_log_dir = "/home/elksnis/logdir"

# Find all checkpoint directories
checkpoint_paths = [path for path in glob.glob(os.path.join(checkpoint_dir, "ckpt_*")) 
                   if os.path.isdir(path) and not path.endswith('.tar')]

print(f"Found {len(checkpoint_paths)} checkpoints to evaluate")

for checkpoint_path in checkpoint_paths:
    checkpoint_name = os.path.basename(checkpoint_path)
    log_dir = os.path.join(base_log_dir, f"{checkpoint_name}_door_eval")
    
    print(f"\nEvaluating checkpoint: {checkpoint_name}")
    print(f"Log directory: {log_dir}")
    print("=" * 50)
    
    # Build the command
    cmd = [
        "python", "dreamerv3/main.py",
        "--configs", "indoorsae_eval",
        "--run.envs", "3",
        "--batch_size", "3",
        "--run.from_checkpoint", checkpoint_path,
        "--logdir", log_dir
    ]
    
    # Run the command
    try:
        subprocess.run(cmd, check=True)
        print(cmd)
        print(f"✓ Successfully evaluated {checkpoint_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to evaluate {checkpoint_name}: {e}")
    
    print("=" * 50)

print("\nAll checkpoint evaluations completed!")
