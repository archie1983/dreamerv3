import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_metrics(file_path):
    """Load metrics from JSON file where each line is a separate JSON object"""
    metrics = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                metrics.append(json.loads(line))
    return pd.DataFrame(metrics)

def plot_loss_components(df, figsize=(15, 10)):
    """Plot all loss components over training steps"""
    loss_columns = [col for col in df.columns if 'train/loss/' in col]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Main losses
    main_losses = ['train/loss/con', 'train/loss/dyn', 'train/loss/image', 'train/loss/policy']
    for i, loss in enumerate(main_losses):
        if loss in df.columns:
            axes[i].plot(df['step'], df[loss], marker='o', linewidth=2, markersize=4)
            axes[i].set_title(f'{loss.replace("train/loss/", "").title()} Loss', fontsize=12)
            axes[i].set_xlabel('Training Steps')
            axes[i].set_ylabel('Loss Value')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Loss Components Over Training', fontsize=16, y=1.02)
    plt.show()

def plot_performance_metrics(df, figsize=(15, 8)):
    """Plot key performance metrics"""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    metrics = [
        ('train/ret', 'Return', 'green'),
        ('train/rew', 'Reward', 'blue'),
        ('train/con', 'Consistency', 'orange'),
        ('train/val', 'Value', 'red'),
        ('train/tar', 'Target', 'purple'),
        ('train/weight', 'Weight', 'brown')
    ]
    
    for i, (metric, title, color) in enumerate(metrics):
        if metric in df.columns:
            axes[i].plot(df['step'], df[metric], marker='o', color=color, linewidth=2, markersize=4)
            axes[i].set_title(title, fontsize=12)
            axes[i].set_xlabel('Training Steps')
            axes[i].set_ylabel('Value')
            axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Performance Metrics Over Training', fontsize=16, y=1.02)
    plt.show()

def plot_optimization_metrics(df, figsize=(15, 6)):
    """Plot optimization-related metrics"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Gradient norm
    if 'train/opt/grad_norm' in df.columns:
        axes[0].plot(df['step'], df['train/opt/grad_norm'], marker='o', color='red', linewidth=2, markersize=4)
        axes[0].set_title('Gradient Norm', fontsize=12)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Gradient Norm')
        axes[0].grid(True, alpha=0.3)
    
    # Learning rate related metrics
    if 'train/opt/grad_rms' in df.columns:
        axes[1].plot(df['step'], df['train/opt/grad_rms'], marker='o', color='blue', linewidth=2, markersize=4)
        axes[1].set_title('Gradient RMS', fontsize=12)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('RMS Value')
        axes[1].grid(True, alpha=0.3)
    
    # Overall optimization loss
    if 'train/opt/loss' in df.columns:
        axes[2].plot(df['step'], df['train/opt/loss'], marker='o', color='green', linewidth=2, markersize=4)
        axes[2].set_title('Optimization Loss', fontsize=12)
        axes[2].set_xlabel('Training Steps')
        axes[2].set_ylabel('Loss Value')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Optimization Metrics', fontsize=16, y=1.02)
    plt.show()

def plot_system_usage(df, figsize=(15, 10)):
    """Plot system resource usage"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # CPU Usage
    if 'usage/psutil/total_cpu_frac' in df.columns:
        axes[0, 0].plot(df['step'], df['usage/psutil/total_cpu_frac'] * 100, marker='o', color='blue', linewidth=2, markersize=4)
        axes[0, 0].set_title('CPU Usage (%)', fontsize=12)
        axes[0, 0].set_xlabel('Training Steps')
        axes[0, 0].set_ylabel('CPU Usage (%)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # RAM Usage
    if 'usage/psutil/proc_ram_gb' in df.columns:
        axes[0, 1].plot(df['step'], df['usage/psutil/proc_ram_gb'], marker='o', color='green', linewidth=2, markersize=4)
        axes[0, 1].set_title('Process RAM Usage (GB)', fontsize=12)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('RAM (GB)')
        axes[0, 1].grid(True, alpha=0.3)
    
    # GPU Compute Usage
    gpu_compute_cols = [col for col in df.columns if 'usage/nvsmi/compute_avg' in col]
    if gpu_compute_cols:
        for i, col in enumerate(gpu_compute_cols):
            gpu_num = col.split('/')[-1]
            axes[1, 0].plot(df['step'], df[col] * 100, marker='o', label=f'{gpu_num.upper()}', linewidth=2, markersize=4)
        axes[1, 0].set_title('GPU Compute Usage (%)', fontsize=12)
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Compute Usage (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # GPU Memory Usage
    gpu_memory_cols = [col for col in df.columns if 'usage/nvsmi/memory_avg' in col]
    if gpu_memory_cols:
        for i, col in enumerate(gpu_memory_cols):
            gpu_num = col.split('/')[-1]
            axes[1, 1].plot(df['step'], df[col] * 100, marker='o', label=f'{gpu_num.upper()}', linewidth=2, markersize=4)
        axes[1, 1].set_title('GPU Memory Usage (%)', fontsize=12)
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Memory Usage (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('System Resource Usage', fontsize=16, y=1.02)
    plt.show()

def plot_replay_buffer(df, figsize=(12, 8)):
    """Plot replay buffer statistics"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    replay_metrics = [
        ('replay/items', 'Replay Items', 'blue'),
        ('replay/replay_ratio', 'Replay Ratio', 'green'),
        ('replay/updates', 'Replay Updates', 'orange'),
        ('replay/ram_gb', 'Replay RAM (GB)', 'red')
    ]
    
    for i, (metric, title, color) in enumerate(replay_metrics):
        if metric in df.columns:
            row, col = i // 2, i % 2
            axes[row, col].plot(df['step'], df[metric], marker='o', color=color, linewidth=2, markersize=4)
            axes[row, col].set_title(title, fontsize=12)
            axes[row, col].set_xlabel('Training Steps')
            axes[row, col].set_ylabel('Value')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Replay Buffer Statistics', fontsize=16, y=1.02)
    plt.show()

def plot_fps_metrics(df, figsize=(10, 6)):
    """Plot FPS metrics"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if 'fps/policy' in df.columns:
        axes[0].plot(df['step'], df['fps/policy'], marker='o', color='blue', linewidth=2, markersize=4)
        axes[0].set_title('Policy FPS', fontsize=12)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('FPS')
        axes[0].grid(True, alpha=0.3)
    
    if 'fps/train' in df.columns:
        axes[1].plot(df['step'], df['fps/train'], marker='o', color='green', linewidth=2, markersize=4)
        axes[1].set_title('Training FPS', fontsize=12)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('FPS')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Performance FPS', fontsize=16, y=1.02)
    plt.show()

def plot_entropy_metrics(df, figsize=(10, 6)):
    """Plot entropy-related metrics"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if 'train/dyn_ent' in df.columns:
        axes[0].plot(df['step'], df['train/dyn_ent'], marker='o', color='purple', linewidth=2, markersize=4)
        axes[0].set_title('Dynamics Entropy', fontsize=12)
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Entropy')
        axes[0].grid(True, alpha=0.3)
    
    if 'train/rep_ent' in df.columns:
        axes[1].plot(df['step'], df['train/rep_ent'], marker='o', color='orange', linewidth=2, markersize=4)
        axes[1].set_title('Representation Entropy', fontsize=12)
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Entropy')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Entropy Metrics', fontsize=16, y=1.02)
    plt.show()

def generate_summary_table(df):
    """Generate a summary table of key metrics"""
    if len(df) < 2:
        print("Not enough data points for comparison")
        return
    
    first_step = df.iloc[0]
    last_step = df.iloc[-1]
    
    key_metrics = [
        'train/ret', 'train/rew', 'train/con', 'train/loss/image', 
        'train/loss/con', 'train/loss/policy', 'train/opt/grad_norm',
        'fps/policy', 'fps/train'
    ]
    
    print(f"\n{'='*60}")
    print(f"TRAINING SUMMARY (Step {first_step['step']} → Step {last_step['step']})")
    print(f"{'='*60}")
    
    for metric in key_metrics:
        if metric in df.columns:
            initial = first_step[metric]
            final = last_step[metric]
            if initial != 0:
                change_pct = ((final - initial) / abs(initial)) * 100
                direction = "↑" if change_pct > 0 else "↓"
                print(f"{metric:25} {initial:10.6f} → {final:10.6f} ({direction}{abs(change_pct):6.1f}%)")
            else:
                print(f"{metric:25} {initial:10.6f} → {final:10.6f}")

def main():
    """Main function to load data and generate all plots"""
    # Load the data
    file_path = "/home/elksnis/logdir/20250716T225615/metrics.jsonl"  # Change this to your file path
    
    try:
        df = load_metrics(file_path)
        print(f"Loaded {len(df)} training steps from {file_path}")
        print(f"Steps range: {df['step'].min()} to {df['step'].max()}")
        
        # Generate all plots
        plot_loss_components(df)
        plot_performance_metrics(df)
        plot_optimization_metrics(df)
        plot_system_usage(df)
        plot_replay_buffer(df)
        plot_fps_metrics(df)
        plot_entropy_metrics(df)
        
        # Generate summary table
        generate_summary_table(df)
        
    except FileNotFoundError:
        print(f"File {file_path} not found. Please check the path.")
    except Exception as e:
        print(f"Error loading data: {e}")

# Run the analysis
if __name__ == "__main__":
    main()

# If you want to run individual plots, you can also do:
# df = load_metrics("metrics.json")
# plot_loss_components(df)
# plot_performance_metrics(df)
# etc...
