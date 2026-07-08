"""
Export training curves and comparison figures for the report.

Requirements from project PDF:
- Training curves with mean, std for all relevant metrics
- Training graphs overlayed for 3 runs
- Comparative analysis: baseline vs custom, SAC vs PPO
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import json

# Get the base directory (Student folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Create figures directory
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'baseline': '#1f77b4', 'custom': '#2ca02c', 'extension': '#ff7f0e'}
TRIAL_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']


def load_tensorboard_scalar(log_dir, tag='rollout/ep_rew_mean'):
    """Load scalar data from TensorBoard event file."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    available_tags = ea.Tags().get('scalars', [])

    if tag not in available_tags:
        # Try alternative tags
        for alt_tag in available_tags:
            if 'rew' in alt_tag.lower() or 'reward' in alt_tag.lower():
                tag = alt_tag
                break
        else:
            return None, None

    try:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        return steps, values
    except Exception as e:
        print(f"Error loading {tag}: {e}")
        return None, None


def find_event_file(log_dir):
    """Find the TensorBoard event file in a log directory."""
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                return root
    return None


def load_experiment_data(experiment, algo, num_trials=3):
    """Load data for all trials of an experiment."""
    all_data = []

    for trial in range(1, num_trials + 1):
        log_path = os.path.join(LOGS_DIR, experiment, f"{algo}_trial{trial}")
        event_dir = find_event_file(log_path)

        if event_dir:
            steps, values = load_tensorboard_scalar(event_dir)
            if steps is not None and len(steps) > 0:
                all_data.append((steps, values, trial))
                print(f"  Loaded {experiment} trial {trial}: {len(steps)} points")
            else:
                print(f"  Warning: No data for {experiment} trial {trial}")
        else:
            print(f"  Warning: No event file for {experiment} trial {trial}")

    return all_data


def interpolate_to_common_steps(all_data, num_points=100):
    """Interpolate all trials to common step values for mean/std calculation."""
    if not all_data:
        return None, None, None

    # Find common step range
    min_step = max(data[0][0] for data in all_data)
    max_step = min(data[0][-1] for data in all_data)

    common_steps = np.linspace(min_step, max_step, num_points)
    interpolated_values = []

    for steps, values, _ in all_data:
        interp_vals = np.interp(common_steps, steps, values)
        interpolated_values.append(interp_vals)

    interpolated_values = np.array(interpolated_values)
    mean_values = np.mean(interpolated_values, axis=0)
    std_values = np.std(interpolated_values, axis=0)

    return common_steps, mean_values, std_values


def plot_single_experiment(experiment, algo, ax, color, show_trials=True):
    """Plot training curves for a single experiment with mean and std."""
    print(f"\nLoading {experiment} ({algo})...")
    all_data = load_experiment_data(experiment, algo)

    if not all_data:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return

    # Plot individual trials
    if show_trials:
        for steps, values, trial in all_data:
            ax.plot(steps, values, alpha=0.3, color=TRIAL_COLORS[trial-1],
                   label=f'Trial {trial}', linewidth=1)

    # Plot mean with std
    common_steps, mean_values, std_values = interpolate_to_common_steps(all_data)

    if common_steps is not None:
        ax.plot(common_steps, mean_values, color=color, linewidth=2, label='Mean')
        ax.fill_between(common_steps, mean_values - std_values, mean_values + std_values,
                       color=color, alpha=0.2, label='±1 Std')

    ax.set_xlabel('Training Steps', fontsize=10)
    ax.set_ylabel('Episode Reward', fontsize=10)
    ax.set_title(f'{experiment.capitalize()} ({algo})', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)


def figure1_individual_training_curves():
    """Figure 1: Individual training curves for each experiment."""
    print("\n" + "="*60)
    print("FIGURE 1: Individual Training Curves")
    print("="*60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    experiments = [('baseline', 'SAC'), ('custom', 'SAC'), ('extension', 'PPO')]

    for ax, (exp, algo) in zip(axes, experiments):
        plot_single_experiment(exp, algo, ax, COLORS[exp])

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'fig1_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def figure2_baseline_vs_custom():
    """Figure 2: Comparison of baseline vs custom reward (SAC)."""
    print("\n" + "="*60)
    print("FIGURE 2: Baseline vs Custom Reward Comparison")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp, algo, color in [('baseline', 'SAC', COLORS['baseline']),
                              ('custom', 'SAC', COLORS['custom'])]:
        print(f"\nLoading {exp}...")
        all_data = load_experiment_data(exp, algo)

        if all_data:
            common_steps, mean_values, std_values = interpolate_to_common_steps(all_data)

            if common_steps is not None:
                ax.plot(common_steps, mean_values, color=color, linewidth=2,
                       label=f'{exp.capitalize()} (mean)')
                ax.fill_between(common_steps, mean_values - std_values,
                               mean_values + std_values, color=color, alpha=0.2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Baseline vs Custom Reward (SAC)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'fig2_baseline_vs_custom.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def figure3_sac_vs_ppo():
    """Figure 3: SAC vs PPO comparison (both with custom reward)."""
    print("\n" + "="*60)
    print("FIGURE 3: SAC vs PPO Comparison (Custom Reward)")
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp, algo, color, label in [('custom', 'SAC', COLORS['custom'], 'SAC (Custom)'),
                                     ('extension', 'PPO', COLORS['extension'], 'PPO (Custom)')]:
        print(f"\nLoading {exp} ({algo})...")
        all_data = load_experiment_data(exp, algo)

        if all_data:
            common_steps, mean_values, std_values = interpolate_to_common_steps(all_data)

            if common_steps is not None:
                ax.plot(common_steps, mean_values, color=color, linewidth=2,
                       label=f'{label} (mean)')
                ax.fill_between(common_steps, mean_values - std_values,
                               mean_values + std_values, color=color, alpha=0.2)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Algorithm Comparison: SAC vs PPO (Custom Reward)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'fig3_sac_vs_ppo.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def figure4_evaluation_metrics():
    """Figure 4: Bar chart comparing evaluation metrics."""
    print("\n" + "="*60)
    print("FIGURE 4: Evaluation Metrics Comparison")
    print("="*60)

    # Load metrics from JSON files
    metrics = {}
    for exp in ['baseline', 'custom', 'extension']:
        json_path = os.path.join(RESULTS_DIR, f'{exp}_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics[exp] = json.load(f)
                print(f"Loaded {exp}_metrics.json")

    if not metrics:
        print("No metrics files found!")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    experiments = list(metrics.keys())
    x = np.arange(len(experiments))
    width = 0.6

    # Plot 1: Mean Reward
    ax1 = axes[0]
    rewards = [metrics[exp]['mean_reward_across_trials'] for exp in experiments]
    stds = [metrics[exp]['std_reward_across_trials'] for exp in experiments]
    colors = [COLORS[exp] for exp in experiments]
    bars = ax1.bar(x, rewards, width, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax1.set_ylabel('Mean Reward', fontsize=10)
    ax1.set_title('Mean Reward (± Std)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([exp.capitalize() for exp in experiments])
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Success Rate
    ax2 = axes[1]
    success_rates = [metrics[exp]['mean_success_rate'] * 100 for exp in experiments]
    success_stds = [metrics[exp]['std_success_rate'] * 100 for exp in experiments]
    bars = ax2.bar(x, success_rates, width, yerr=success_stds, capsize=5, color=colors, alpha=0.8)
    ax2.set_ylabel('Success Rate (%)', fontsize=10)
    ax2.set_title('Success Rate (± Std)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([exp.capitalize() for exp in experiments])
    ax2.set_ylim(0, 110)

    # Plot 3: Episode Length
    ax3 = axes[2]
    lengths = [metrics[exp]['mean_episode_length_across_trials'] for exp in experiments]
    bars = ax3.bar(x, lengths, width, color=colors, alpha=0.8)
    ax3.set_ylabel('Episode Length', fontsize=10)
    ax3.set_title('Mean Episode Length', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([exp.capitalize() for exp in experiments])
    ax3.axhline(y=999, color='red', linestyle='--', alpha=0.5, label='Timeout (999)')
    ax3.legend(fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'fig4_evaluation_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def figure5_all_comparison():
    """Figure 5: All three experiments overlayed."""
    print("\n" + "="*60)
    print("FIGURE 5: All Experiments Comparison")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 6))

    experiments = [('baseline', 'SAC', 'Baseline (SAC)'),
                   ('custom', 'SAC', 'Custom Reward (SAC)'),
                   ('extension', 'PPO', 'Extension (PPO)')]

    for exp, algo, label in experiments:
        print(f"\nLoading {exp} ({algo})...")
        all_data = load_experiment_data(exp, algo)

        if all_data:
            common_steps, mean_values, std_values = interpolate_to_common_steps(all_data)

            if common_steps is not None:
                ax.plot(common_steps, mean_values, color=COLORS[exp],
                       linewidth=2, label=label)
                ax.fill_between(common_steps, mean_values - std_values,
                               mean_values + std_values, color=COLORS[exp], alpha=0.15)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Training Comparison: All Experiments', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, 'fig5_all_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def print_summary_table():
    """Print summary table for the report."""
    print("\n" + "="*60)
    print("SUMMARY TABLE FOR REPORT")
    print("="*60)

    print("\n| Experiment | Algorithm | Mean Reward | Success Rate | Avg Episode Length |")
    print("|------------|-----------|-------------|--------------|-------------------|")

    for exp in ['baseline', 'custom', 'extension']:
        json_path = os.path.join(RESULTS_DIR, f'{exp}_metrics.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                m = json.load(f)
            algo = 'PPO' if exp == 'extension' else 'SAC'
            print(f"| {exp.capitalize():10} | {algo:9} | {m['mean_reward_across_trials']:11.2f} | {m['mean_success_rate']*100:12.1f}% | {m['mean_episode_length_across_trials']:17.1f} |")


if __name__ == '__main__':
    print("="*60)
    print("EXPORTING FIGURES FOR REPORT")
    print("="*60)

    # Generate all figures
    figure1_individual_training_curves()
    figure2_baseline_vs_custom()
    figure3_sac_vs_ppo()
    figure4_evaluation_metrics()
    figure5_all_comparison()

    # Print summary
    print_summary_table()

    print("\n" + "="*60)
    print(f"ALL FIGURES SAVED TO: {FIGURES_DIR}")
    print("="*60)
    print("\nFigures generated:")
    print("  - fig1_training_curves.png (individual experiments)")
    print("  - fig2_baseline_vs_custom.png (reward comparison)")
    print("  - fig3_sac_vs_ppo.png (algorithm comparison)")
    print("  - fig4_evaluation_metrics.png (bar charts)")
    print("  - fig5_all_comparison.png (all overlayed)")
