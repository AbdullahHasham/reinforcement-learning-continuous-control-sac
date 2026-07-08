import gymnasium as gym
from stable_baselines3 import PPO, DQN, A2C, SAC
import numpy as np
from utils import get_algorithm
import os
import json

# Get the base directory (Student folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def evaluate(model_path, env_id, algorithm="SAC", episodes=10, render=False):
    """
    Evaluate a trained model without exploration (deterministic=True).

    Args:
        model_path: Path to the saved model
        env_id: Gymnasium environment ID
        algorithm: Algorithm used (PPO, DQN, A2C, SAC)
        episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Dictionary with evaluation metrics
    """
    # Get algorithm class and load model
    algo = get_algorithm(algorithm)
    model = algo.load(model_path)

    # Create environment
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)

    # Metrics storage
    episode_rewards = []
    episode_lengths = []
    successes = []  # For MountainCar: reached goal position

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        max_position = -np.inf

        while not done and not truncated:
            # IMPORTANT: deterministic=True to disable exploration
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Track max position for MountainCar
            if len(obs) >= 1:
                max_position = max(max_position, obs[0])

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

        # Success = reached goal (position >= 0.45 for MountainCarContinuous)
        success = max_position >= 0.45
        successes.append(success)

        print(f"Episode {ep + 1}/{episodes}: Reward = {total_reward:.2f}, "
              f"Steps = {steps}, Max Position = {max_position:.3f}, Success = {success}")

    env.close()

    # Calculate statistics
    metrics = {
        "model_path": model_path,
        "environment": env_id,
        "episodes": episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "std_episode_length": float(np.std(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "all_rewards": episode_rewards,
        "all_lengths": episode_lengths
    }

    print(f"\n{'='*50}")
    print("EVALUATION SUMMARY")
    print(f"{'='*50}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
    print(f"Mean Episode Length: {metrics['mean_episode_length']:.1f} +/- {metrics['std_episode_length']:.1f}")
    print(f"Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"{'='*50}\n")

    return metrics


def evaluate_all_trials(results_dir, env_id, algorithm, num_trials=3, episodes=10):
    """
    Evaluate all trials of an experiment and aggregate results.

    Args:
        results_dir: Directory containing saved models (e.g., "results/baseline")
        env_id: Gymnasium environment ID
        algorithm: Algorithm name
        num_trials: Number of trials to evaluate
        episodes: Episodes per trial

    Returns:
        Aggregated metrics across all trials
    """
    all_metrics = []

    for trial in range(1, num_trials + 1):
        model_path = f"{results_dir}/{algorithm}_trial{trial}"
        print(f"\n{'#'*60}")
        print(f"Evaluating Trial {trial}")
        print(f"{'#'*60}")

        if os.path.exists(model_path + ".zip"):
            metrics = evaluate(model_path, env_id, algorithm, episodes)
            all_metrics.append(metrics)
        else:
            print(f"Warning: Model not found at {model_path}")

    if not all_metrics:
        print("No models found to evaluate!")
        return None

    # Aggregate across trials
    all_mean_rewards = [m['mean_reward'] for m in all_metrics]
    all_success_rates = [m['success_rate'] for m in all_metrics]
    all_mean_lengths = [m['mean_episode_length'] for m in all_metrics]

    aggregated = {
        "experiment": results_dir,
        "num_trials": len(all_metrics),
        "episodes_per_trial": episodes,
        "mean_reward_across_trials": float(np.mean(all_mean_rewards)),
        "std_reward_across_trials": float(np.std(all_mean_rewards)),
        "mean_success_rate": float(np.mean(all_success_rates)),
        "std_success_rate": float(np.std(all_success_rates)),
        "mean_episode_length_across_trials": float(np.mean(all_mean_lengths)),
        "trial_metrics": all_metrics
    }

    print(f"\n{'='*60}")
    print("AGGREGATED RESULTS ACROSS ALL TRIALS")
    print(f"{'='*60}")
    print(f"Mean Reward: {aggregated['mean_reward_across_trials']:.2f} +/- {aggregated['std_reward_across_trials']:.2f}")
    print(f"Success Rate: {aggregated['mean_success_rate']*100:.1f}% +/- {aggregated['std_success_rate']*100:.1f}%")
    print(f"Mean Episode Length: {aggregated['mean_episode_length_across_trials']:.1f}")
    print(f"{'='*60}\n")

    return aggregated


def save_metrics(metrics, filepath):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {filepath}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained RL models")
    parser.add_argument("--model", type=str, help="Path to single model")
    parser.add_argument("--results_dir", type=str, help="Directory with trial models")
    parser.add_argument("--env", type=str, default="MountainCarContinuous-v0")
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--save", type=str, help="Path to save metrics JSON")

    args = parser.parse_args()

    if args.model:
        # Evaluate single model
        metrics = evaluate(args.model, args.env, args.algo, args.episodes, args.render)
        if args.save:
            save_metrics(metrics, args.save)
    elif args.results_dir:
        # Evaluate all trials
        metrics = evaluate_all_trials(args.results_dir, args.env, args.algo, args.trials, args.episodes)
        if args.save and metrics:
            save_metrics(metrics, args.save)
    else:
        # Default: evaluate baseline and custom
        print("Evaluating Baseline Models...")
        baseline_dir = os.path.join(BASE_DIR, "results", "baseline")
        baseline_metrics = evaluate_all_trials(baseline_dir, "MountainCarContinuous-v0", "SAC", 3, 10)
        if baseline_metrics:
            save_metrics(baseline_metrics, os.path.join(BASE_DIR, "results", "baseline_metrics.json"))

        print("\nEvaluating Custom Reward Models...")
        custom_dir = os.path.join(BASE_DIR, "results", "custom")
        custom_metrics = evaluate_all_trials(custom_dir, "MountainCarContinuous-v0", "SAC", 3, 10)
        if custom_metrics:
            save_metrics(custom_metrics, os.path.join(BASE_DIR, "results", "custom_metrics.json"))

        print("\nEvaluating Extension Models (PPO)...")
        extension_dir = os.path.join(BASE_DIR, "results", "extension")
        extension_metrics = evaluate_all_trials(extension_dir, "MountainCarContinuous-v0", "PPO", 3, 10)
        if extension_metrics:
            save_metrics(extension_metrics, os.path.join(BASE_DIR, "results", "extension_metrics.json"))