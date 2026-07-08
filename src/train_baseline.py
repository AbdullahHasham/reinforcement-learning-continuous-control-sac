from stable_baselines3 import PPO, DQN, A2C, SAC
import gymnasium as gym
from utils import load_config, get_algorithm
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Get the base directory (Student folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_baseline(cfg_path=None):
    """
    Train baseline model with default reward function.
    Runs multiple trials for robustness.
    """
    if cfg_path is None:
        cfg_path = os.path.join(BASE_DIR, "config", "config_baseline.yaml")
    cfg = load_config(cfg_path)

    # Create directories if they don't exist
    results_dir = os.path.join(BASE_DIR, "results", "baseline")
    os.makedirs(results_dir, exist_ok=True)

    # Get algorithm class
    algo = get_algorithm(cfg['algorithm'])

    for trial in range(1, cfg['num_trials'] + 1):
        print(f"\n{'='*50}")
        print(f"Starting Baseline Trial {trial}/{cfg['num_trials']}")
        print(f"{'='*50}\n")

        # Create environment
        env = gym.make(cfg['environment'])

        # Create log directory for this trial
        log_dir = os.path.join(BASE_DIR, "logs", "baseline", f"{cfg['algorithm']}_trial{trial}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # Initialize model with hyperparameters from config
        if cfg['algorithm'] == 'SAC':
            model = algo(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                seed=trial,
                learning_rate=cfg.get('learning_rate', 3e-4),
                buffer_size=cfg.get('buffer_size', 50000),
                learning_starts=cfg.get('learning_starts', 0),
                batch_size=cfg.get('batch_size', 256),
                tau=cfg.get('tau', 0.005),
                gamma=cfg.get('gamma', 0.99),
                train_freq=cfg.get('train_freq', 1),
                gradient_steps=cfg.get('gradient_steps', 1)
            )
        else:
            model = algo(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                seed=trial
            )

        # Setup checkpoint callback
        cb = CheckpointCallback(
            save_freq=cfg["checkpoint_freq"],
            save_path=os.path.join(log_dir, "checkpoints"),
            name_prefix=f"{cfg['algorithm']}_baseline"
        )

        # Train the model
        model.learn(total_timesteps=cfg["timesteps"], callback=cb)

        # Save final model
        model_path = os.path.join(results_dir, f"{cfg['algorithm']}_trial{trial}")
        model.save(model_path)

        # Close environment
        env.close()

        print(f"\nTrial {trial} completed. Model saved to {model_path}")


if __name__ == '__main__':
    train_baseline()
