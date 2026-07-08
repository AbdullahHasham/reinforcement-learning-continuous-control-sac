from custom_env_wrapper import CustomRewardWrapper
from stable_baselines3 import PPO, DQN, A2C, SAC
from utils import load_config, get_algorithm
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_extension(cfg_path=None):
    """
    Train model with a different algorithm (Extension Option A).

    Extension: Compare PPO vs SAC on the same custom reward environment.
    The primary experiments utilized SAC, so we use PPO here for comparison.
    Runs multiple trials for robustness.
    """
    if cfg_path is None:
        cfg_path = os.path.join(BASE_DIR, "config", "config_extension.yaml")
    cfg = load_config(cfg_path)

    # Create directories if they don't exist
    results_dir = os.path.join(BASE_DIR, "results", "extension")
    os.makedirs(results_dir, exist_ok=True)

    # Get algorithm class
    algo = get_algorithm(cfg['algorithm'])

    for trial in range(1, cfg['num_trials'] + 1):
        print(f"\n{'='*50}")
        print(f"Starting Extension Trial {trial}/{cfg['num_trials']}")
        print(f"Algorithm: {cfg['algorithm']}")
        print(f"{'='*50}\n")

        # Create base environment
        env = gym.make(cfg['environment'])

        # Wrap with custom reward (same as custom reward training configuration)
        env = CustomRewardWrapper(env, cfg.get('reward_params', {}))

        # Create log directory for this trial
        log_dir = os.path.join(BASE_DIR, "logs", "extension", f"{cfg['algorithm']}_trial{trial}")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # Initialize model with algorithm-specific settings
        if cfg['algorithm'] == 'PPO':
            model = algo(
                "MlpPolicy",
                env,
                verbose=1,
                tensorboard_log=log_dir,
                seed=trial,
                n_steps=cfg.get('n_steps', 2048),
                batch_size=cfg.get('batch_size', 64),
                learning_rate=cfg.get('learning_rate', 1e-4),
                gamma=cfg.get('gamma', 0.99),
                gae_lambda=cfg.get('gae_lambda', 0.95),
                clip_range=cfg.get('clip_range', 0.2),
                ent_coef=cfg.get('ent_coef', 0.01),
                n_epochs=cfg.get('n_epochs', 10)
            )
        else:
            # Generic initialization for other algorithms
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
            name_prefix=f"{cfg['algorithm']}_extension"
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
    train_extension()
