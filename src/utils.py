import yaml
from stable_baselines3 import PPO, DQN, A2C, SAC


def load_config(p):
    """Load YAML configuration file."""
    with open(p) as f:
        return yaml.safe_load(f)


def get_algorithm(name):
    """
    Get Stable Baselines 3 algorithm class by name.

    Args:
        name: Algorithm name (PPO, DQN, A2C, SAC)

    Returns:
        Algorithm class
    """
    algorithms = {
        "PPO": PPO,
        "DQN": DQN,
        "A2C": A2C,
        "SAC": SAC
    }
    if name not in algorithms:
        raise ValueError(f"Unknown algorithm: {name}. Choose from {list(algorithms.keys())}")
    return algorithms[name]
