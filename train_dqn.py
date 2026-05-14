"""Train DQN policy for traffic signal control."""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from rl_environments.signal_env import TrafficSignalEnv

DEFAULT_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL_STEPS = 10_000
LOG_INTERVAL_STEPS = 1_000
ROLLING_EPISODES = 10
MAX_STEPS_PER_EPISODE = 1_800
MODEL_DIR = os.path.join("outputs", "models")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "dqn_signal_final")
TENSORBOARD_DIR = os.path.join("outputs", "tensorboard")


class TrainingCallback(BaseCallback):
    """Custom callback for periodic logging and checkpointing."""

    def __init__(self, verbose: int = 0) -> None:
        """Initialize callback state."""
        super().__init__(verbose=verbose)
        self._episode_rewards_window: deque[float] = deque(maxlen=ROLLING_EPISODES)
        self._current_episode_reward = 0.0

    def _on_step(self) -> bool:
        """Process rollout step data and perform periodic logging."""
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None and len(rewards) > 0:
            self._current_episode_reward += float(rewards[0])

        if dones is not None and len(dones) > 0 and bool(dones[0]):
            self._episode_rewards_window.append(self._current_episode_reward)
            self._current_episode_reward = 0.0

        if self.num_timesteps % LOG_INTERVAL_STEPS == 0:
            mean_reward = (
                sum(self._episode_rewards_window) / len(self._episode_rewards_window)
                if self._episode_rewards_window
                else 0.0
            )
            exploration = float(getattr(self.model, "exploration_rate", 0.0))
            print(
                f"Step {self.num_timesteps}: "
                f"mean_reward={mean_reward:.3f}, exploration={exploration:.3f}"
            )

        if self.num_timesteps % CHECKPOINT_INTERVAL_STEPS == 0:
            checkpoint_path = os.path.join(
                MODEL_DIR, f"dqn_checkpoint_{self.num_timesteps}"
            )
            self.model.save(checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        return True


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN for traffic signal control.")
    parser.add_argument("--resume", action="store_true", help="Resume from final model.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps.",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Training uses headless SUMO by default; this flag is accepted for clarity.",
    )
    return parser.parse_args()


def make_env() -> TrafficSignalEnv:
    """Create a fresh traffic signal environment instance."""
    return TrafficSignalEnv(sumo_cfg=config.SUMO_CFG, max_steps=MAX_STEPS_PER_EPISODE)


def build_model(vec_env: DummyVecEnv, resume: bool) -> DQN:
    """Build or restore DQN model."""
    if resume and os.path.exists(f"{FINAL_MODEL_PATH}.zip"):
        print(f"Resuming model from: {FINAL_MODEL_PATH}")
        return DQN.load(FINAL_MODEL_PATH, env=vec_env)

    return DQN(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        buffer_size=20000,
        learning_starts=1000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
    )


def main() -> None:
    """Train DQN and save final artifact."""
    args = parse_args()
    if args.no_gui:
        print("Training runs in no-GUI mode by default.")

    os.makedirs("outputs", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    start_time = time.time()
    vec_env = DummyVecEnv([make_env])
    model = build_model(vec_env=vec_env, resume=args.resume)
    callback = TrainingCallback()

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(FINAL_MODEL_PATH)
        print(f"Final model saved: {FINAL_MODEL_PATH}")
    finally:
        vec_env.close()

    elapsed_seconds = time.time() - start_time
    print(f"Total training time: {elapsed_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
