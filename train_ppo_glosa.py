"""Train PPO model for fleet-level GLOSA control."""

from __future__ import annotations

import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from rl_environments.glosa_env import GLOSAEnv

DEFAULT_TIMESTEPS = 40_000
CHECKPOINT_INTERVAL = 10_000
MODEL_DIR = os.path.join("outputs", "models")
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_glosa_final")
TENSORBOARD_DIR = os.path.join("outputs", "tensorboard")
ROLLOUT_STEPS = 1_800


class CheckpointCallback(BaseCallback):
    """Periodic checkpoint callback."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose=verbose)

    def _on_step(self) -> bool:
        if self.num_timesteps > 0 and self.num_timesteps % CHECKPOINT_INTERVAL == 0:
            path = os.path.join(MODEL_DIR, f"ppo_glosa_checkpoint_{self.num_timesteps}")
            self.model.save(path)
            print(f"Checkpoint saved: {path}")
        return True


def parse_args() -> argparse.Namespace:
    """Parse CLI options."""
    parser = argparse.ArgumentParser(description="Train PPO-GLOSA model.")
    parser.add_argument("--resume", action="store_true", help="Resume previous PPO model.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Total training timesteps.",
    )
    return parser.parse_args()


def make_env() -> GLOSAEnv:
    """Create one GLOSA environment instance."""
    return GLOSAEnv(config.SUMO_CFG)


def build_model(env: DummyVecEnv, resume: bool) -> PPO:
    """Create or load PPO model."""
    if resume and os.path.exists(f"{FINAL_MODEL_PATH}.zip"):
        print(f"Resuming PPO model from: {FINAL_MODEL_PATH}")
        return PPO.load(FINAL_MODEL_PATH, env=env)

    return PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=ROLLOUT_STEPS,
        batch_size=256,
        n_epochs=5,
        gamma=0.98,
        ent_coef=0.005,
        learning_rate=3e-4,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[128, 128]),
        tensorboard_log=TENSORBOARD_DIR,
        verbose=1,
    )


def main() -> None:
    """Entry point for PPO-GLOSA training."""
    args = parse_args()
    os.makedirs("outputs", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_DIR, exist_ok=True)

    start_time = time.time()
    env = DummyVecEnv([lambda: make_env()])
    model = build_model(env=env, resume=args.resume)
    callback = CheckpointCallback()

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        model.save(FINAL_MODEL_PATH)
        print(f"Final PPO-GLOSA model saved: {FINAL_MODEL_PATH}")
    finally:
        env.close()

    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
