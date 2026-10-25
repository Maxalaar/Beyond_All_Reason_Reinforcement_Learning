"""Evaluate a checkpoint's policy against a scripted controller: the agents keep the first team, the scripted
controller takes every other team, and the victory, defeat and draw rates are counted over a number of episodes.

Made for the self-play trainings, whose victory rate (~50 %) says nothing of the level reached.
The outcome of each episode is VictoryCallback's; every episode of the evaluation counts (VictoryCallback's own rates
cover the last 100 of an env runner only). Each env runner prints its running count after every episode.

Example (from this repository root):
    python scripts/evaluate_against_controller.py \
        --reinforcement_learning_path experiments/combined_arms_commander_self_play_ppo/reinforcement_learning/base/version_0 \
        --controller AttackNearest --episodes 100
To watch the games in the game window, on one env runner: --game_speed 4 (a multiple of real time).
The environment configuration is the checkpoint's, as it was pickled when the training started; with
--experimentation_configuration_file, the file's as the code makes it now (team colours added since, for example). The
observation and actions must not have changed.
"""

import argparse
import dataclasses
import warnings
from pathlib import Path

import beyond_all_reason_environment.controllers as controllers
import ray
from beyond_all_reason_environment import Agent, ReinforcementLearningEnvironmentConfiguration
from maxalaar_reinforcement_learning.checkpoints.get_checkpoint_algorithm_configuration import (
    get_checkpoint_algorithm_configuration,
)
from maxalaar_reinforcement_learning.environments.register_environments import register_environment
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class
from ray.rllib.algorithms import AlgorithmConfig

from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback


class OutcomeCountCallback(VictoryCallback):
    """VictoryCallback, plus the number of victories, defeats and episodes of the evaluation (summed over its env
    runners), and the running count of this env runner printed after each episode."""

    def __init__(self):
        super().__init__()
        self._victories = self._defeats = self._episodes = 0

    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs) -> None:
        outcomes = len(self._outcomes)
        super().on_episode_end(episode=episode, metrics_logger=metrics_logger, **kwargs)
        if metrics_logger is None or (len(self._outcomes) == outcomes and len(self._outcomes) < self._outcomes.maxlen):
            return
        victory, defeat = self._outcomes[-1]
        self._victories += victory
        self._defeats += defeat
        self._episodes += 1
        metrics_logger.log_value("victories", victory, reduce="sum")
        metrics_logger.log_value("defeats", defeat, reduce="sum")
        metrics_logger.log_value("counted_episodes", 1, reduce="sum")
        outcome = "victory" if victory else "defeat" if defeat else "draw"
        print(
            f"episode {self._episodes}: {outcome}; {self._victories:.0f} victories, {self._defeats:.0f} defeats"
            f" ({self._victories / self._episodes:.0%} victories)",
            flush=True,
        )


def latest_checkpoint_path(reinforcement_learning_path: Path) -> Path:
    checkpoints = sorted(reinforcement_learning_path.glob("*/checkpoint_*"), key=lambda path: path.name)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoint under {reinforcement_learning_path}")
    return checkpoints[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--reinforcement_learning_path", type=Path, help="A training (version_N): its latest checkpoint"
    )
    parser.add_argument("--checkpoint_path", type=Path, help="A checkpoint, instead of the latest of the training")
    parser.add_argument(
        "--experimentation_configuration_file", help="The environment configuration of this file, not the checkpoint's"
    )
    parser.add_argument("--controller", default="AttackNearest", help="A controller of beyond_all_reason_environment")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--environment_runners", type=int, default=2)
    parser.add_argument("--deterministic", action="store_true", help="The most likely action instead of a draw")
    parser.add_argument("--game_speed", type=float, help="Shows the games in the game window, at this speed")
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    checkpoint_path = (
        arguments.checkpoint_path or latest_checkpoint_path(arguments.reinforcement_learning_path)
    ).resolve()
    algorithm_configuration = get_checkpoint_algorithm_configuration(checkpoint_path)
    register_environment(algorithm_configuration.env)

    environment_configuration = algorithm_configuration.env_config
    if arguments.experimentation_configuration_file is not None:
        environment_configuration = get_configuration_class(
            arguments.experimentation_configuration_file
        ).environment_configuration
    configuration: ReinforcementLearningEnvironmentConfiguration = environment_configuration["configuration"]
    opponent = getattr(controllers, arguments.controller)()
    teams = len(configuration.game.controllers)
    game = dataclasses.replace(configuration.game, controllers=[Agent()] + [opponent] * (teams - 1))
    environment_configuration = {
        **environment_configuration,
        "configuration": dataclasses.replace(configuration, game=game),
    }

    algorithm_configuration.environment(env_config=environment_configuration)
    # The window only for the evaluation env runners: the local one also builds an environment, never stepped.
    evaluation_environment_configuration = environment_configuration
    if arguments.game_speed is not None:
        evaluation_environment_configuration = {
            **environment_configuration,
            "render_mode": "human",
            "game_speed": arguments.game_speed,
        }
        arguments.environment_runners = 1
    algorithm_configuration.callbacks(OutcomeCountCallback)
    algorithm_configuration.learners(num_learners=0, num_gpus_per_learner=0)
    algorithm_configuration.env_runners(num_env_runners=0, num_envs_per_env_runner=1)
    algorithm_configuration.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=arguments.environment_runners,
        evaluation_duration=arguments.episodes,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
        evaluation_sample_timeout_s=60 * 60,
        evaluation_config=AlgorithmConfig.overrides(
            explore=not arguments.deterministic, env_config=evaluation_environment_configuration
        ),
    )

    ray.init()
    algorithm = algorithm_configuration.build()
    try:
        algorithm.restore(str(checkpoint_path))
        results = algorithm.evaluate()["env_runners"]
    finally:
        algorithm.stop()
        ray.shutdown()

    episodes = results["counted_episodes"]
    victory, defeat = results["victories"] / episodes, results["defeats"] / episodes
    print(f"checkpoint: {checkpoint_path}")
    print(
        f"against {arguments.controller}, {episodes:.0f} episodes"
        f" ({'deterministic' if arguments.deterministic else 'sampled'} actions):"
    )
    print(
        f"  victories {victory:.3f}, defeats {defeat:.3f}, draws {1 - victory - defeat:.3f},"
        f" episode length {results['episode_len_mean']:.1f}"
    )


if __name__ == "__main__":
    main()
