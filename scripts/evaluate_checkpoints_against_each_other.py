"""Two checkpoints of the same architecture against each other: the first one's weights play one team, the second
one's the other, and the first one's victory, defeat and draw rates are counted.

Half of the episodes with the first checkpoint on the first team (the west), half on the second (the east), so that
an advantage of a side does not count for a checkpoint. Made to compare two stages of a self-play training, whose own
victory rate (~50 %) says nothing of the progress. The scenario needs two teams played by agents.
--game_speed shows the games in the game window (one window per side, one after the other), and
--environment_configuration adds entries to its env_config, for example '{"interface": "reduced", "video": true}'
(videos/first_against_second_<n>.mp4).

Example (from this repository root):
    python scripts/evaluate_checkpoints_against_each_other.py \
        --experimentation_configuration_file configurations/combined_arms_commander_rear_spawn_self_play_ppo.py \
        --first_checkpoint_path experiments/.../checkpoint_000253 \
        --second_checkpoint_path experiments/combined_arms_commander_self_play_ppo/saved_checkpoints/<checkpoint> \
        --episodes 200
"""

import argparse
import dataclasses
import json
import warnings
from pathlib import Path

import beyond_all_reason_environment.scenarios as scenarios
import ray
from beyond_all_reason_environment import ReinforcementLearningEnvironmentConfiguration
from maxalaar_reinforcement_learning.checkpoints.get_checkpoint_algorithm_configuration import (
    get_checkpoint_algorithm_configuration,
)
from maxalaar_reinforcement_learning.environments.register_environments import register_environment
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec

POLICIES = ("first", "second")
# Where a checkpoint keeps the weights of the shared policy of the self-play trainings.
SHARED_POLICY = Path("learner_group") / "learner" / "rl_module" / "shared_policy"


class FirstCheckpointOutcomeCallback(RLlibCallback):
    """Victories, defeats and episodes of the first checkpoint, summed over the evaluation.

    The latest information of the episode tells who lost, seen from its agent's team (allies_defeated,
    enemies_defeated); the team of the first checkpoint is env_config["first_team"].
    """

    def on_episode_end(self, *, episode, env_runner=None, metrics_logger=None, **kwargs) -> None:
        latest = None
        for agent_id, agent_episode in episode.agent_episodes.items():
            if len(agent_episode.infos) > 0:
                information = agent_episode.get_infos(-1)
                if latest is None or information.get("episode_frame", 0) > latest[1].get("episode_frame", 0):
                    latest = (agent_id, information)
        if latest is None or metrics_logger is None or env_runner is None:
            return
        agent_id, information = latest
        allies_defeated, enemies_defeated = bool(information["allies_defeated"]), bool(information["enemies_defeated"])
        if not agent_id.startswith(f"team{env_runner.config.env_config['first_team']}_"):
            allies_defeated, enemies_defeated = enemies_defeated, allies_defeated
        metrics_logger.log_value("first_victories", float(enemies_defeated and not allies_defeated), reduce="sum")
        metrics_logger.log_value("first_defeats", float(allies_defeated and not enemies_defeated), reduce="sum")
        metrics_logger.log_value("counted_episodes", 1.0, reduce="sum")


def parameter_sum(module) -> float:
    return round(float(sum(parameter.detach().double().sum() for parameter in module.parameters())), 4)


def evaluate(arguments: argparse.Namespace, first_team: int, episodes: int) -> dict[str, float]:
    """The outcomes of ``episodes`` episodes with the first checkpoint on ``first_team``."""
    algorithm_configuration = get_checkpoint_algorithm_configuration(arguments.first_checkpoint_path.resolve())
    register_environment(algorithm_configuration.env)
    environment_configuration = get_configuration_class(
        arguments.experimentation_configuration_file
    ).environment_configuration
    configuration: ReinforcementLearningEnvironmentConfiguration = environment_configuration["configuration"]
    if arguments.scenario is not None:
        game = dataclasses.replace(configuration.game, scenario=getattr(scenarios, arguments.scenario))
        configuration = dataclasses.replace(configuration, game=game)
    environment_configuration = {**environment_configuration, "configuration": configuration, "first_team": first_team}

    # The window only for the evaluation env runner: the local one also builds an environment, never stepped.
    evaluation_environment_configuration = environment_configuration
    environment_runners = arguments.environment_runners
    if arguments.game_speed is not None:
        evaluation_environment_configuration = {
            **environment_configuration,
            "render_mode": "human",
            "game_speed": arguments.game_speed,
            "experimentation_name": "first_against_second",
            **arguments.environment_configuration,
        }
        environment_runners = 1

    module_spec = algorithm_configuration.rl_module_spec
    algorithm_configuration.environment(env_config=environment_configuration)
    algorithm_configuration.multi_agent(
        policies=set(POLICIES),
        policy_mapping_fn=lambda agent_id, episode, **kwargs: (
            "first" if agent_id.startswith(f"team{first_team}_") else "second"
        ),
    )
    if hasattr(module_spec, "rl_module_specs"):
        module_spec = next(iter(module_spec.rl_module_specs.values()))
    algorithm_configuration.rl_module(
        rl_module_spec=MultiRLModuleSpec(
            rl_module_specs={policy: dataclasses.replace(module_spec, load_state_path=None) for policy in POLICIES}
        )
    )
    algorithm_configuration.callbacks(FirstCheckpointOutcomeCallback)
    algorithm_configuration.learners(num_learners=0, num_gpus_per_learner=0)
    algorithm_configuration.env_runners(num_env_runners=0, num_envs_per_env_runner=1)
    algorithm_configuration.evaluation(
        evaluation_interval=1,
        evaluation_num_env_runners=environment_runners,
        evaluation_duration=episodes,
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
        evaluation_sample_timeout_s=60 * 60,
        evaluation_config=AlgorithmConfig.overrides(explore=True, env_config=evaluation_environment_configuration),
    )

    algorithm = algorithm_configuration.build()
    try:
        # The weights of each checkpoint into its policy, on the learner, then to the evaluation env runners.
        for policy, checkpoint_path in zip(
            POLICIES, (arguments.first_checkpoint_path, arguments.second_checkpoint_path), strict=True
        ):
            # Into the learner's module, the source of sync_weights (algorithm.get_module is the local env runner's).
            path = str((checkpoint_path / SHARED_POLICY).resolve())
            algorithm.learner_group.foreach_learner(
                lambda learner, policy=policy, path=path: learner.module[policy].restore_from_path(path)
            )
        algorithm.eval_env_runner_group.sync_weights(
            from_worker_or_learner_group=algorithm.learner_group, inference_only=True
        )
        # Check: the weights the evaluation env runners play with are the checkpoints' (sums of their parameters).
        for policy, checkpoint_path in zip(
            POLICIES, (arguments.first_checkpoint_path, arguments.second_checkpoint_path), strict=True
        ):
            played = algorithm.eval_env_runner_group.foreach_env_runner(
                lambda env_runner, policy=policy: parameter_sum(env_runner.module[policy]), local_env_runner=False
            )
            print(f"weights of {policy} ({checkpoint_path.name}): played {played}", flush=True)
        results = algorithm.evaluate()["env_runners"]
    finally:
        algorithm.stop()
    return {key: results[key] for key in ("first_victories", "first_defeats", "counted_episodes")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experimentation_configuration_file", required=True, help="Its environment configuration")
    parser.add_argument("--first_checkpoint_path", type=Path, required=True)
    parser.add_argument("--second_checkpoint_path", type=Path, required=True)
    parser.add_argument("--scenario", help="A scenario of beyond_all_reason_environment.scenarios, instead of the one")
    parser.add_argument("--episodes", type=int, default=200, help="In all, half with the first checkpoint each side")
    parser.add_argument("--environment_runners", type=int, default=2)
    parser.add_argument(
        "--game_speed", type=float, help="Shows the games in the game window (one per side), at this speed"
    )
    parser.add_argument(
        "--environment_configuration",
        type=json.loads,
        default={},
        help="""Entries added to the env_config of the game window, as JSON (e.g., '{"interface": "reduced"}')""",
    )
    arguments = parser.parse_args()
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    ray.init()
    try:
        sides = [evaluate(arguments, first_team, arguments.episodes // 2) for first_team in (0, 1)]
    finally:
        ray.shutdown()

    print(f"first: {arguments.first_checkpoint_path}\nsecond: {arguments.second_checkpoint_path}")
    for name, outcomes in (("first on the west", sides[0]), ("first on the east", sides[1])):
        episodes = outcomes["counted_episodes"]
        print(
            f"  {name}: {episodes:.0f} episodes, victories {outcomes['first_victories'] / episodes:.3f},"
            f" defeats {outcomes['first_defeats'] / episodes:.3f}"
        )
    episodes = sum(side["counted_episodes"] for side in sides)
    victories = sum(side["first_victories"] for side in sides) / episodes
    defeats = sum(side["first_defeats"] for side in sides) / episodes
    print(
        f"  first in all: {episodes:.0f} episodes, victories {victories:.3f}, defeats {defeats:.3f},"
        f" draws {1 - victories - defeats:.3f}"
    )


if __name__ == "__main__":
    main()
