"""Games between teams, and how often the first checkpoint given wins them. Each team is played by a checkpoint, a
training (version_N: its latest checkpoint) or a scripted controller of beyond_all_reason_environment.

Made for the self-play trainings, whose own victory rate (~50 %) says nothing of the level reached.

Examples (from this repository root):
    python scripts/play_games.py \
        --experimentation_configuration_file configurations/combined_arms_commander_rear_spawn_self_play_ppo.py \
        --teams experiments/.../version_1 AttackNearest --episodes 100
Two checkpoints, half of the games on each side so that an advantage of a side does not count for a checkpoint:
    python scripts/play_games.py \
        --experimentation_configuration_file configurations/combined_arms_commander_rear_spawn_self_play_ppo.py \
        --teams experiments/.../checkpoint_000253 experiments/.../checkpoint_000110 --swap_sides --episodes 200

The observation and actions must be those the checkpoints were trained on (the shapes are checked). --scenario puts
another scenario of the catalogue (beyond_all_reason_environment.scenarios) in the configuration, with as many teams.
To watch the games in the game window: --game_speed 4 (a multiple of real time); --environment_configuration adds
entries for the window, as JSON (see RayMultiAgentEnvironment): '{"interface": "cinematic", "video": true}' records into
videos/<experimentation name>_<n>.mp4.
"""

import argparse
import dataclasses
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import beyond_all_reason_environment.scenarios as scenarios
from beyond_all_reason_environment.reinforcement_learning_environment import ReinforcementLearningEnvironment
from maxalaar_reinforcement_learning.utilities.save.get_configuration_class import get_configuration_class

from beyond_all_reason_reinforcement_learning.checkpoint_controller import (
    Checkpoint,
    CheckpointPlayers,
    team_controller,
    team_outcome,
)
from beyond_all_reason_reinforcement_learning.ray_multi_agent_environment import window_configuration


def play_games(configuration, episodes: int, team: int) -> Counter:
    """The outcomes of ``episodes`` games for ``team``, and their number of decisions, in one engine."""
    # The module moves to the GPU when it sees one, which a training may need: inference stays on the CPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    environment = ReinforcementLearningEnvironment(configuration)
    players = CheckpointPlayers.of(environment)
    counts = Counter()
    try:
        for episode in range(1, episodes + 1):
            observations, informations = environment.reset()
            while environment.agents:
                observations, _, _, _, informations = environment.step(players.actions(observations))
                counts["decisions"] += 1
            outcome = team_outcome(informations, team)
            counts[outcome] += 1
            print(f"process {os.getpid()}, episode {episode}/{episodes}: {outcome}", flush=True)
    finally:
        environment.close()
    return counts


def play_side(configuration, episodes: int, team: int, processes: int) -> Counter:
    """``episodes`` games shared between ``processes`` engines, in spawned processes (torch does not survive a fork)."""
    shares = [episodes // processes + (process < episodes % processes) for process in range(processes)]
    with ProcessPoolExecutor(processes, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = [pool.submit(play_games, configuration, share, team) for share in shares if share > 0]
        return sum((result.result() for result in results), Counter())


def summary(counts: Counter) -> str:
    episodes = counts["victory"] + counts["defeat"] + counts["draw"]
    return (
        f"{episodes} episodes, victories {counts['victory'] / episodes:.3f}, defeats {counts['defeat'] / episodes:.3f},"
        f" draws {counts['draw'] / episodes:.3f}, episode length {counts['decisions'] / episodes:.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experimentation_configuration_file", required=True, help="Its environment configuration")
    parser.add_argument("--teams", nargs="+", required=True, help="Who plays each team, in the scenario's order")
    parser.add_argument("--swap_sides", action="store_true", help="Half of the games with the teams in reverse order")
    parser.add_argument("--scenario", help="A scenario of beyond_all_reason_environment.scenarios, instead of the one")
    parser.add_argument("--episodes", type=int, default=100, help="In all, whatever --swap_sides")
    parser.add_argument("--environment_runners", type=int, default=2, help="Engines playing at the same time")
    parser.add_argument("--deterministic", action="store_true", help="The most likely action instead of a draw")
    parser.add_argument("--game_speed", type=float, help="Shows the games in the game window, at this speed")
    parser.add_argument(
        "--environment_configuration",
        type=json.loads,
        default={},
        help="""Entries added for the game window, as JSON (e.g., '{"interface": "cinematic"}')""",
    )
    arguments = parser.parse_args()

    experimentation_configuration = get_configuration_class(arguments.experimentation_configuration_file)
    configuration = experimentation_configuration.environment_configuration["configuration"]
    scenario = configuration.game.scenario if arguments.scenario is None else getattr(scenarios, arguments.scenario)
    teams = [team_controller(team) for team in arguments.teams]
    teams = [
        dataclasses.replace(team, deterministic=True)
        if arguments.deterministic and isinstance(team, Checkpoint)
        else team
        for team in teams
    ]
    counted = next((index for index, team in enumerate(teams) if isinstance(team, Checkpoint)), None)
    if counted is None:
        raise ValueError("no checkpoint among --teams: scripted teams alone play with run.py of the environment")

    # The window on one engine.
    window = {}
    processes = arguments.environment_runners
    if arguments.game_speed is not None:
        window = {
            "render_mode": "human",
            "game_speed": arguments.game_speed,
            "experimentation_name": experimentation_configuration.experimentation_name,
            **arguments.environment_configuration,
        }
        processes = 1

    sides = [(teams, counted)]
    if arguments.swap_sides:
        sides.append((teams[::-1], len(teams) - 1 - counted))
    results = []
    for controllers, team in sides:
        game = dataclasses.replace(configuration.game, scenario=scenario, controllers=controllers)
        side_configuration = window_configuration(dataclasses.replace(configuration, game=game), window)
        counts = play_side(side_configuration, arguments.episodes // len(sides), team, processes)
        results.append((f"on team {team}", counts))

    print(f"{teams[counted].path}\nagainst {[team.label for index, team in enumerate(teams) if index != counted]}:")
    for name, counts in results:
        print(f"  {name}: {summary(counts)}")
    if len(results) > 1:
        print(f"  in all: {summary(sum((counts for _, counts in results), Counter()))}")


if __name__ == "__main__":
    main()
