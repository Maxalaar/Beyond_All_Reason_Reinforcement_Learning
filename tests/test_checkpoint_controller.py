"""Teams played by checkpoints, without the engine."""

import pytest

pytest.importorskip("ray")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from beyond_all_reason_environment.controllers import Agent, AttackNearest  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import (  # noqa: E402
    action_distribution_class,
)
from beyond_all_reason_reinforcement_learning.checkpoint_controller import (  # noqa: E402
    CHECKPOINT_FILE,
    Checkpoint,
    CheckpointPlayers,
    load_policy,
    team_controller,
    team_outcome,
)

ACTION_SPACE = gym.spaces.Dict({"command": gym.spaces.Discrete(5), "target": gym.spaces.Box(0, 1, (2,), np.float32)})


class FakePolicy:
    """Always prefers the command of the observation's first value; records the batch it receives."""

    def forward_inference(self, batch):
        self.batch = batch[Columns.OBS]
        commands = self.batch["preferred"].long()
        logits = torch.nn.functional.one_hot(commands, 5).float() * 100
        return {Columns.ACTION_DIST_INPUTS: torch.cat([logits, torch.zeros(len(commands), 4)], dim=-1)}

    def get_inference_action_dist_cls(self):
        return action_distribution_class(ACTION_SPACE)


def observation(preferred: int) -> dict:
    return {"preferred": np.array(preferred, np.float32)}


def test_a_checkpoint_is_an_agent_for_the_environment_with_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    checkpoint = Checkpoint("experiments/checkpoint_000012")
    assert isinstance(checkpoint, Agent)
    assert checkpoint.path == str(tmp_path / "experiments" / "checkpoint_000012")
    assert checkpoint.label == "checkpoint_000012"


def test_players_decide_for_their_teams_only():
    first, second = FakePolicy(), FakePolicy()
    players = CheckpointPlayers({"team0_": (first, True), "team1_": (second, False)})
    actions = players.actions(
        {
            "team0_unit_0": observation(1),
            "team0_unit_2": observation(3),
            "team1_unit_1": observation(4),
            "team2_unit_0": {},
        }
    )
    assert {agent: action["command"] for agent, action in actions.items()} == {
        "team0_unit_0": 1,
        "team0_unit_2": 3,
        "team1_unit_1": 4,
    }
    assert len(first.batch["preferred"]) == 2 and len(second.batch["preferred"]) == 1
    assert all(ACTION_SPACE["target"].contains(action["target"]) for action in actions.values())
    assert players.plays("team1_unit_0") and not players.plays("team2_unit_0")
    assert not CheckpointPlayers({"team1_": (first, False)}).plays("team10_unit_0")


def test_a_team_without_a_living_unit_is_not_asked_to_decide():
    policy = FakePolicy()
    assert CheckpointPlayers({"team1_": (policy, False)}).actions({"team0_unit_0": observation(0)}) == {}
    assert not hasattr(policy, "batch")


def test_a_checkpoint_of_another_observation_is_refused(monkeypatch):
    class Module:
        observation_space = gym.spaces.Dict({"terrain": gym.spaces.Box(0, 1, (4,)), "kept": gym.spaces.Box(0, 1, (1,))})

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        "beyond_all_reason_reinforcement_learning.checkpoint_controller.RLModule.from_checkpoint", lambda path: Module()
    )
    played = gym.spaces.Dict({"terrain": gym.spaces.Box(0, 1, (5,)), "kept": gym.spaces.Box(0, 1, (1,))})
    with pytest.raises(ValueError, match="terrain"):
        load_policy(Checkpoint("checkpoint"), played)
    assert isinstance(load_policy(Checkpoint("checkpoint"), Module.observation_space), Module)


def test_a_team_is_a_scripted_controller_a_checkpoint_or_the_latest_of_a_training(tmp_path):
    assert team_controller("AttackNearest") == AttackNearest()
    checkpoint = tmp_path / "checkpoint_000003"
    checkpoint.mkdir()
    (checkpoint / CHECKPOINT_FILE).touch()
    assert team_controller(str(checkpoint)) == Checkpoint(checkpoint)
    for trial, name in (
        ("trial_a", "checkpoint_000009"),
        ("trial_a", "checkpoint_000010"),
        ("trial_b", "checkpoint_000002"),
    ):
        (tmp_path / "version_0" / trial / name).mkdir(parents=True)
    assert team_controller(str(tmp_path / "version_0")).path.endswith("trial_a/checkpoint_000010")
    with pytest.raises(ValueError, match="needs RLlib"):
        team_controller("Agent")
    with pytest.raises(ValueError, match="neither"):
        team_controller(str(tmp_path / "nothing_here"))


def test_the_outcome_is_seen_from_the_team():
    def informations(allies_defeated: bool, enemies_defeated: bool) -> dict:
        return {"team1_unit_0": {"allies_defeated": allies_defeated, "enemies_defeated": enemies_defeated}}

    assert team_outcome(informations(False, True), 1) == "victory"
    assert team_outcome(informations(True, False), 1) == "defeat"
    assert team_outcome(informations(False, False), 1) == "draw"
    assert team_outcome(informations(True, True), 1) == "draw"
