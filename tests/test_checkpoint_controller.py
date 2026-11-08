"""Teams played by checkpoints, without the engine."""

import pytest

pytest.importorskip("ray")

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from beyond_all_reason_environment.controllers import Agent, AttackNearest  # noqa: E402
from beyond_all_reason_environment.game import GameConfiguration  # noqa: E402
from beyond_all_reason_environment.reinforcement_learning_environment import (  # noqa: E402
    ReinforcementLearningEnvironment,
)
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import (  # noqa: E402
    ObservationConfiguration,
    ReinforcementLearningEnvironmentConfiguration,
)
from beyond_all_reason_environment.scenarios import one_versus_one  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import (  # noqa: E402
    BeyondAllReasonModule,
)
from beyond_all_reason_reinforcement_learning.checkpoint_controller import (  # noqa: E402
    CHECKPOINT_FILE,
    Checkpoint,
    CheckpointPlayers,
    load_policy,
    team_controller,
    team_outcome,
    trained_observation,
)

ACTION_SPACE = gym.spaces.Dict({"command": gym.spaces.Discrete(5), "target": gym.spaces.Box(0, 1, (2,), np.float32)})


class FakePolicy:
    """Always prefers the command of the observation's first value; records the batch it receives."""

    action_space = ACTION_SPACE

    def distribution_inputs(self, batch):
        self.batch = batch
        commands = torch.from_numpy(batch["preferred"]).long()
        logits = torch.nn.functional.one_hot(commands, 5).float() * 100
        return torch.cat([logits, torch.zeros(len(commands), 4)], dim=-1).numpy()


def observation(preferred: int) -> dict:
    return {"preferred": np.array(preferred, np.float32)}


def environment_configuration(observation_configuration=None, controllers=None):
    game = GameConfiguration(one_versus_one, controllers or [Agent(), AttackNearest()])
    return ReinforcementLearningEnvironmentConfiguration(
        game=game, observation=observation_configuration or ObservationConfiguration()
    )


def exported_checkpoint(directory: Path, observation_configuration: ObservationConfiguration):
    """A checkpoint folder whose policy, a small BeyondAllReasonModule trained on ``observation_configuration``, is
    exported as a training exports it. Returns the module."""
    from maxalaar_reinforcement_learning.export.policy_export import export_policy

    configuration = environment_configuration(observation_configuration)
    environment = ReinforcementLearningEnvironment(configuration)
    torch.manual_seed(0)
    module = BeyondAllReasonModule(
        observation_space=environment.observation_space("team0_unit_0"),
        action_space=environment.action_space("team0_unit_0"),
        model_config={"d_level1": 16, "critic_d": 16, "terrain_cnn_channels": [4], "action_hidden_sizes": [16]},
    )
    (directory / CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    (directory / CHECKPOINT_FILE).touch()
    export_policy(
        module,
        directory / "policies" / "shared_policy",
        {"environment_configuration": {"configuration": configuration}},
    )
    return module


def test_a_checkpoint_is_an_agent_for_the_environment_with_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exported_checkpoint(tmp_path / "experiments" / "checkpoint_000012", ObservationConfiguration())
    checkpoint = Checkpoint("experiments/checkpoint_000012")
    assert isinstance(checkpoint, Agent)
    assert checkpoint.path == str(tmp_path / "experiments" / "checkpoint_000012")
    assert checkpoint.label == "checkpoint_000012"


def test_a_checkpoint_plays_its_exported_policy_on_the_observation_it_was_trained_on(tmp_path):
    trained_on = ObservationConfiguration(maximum_weapons_per_unit=1, maximum_enemy_units=3)
    module = exported_checkpoint(tmp_path, trained_on)
    checkpoint = Checkpoint(tmp_path, deterministic=True)
    assert checkpoint.observation is not None
    assert (checkpoint.observation.maximum_weapons_per_unit, checkpoint.observation.version) == (1, 1)

    # The environment now sees with 4 weapons by default; the checkpoint's team with the one it was trained on.
    environment = ReinforcementLearningEnvironment(environment_configuration(controllers=[Agent(), checkpoint]))
    space = environment.observation_space("team1_unit_0")
    assert (environment.observation_space("team0_unit_0")["self_weapons"].shape[0], space["self_weapons"].shape[0]) == (
        4,
        1,
    )
    players = CheckpointPlayers.of(environment)
    observations = {f"team1_unit_{index}": space.sample() for index in range(3)}
    for agent_observation in observations.values():
        agent_observation["command_mask"][:] = 1.0
    actions = players.actions(observations)

    batch = {key: torch.from_numpy(np.stack([o[key] for o in observations.values()])) for key in space.spaces}
    with torch.no_grad():
        inputs = module.forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
    expected = module.get_inference_action_dist_cls().from_logits(inputs).to_deterministic().sample()
    assert [action["command"] for action in actions.values()] == expected["command"].tolist()
    assert np.allclose([action["target"] for action in actions.values()], expected["target"].numpy(), atol=1e-4)


def test_a_checkpoint_without_exported_policy_says_how_to_export_it(tmp_path):
    with pytest.raises(ValueError, match="export_policies.py"):
        Checkpoint(tmp_path)


def test_a_checkpoint_of_another_observation_is_refused(tmp_path):
    exported_checkpoint(tmp_path, ObservationConfiguration(maximum_weapons_per_unit=1))
    checkpoint = Checkpoint(tmp_path, observation=ObservationConfiguration(maximum_weapons_per_unit=2))
    environment = ReinforcementLearningEnvironment(environment_configuration(controllers=[Agent(), checkpoint]))
    with pytest.raises(ValueError, match="self_weapons"):
        load_policy(checkpoint, environment.observation_space("team1_unit_0"))


def test_an_observation_saved_before_the_versions_is_version_1():
    saved = {"class": "ObservationConfiguration", "maximum_ally_units": 2, "terrain_resolution": [64, 64]}
    observation = trained_observation({"environment_configuration": {"configuration": {"observation": saved}}})
    assert (observation.version, observation.maximum_ally_units, observation.terrain_resolution) == (1, 2, (64, 64))
    with pytest.raises(ValueError, match="does not know"):
        trained_observation(
            {"environment_configuration": {"configuration": {"observation": {**saved, "can_see_the_future": True}}}}
        )


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


def test_a_team_is_a_scripted_controller_a_checkpoint_or_the_latest_of_a_training(tmp_path):
    assert team_controller("AttackNearest") == AttackNearest()
    checkpoint = tmp_path / "checkpoint_000003"
    exported_checkpoint(checkpoint, ObservationConfiguration())
    assert team_controller(str(checkpoint)) == Checkpoint(checkpoint)
    for trial, name in (
        ("trial_a", "checkpoint_000009"),
        ("trial_a", "checkpoint_000010"),
        ("trial_b", "checkpoint_000002"),
    ):
        exported_checkpoint(tmp_path / "version_0" / trial / name, ObservationConfiguration())
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
