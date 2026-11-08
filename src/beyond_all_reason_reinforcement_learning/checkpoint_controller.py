"""A team played by the policy of a checkpoint, without RLlib's Algorithm and without the policy's code.

For the environment, Checkpoint is an Agent: the units of its team are agents, decided from Python. CheckpointPlayers
decides for them: RayMultiAgentEnvironment hides them from RLlib, as if they were a scripted team, and
scripts/play_games.py plays whole games with them alone.

A checkpoint plays its exported policy (<checkpoint>/policies/<policy>/policy.onnx, written by
Maxalaar_Reinforcement_Learning's export/policy_export.py, during the training or by its scripts/export_policies.py),
with the observation its policy.json says it was trained on: the policy plays whatever the architecture and the
observation of the environment became since. The actions are drawn as RLlib's GetActions does, on the CPU.
"""

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import beyond_all_reason_environment.controllers as controllers
import numpy as np
import torch
from beyond_all_reason_environment.controllers import Agent, Controller
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import ObservationConfiguration
from ray.rllib.utils.serialization import gym_space_from_dict

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import action_distribution_class

# What marks a folder as a checkpoint, and where it keeps its exported policies (policy_export.py).
CHECKPOINT_FILE = "rllib_checkpoint.json"
POLICIES = "policies"
DESCRIPTION_FILE = "policy.json"


def policy_directory(checkpoint_path: Path) -> Path:
    """The folder of the exported policy of a checkpoint: its only one."""
    directories = sorted(path.parent for path in (checkpoint_path / POLICIES).glob(f"*/{DESCRIPTION_FILE}"))
    if len(directories) != 1:
        raise ValueError(
            f"{checkpoint_path}: {len(directories)} exported policies, one expected; export it with "
            "python ../Maxalaar_Reinforcement_Learning/scripts/export_policies.py <checkpoint>"
        )
    return directories[0]


def trained_observation(description: dict) -> ObservationConfiguration:
    """The observation a policy was trained on, from its policy.json. A configuration without a version was trained
    on version 1: the versions began on 2026-11-08, after them."""
    fields = dict(description["environment_configuration"]["configuration"]["observation"])
    fields.pop("class", None)
    known = {field.name for field in dataclasses.fields(ObservationConfiguration)}
    if set(fields) - known:
        raise ValueError(f"an observation this code does not know: {sorted(set(fields) - known)}")
    fields["terrain_resolution"] = tuple(fields["terrain_resolution"])
    return ObservationConfiguration(**fields)


@dataclass(frozen=True)
class Checkpoint(Agent):
    """The policy of the checkpoint at ``path`` plays the team: the most likely action if ``deterministic``, a draw
    otherwise. Its agents see with the observation it was trained on (its policy.json), unless ``observation`` says
    otherwise."""

    path: str
    deterministic: bool = False

    def __post_init__(self) -> None:
        # Absolute: Ray's workers may not share the working directory.
        object.__setattr__(self, "path", str(Path(self.path).resolve()))
        if self.observation is None:
            description = json.loads((policy_directory(Path(self.path)) / DESCRIPTION_FILE).read_text())
            object.__setattr__(self, "observation", trained_observation(description))

    @property
    def label(self) -> str:
        return Path(self.path).name


class ExportedPolicy:
    """An exported policy (policy.onnx and policy.json): the action distribution inputs of a batch of observations."""

    def __init__(self, directory: Path):
        import onnxruntime

        description = json.loads((directory / DESCRIPTION_FILE).read_text())
        self.inputs = description["inputs"]
        self.output = description["output"]["name"]
        self.observation_space = gym_space_from_dict(description["observation_space"])
        self.action_space = gym_space_from_dict(description["action_space"])
        options = onnxruntime.SessionOptions()
        # One thread: the env runners and the processes of play_games.py already share the cores.
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(directory / description["onnx_file"]), options, providers=["CPUExecutionProvider"]
        )

    def distribution_inputs(self, batch: dict[str, np.ndarray]) -> np.ndarray:
        feed = {}
        for entry in self.inputs:
            value = batch
            for key in entry["observation_path"]:
                value = value[key]
            feed[entry["name"]] = np.asarray(value, dtype=entry["dtype"])
        return self._session.run([self.output], feed)[0]


def load_policy(checkpoint: Checkpoint, observation_space) -> ExportedPolicy:
    """The exported policy of ``checkpoint``, which must have been trained on this observation."""
    policy = ExportedPolicy(policy_directory(Path(checkpoint.path)))
    trained, played = (
        {key: space.shape for key, space in spaces.spaces.items()}
        for spaces in (policy.observation_space, observation_space)
    )
    if trained != played:
        differences = sorted(key for key in trained.keys() | played.keys() if trained.get(key) != played.get(key))
        raise ValueError(f"{checkpoint.path} was trained on another observation; differences: {differences}")
    return policy


class CheckpointPlayers:
    """The actions of the teams played by checkpoints; ``policies``: (policy, deterministic) by agent name prefix, a
    policy giving the action distribution inputs of a batch (ExportedPolicy)."""

    def __init__(self, policies: dict[str, tuple]):
        self._policies = policies

    @classmethod
    def of(cls, environment) -> "CheckpointPlayers":
        """The players of the Checkpoint teams of a ReinforcementLearningEnvironment, each on the observation of its
        team."""
        return cls(
            {
                f"team{team}_": (
                    load_policy(controller, environment.observation_space(f"team{team}_unit_0")),
                    controller.deterministic,
                )
                for team, controller in enumerate(environment.configuration.game.controllers)
                if isinstance(controller, Checkpoint)
            }
        )

    def plays(self, agent: str) -> bool:
        return agent.startswith(tuple(self._policies))

    def actions(self, observations: dict[str, dict]) -> dict[str, dict]:
        """The action of every agent of ``observations`` that a checkpoint plays, one batch per team."""
        actions = {}
        for prefix, (policy, deterministic) in self._policies.items():
            agents = [agent for agent in observations if agent.startswith(prefix)]
            if not agents:
                continue
            batch = {key: np.stack([observations[agent][key] for agent in agents]) for key in observations[agents[0]]}
            inputs = torch.from_numpy(policy.distribution_inputs(batch))
            with torch.no_grad():
                distribution = action_distribution_class(policy.action_space).from_logits(inputs)
                drawn = (distribution.to_deterministic() if deterministic else distribution).sample()
            for index, agent in enumerate(agents):
                actions[agent] = {"command": int(drawn["command"][index]), "target": drawn["target"][index].numpy()}
        return actions


def team_controller(team: str) -> Controller:
    """The controller named ``team``: a scripted one of beyond_all_reason_environment.controllers, a checkpoint, or a
    training (its folder version_N), for its latest checkpoint."""
    scripted = getattr(controllers, team, None)
    if isinstance(scripted, type) and issubclass(scripted, Controller):
        if scripted is Agent:
            raise ValueError("Agent() needs RLlib to play: a team here is a checkpoint or a scripted controller")
        return scripted()
    path = Path(team)
    if (path / CHECKPOINT_FILE).exists():
        return Checkpoint(path)
    checkpoints = sorted(path.glob("*/checkpoint_*"), key=lambda checkpoint: checkpoint.name)
    if not checkpoints:
        raise ValueError(
            f"{team} is neither a controller of beyond_all_reason_environment nor a checkpoint or training"
        )
    return Checkpoint(checkpoints[-1])


def team_outcome(informations: dict[str, dict], team: int) -> str:
    """ "victory", "defeat" or "draw" of ``team``, which agents play, from the informations of the last step of a game
    (it gives those of every agent, its dead units too)."""
    information = informations[f"team{team}_unit_0"]
    allies_defeated, enemies_defeated = bool(information["allies_defeated"]), bool(information["enemies_defeated"])
    if enemies_defeated and not allies_defeated:
        return "victory"
    return "defeat" if allies_defeated and not enemies_defeated else "draw"
