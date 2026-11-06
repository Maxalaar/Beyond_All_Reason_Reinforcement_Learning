"""A team played by the policy of a checkpoint, without RLlib's Algorithm.

For the environment, Checkpoint is an Agent: the units of its team are agents, decided from Python. CheckpointPlayers
decides for them: RayMultiAgentEnvironment hides them from RLlib, as if they were a scripted team, and
scripts/play_games.py plays whole games with them alone.

The actions are drawn as RLlib's GetActions does, on the CPU, as the env runners do during a training.
"""

from dataclasses import dataclass
from pathlib import Path

import beyond_all_reason_environment.controllers as controllers
import numpy as np
import torch
from beyond_all_reason_environment.controllers import Agent, Controller
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

# Where a checkpoint keeps the weights of the shared policy, and what marks a folder as a checkpoint.
POLICY = Path("learner_group") / "learner" / "rl_module" / "shared_policy"
CHECKPOINT_FILE = "rllib_checkpoint.json"


@dataclass(frozen=True)
class Checkpoint(Agent):
    """The policy of the checkpoint at ``path`` plays the team: the most likely action if ``deterministic``, a draw
    otherwise."""

    path: str
    deterministic: bool = False

    def __post_init__(self) -> None:
        # RLModule.from_checkpoint needs an absolute path, and Ray's workers may not share the working directory.
        object.__setattr__(self, "path", str(Path(self.path).resolve()))

    @property
    def label(self) -> str:
        return Path(self.path).name


def load_policy(checkpoint: Checkpoint, observation_space) -> torch.nn.Module:
    """The policy of ``checkpoint``, on the CPU, which must have been trained on this observation."""
    module = RLModule.from_checkpoint(str(Path(checkpoint.path) / POLICY)).to("cpu").eval()
    trained, played = (
        {key: space.shape for key, space in spaces.spaces.items()}
        for spaces in (module.observation_space, observation_space)
    )
    if trained != played:
        differences = sorted(key for key in trained.keys() | played.keys() if trained.get(key) != played.get(key))
        raise ValueError(f"{checkpoint.path} was trained on another observation; differences: {differences}")
    return module


class CheckpointPlayers:
    """The actions of the teams played by checkpoints; ``policies``: (policy, deterministic) by agent name prefix."""

    def __init__(self, policies: dict[str, tuple]):
        self._policies = policies

    @classmethod
    def of(cls, environment) -> "CheckpointPlayers":
        """The players of the Checkpoint teams of a ReinforcementLearningEnvironment."""
        observation_space = environment.observation_space(environment.possible_agents[0])
        return cls(
            {
                f"team{team}_": (load_policy(controller, observation_space), controller.deterministic)
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
            batch = {
                key: torch.from_numpy(np.stack([observations[agent][key] for agent in agents]))
                for key in observations[agents[0]]
            }
            with torch.no_grad():
                inputs = policy.forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
                distribution = policy.get_inference_action_dist_cls().from_logits(inputs)
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
