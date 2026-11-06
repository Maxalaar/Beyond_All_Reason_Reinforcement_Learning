"""RLlib MultiAgentEnv wrapper for ReinforcementLearningEnvironment.

Thin adapter: no business logic, pure delegation. Configuration is passed via
``env_config["configuration"]`` (a ReinforcementLearningEnvironmentConfiguration instance);
falls back to defaults if absent. A ``render_mode`` in env_config (any value, as RLlib evaluation passes it)
opens the game window, like the Godot environments, at ``env_config["game_speed"]`` (a multiple of real time) or at
the graphical speed (real time); in the window, Alt+= and Alt+- change it. The other keys named like a field of
GraphicsConfiguration set it for the window: interface ("complete", "reduced", "cinematic"), hidden_widgets,
video_path (record a video), video_frame_rate... ``"video": true`` records into videos/<experimentation_name>_<n>.mp4
(n following the videos already there; experimentation_name: given by display_episodes.py, "video" otherwise),
relative to the working directory (the repository root). From display_episodes.py:
``--environment_configuration '{"game_speed": 2, "interface": "cinematic", "video": true}'``.

All agents share the same observation and action spaces (homogeneous, shared policy).
An agent whose unit dies stops acting but stays in the episode: RLlib keeps its last action hanging, adds up its
rewards and completes that step at the end of the episode, so the whole team shares the victory or the defeat and no
step of a dead unit enters the batches. ``__all__`` is set once no agent is left: in truncations when the time limit
ended the episode, otherwise in terminations (one side defeated).

The teams with a Checkpoint controller are played here: RLlib only sees the agents of the Agent() teams.
"""

from dataclasses import fields, replace
from pathlib import Path

from beyond_all_reason_environment.engine_configuration import GraphicsConfiguration
from beyond_all_reason_environment.reinforcement_learning_environment import ReinforcementLearningEnvironment
from beyond_all_reason_environment.reinforcement_learning_environment_configuration import (
    ReinforcementLearningEnvironmentConfiguration,
)
from beyond_all_reason_environment.video_recording import next_video_path
from ray.rllib import MultiAgentEnv

from beyond_all_reason_reinforcement_learning.checkpoint_controller import CheckpointPlayers

# Where "video": true records, relative to the working directory (the repository root, see .gitignore).
VIDEOS_DIRECTORY = Path("videos")
# Keys of env_config that set the window, with render_mode.
GRAPHICS_FIELDS = {field.name for field in fields(GraphicsConfiguration)}


def window_configuration(
    configuration: ReinforcementLearningEnvironmentConfiguration, env_config: dict
) -> ReinforcementLearningEnvironmentConfiguration:
    """``configuration``, with the game window if ``env_config`` asks to render (to watch a trained policy), at the
    speed asked or the graphical one."""
    if env_config.get("render_mode") is None:
        return configuration
    speed = env_config.get("game_speed")
    graphics = {
        key: tuple(value) if key == "hidden_widgets" else value
        for key, value in env_config.items()
        if key in GRAPHICS_FIELDS
    }
    if env_config.get("video") and "video_path" not in graphics:
        name = env_config.get("experimentation_name") or "video"
        graphics["video_path"] = next_video_path(VIDEOS_DIRECTORY, name)
    engine = configuration.engine
    return replace(
        configuration,
        engine=replace(
            engine,
            speed=None if speed is None else float(speed),
            graphics=replace(engine.graphics, **{**graphics, "show": True}),
        ),
    )


class RayMultiAgentEnvironment(MultiAgentEnv):
    def __init__(self, env_config=None):
        super().__init__()
        env_config = env_config or {}
        configuration = env_config.get("configuration") or ReinforcementLearningEnvironmentConfiguration()
        self._env = ReinforcementLearningEnvironment(window_configuration(configuration, env_config))
        self._checkpoint_players = CheckpointPlayers.of(self._env)
        self._observations = {}
        self.possible_agents = [
            agent for agent in self._env.possible_agents if not self._checkpoint_players.plays(agent)
        ]
        per_agent_obs = self._env.observation_space(self._env.possible_agents[0])
        per_agent_act = self._env.action_space(self._env.possible_agents[0])
        self.observation_spaces = {aid: per_agent_obs for aid in self.possible_agents}
        self.action_spaces = {aid: per_agent_act for aid in self.possible_agents}

    def _for_rllib(self, values: dict) -> dict:
        """``values`` without the agents the checkpoints play."""
        return {agent: value for agent, value in values.items() if not self._checkpoint_players.plays(agent)}

    def reset(self, *, seed=None, options=None):
        self._observations, infos = self._env.reset(seed=seed, options=options)
        return self._for_rllib(self._observations), self._for_rllib(infos)

    def step(self, action_dict):
        actions = {**action_dict, **self._checkpoint_players.actions(self._observations)}
        self._observations, rewards, terminations, truncations, infos = self._env.step(actions)
        observations = self._for_rllib(self._observations)
        rewards, terminations, truncations, infos = map(self._for_rllib, (rewards, terminations, truncations, infos))
        # A dead unit does not end the episode: only when the environment has no agent left.
        over = not self._env.agents
        truncations["__all__"] = over and any(truncations.values())
        terminations["__all__"] = over and not truncations["__all__"]
        return observations, rewards, terminations, truncations, infos

    def close(self):
        self._env.close()
