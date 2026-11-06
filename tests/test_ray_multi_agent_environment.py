"""RLlib adapter: when the episode ends for every agent (__all__), without the engine."""

import pytest

pytest.importorskip("ray")

from beyond_all_reason_environment.engine_configuration import GRAPHICAL_SPEED, HEADLESS_SPEED  # noqa: E402

from beyond_all_reason_reinforcement_learning.checkpoint_controller import CheckpointPlayers  # noqa: E402
from beyond_all_reason_reinforcement_learning.ray_multi_agent_environment import RayMultiAgentEnvironment  # noqa: E402


class ScriptedEnvironment:
    """Stands for ReinforcementLearningEnvironment: each step plays the next (terminations, truncations, agents
    left)."""

    def __init__(self, steps):
        self.agents = ["a", "b"]
        self._steps = iter(steps)

    def step(self, actions):
        terminations, truncations, self.agents = next(self._steps)
        done = {**terminations, **truncations}
        return {agent: None for agent in done}, {agent: 0.0 for agent in done}, terminations, truncations, {}


def adapter(steps, players=None) -> RayMultiAgentEnvironment:
    environment = RayMultiAgentEnvironment.__new__(RayMultiAgentEnvironment)
    environment._env = ScriptedEnvironment(steps)
    environment._checkpoint_players = players or CheckpointPlayers({})
    environment._observations = {}
    return environment


def test_rllib_does_not_see_the_agents_a_checkpoint_plays():
    class Players(CheckpointPlayers):
        def actions(self, observations):
            return {agent: "played by the checkpoint" for agent in observations if self.plays(agent)}

    environment = adapter(
        [
            ({"a": False, "team1_unit_0": False}, {"a": False, "team1_unit_0": False}, ["a", "team1_unit_0"]),
            ({"a": True, "team1_unit_0": True}, {"a": False, "team1_unit_0": False}, []),
        ],
        Players({"team1_": None}),
    )
    received = []
    step = environment._env.step
    environment._env.step = lambda actions: (received.append(actions), step(actions))[1]
    observations, rewards, terminations, truncations, _ = environment.step({"a": "played by RLlib"})
    assert received == [{"a": "played by RLlib"}]
    assert set(observations) == set(rewards) == {"a"}
    assert terminations == {"a": False, "__all__": False} and truncations == {"a": False, "__all__": False}
    # The observations of the step before give the checkpoint's next action.
    _, _, terminations, _, _ = environment.step({"a": "played by RLlib"})
    assert received[1] == {"a": "played by RLlib", "team1_unit_0": "played by the checkpoint"}
    assert terminations == {"a": True, "__all__": True}


def test_the_death_of_one_agent_does_not_end_the_episode():
    environment = adapter(
        [
            ({"a": True, "b": False}, {"a": False, "b": False}, ["b"]),  # a's unit dies
            ({"b": True}, {"b": False}, []),  # b wins or dies: the episode is over
        ]
    )
    _, _, terminations, truncations, _ = environment.step({"a": {}, "b": {}})
    assert terminations == {"a": True, "b": False, "__all__": False}
    assert truncations["__all__"] is False
    _, _, terminations, truncations, _ = environment.step({"b": {}})
    assert terminations["__all__"] is True and truncations["__all__"] is False


def test_the_time_limit_truncates_the_episode():
    environment = adapter([({"a": False, "b": False}, {"a": True, "b": True}, [])])
    _, _, terminations, truncations, _ = environment.step({"a": {}, "b": {}})
    assert truncations["__all__"] is True and terminations["__all__"] is False


def test_a_render_mode_opens_the_game_window_at_the_graphical_speed():
    # As RLlib's evaluation passes it (display_episodes.py of Maxalaar_Reinforcement_Learning); no engine starts here.
    watched = RayMultiAgentEnvironment({"render_mode": "rgb_array"})._env.configuration.engine
    assert watched.graphics.show and watched.effective_speed == GRAPHICAL_SPEED
    trained = RayMultiAgentEnvironment({})._env.configuration.engine
    assert not trained.graphics.show and trained.effective_speed == HEADLESS_SPEED


def test_rllib_gives_a_dead_agent_the_team_outcome_on_its_last_action():
    """The contract of ReinforcementLearningEnvironment.step on a real MultiAgentEpisode: an agent stops having
    observations, and its last action is completed at the end of the episode with the team's reward."""
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

    space = spaces.Box(-1.0, 1.0, (1,))
    agents = ["alive", "dead"]
    episode = MultiAgentEpisode(
        observation_space=dict.fromkeys(agents, space), action_space=dict.fromkeys(agents, space)
    )
    observation = [0.0]
    not_done = dict.fromkeys(agents, False)
    episode.add_env_reset(observations=dict.fromkeys(agents, observation), infos=dict.fromkeys(agents, {}))
    # The unit of "dead" dies during this step: no observation, not terminated.
    episode.add_env_step(
        observations={"alive": observation},
        actions=dict.fromkeys(agents, [0.0]),
        rewards=dict.fromkeys(agents, 0.0),
        terminateds=not_done,
        truncateds=not_done,
    )
    assert len(episode.agent_episodes["dead"]) == 0
    episode.add_env_step(
        observations={"alive": observation},
        actions={"alive": [0.0]},
        rewards=dict.fromkeys(agents, 0.0),
        terminateds=not_done,
        truncateds=not_done,
    )
    # The team wins: the last step gives every agent its observation, reward and end.
    episode.add_env_step(
        observations=dict.fromkeys(agents, observation),
        actions={"alive": [0.0]},
        rewards=dict.fromkeys(agents, 1.0),
        terminateds={**dict.fromkeys(agents, True), "__all__": True},
        truncateds={**not_done, "__all__": False},
    )
    dead = episode.agent_episodes["dead"]
    assert dead.is_terminated and len(dead) == 1 and list(dead.get_rewards()) == [1.0]
    assert list(episode.agent_episodes["alive"].get_rewards()) == [0.0, 0.0, 1.0]


def test_rllib_accepts_a_dead_agent_at_the_time_limit():
    """A draw: the last step gives the dead agent an observation too, which RLlib asks for at a truncation."""
    from gymnasium import spaces
    from ray.rllib.env.multi_agent_episode import MultiAgentEpisode

    space = spaces.Box(-1.0, 1.0, (1,))
    agents = ["alive", "dead"]
    episode = MultiAgentEpisode(
        observation_space=dict.fromkeys(agents, space), action_space=dict.fromkeys(agents, space)
    )
    observation = [0.0]
    not_done = dict.fromkeys(agents, False)
    episode.add_env_reset(observations=dict.fromkeys(agents, observation), infos=dict.fromkeys(agents, {}))
    episode.add_env_step(
        observations={"alive": observation},
        actions=dict.fromkeys(agents, [0.0]),
        rewards=dict.fromkeys(agents, 0.0),
        terminateds=not_done,
        truncateds=not_done,
    )
    episode.add_env_step(
        observations=dict.fromkeys(agents, observation),
        actions={"alive": [0.0]},
        rewards=dict.fromkeys(agents, 0.0),
        terminateds={**not_done, "__all__": False},
        truncateds={**dict.fromkeys(agents, True), "__all__": True},
    )
    assert episode.is_done and len(episode.agent_episodes["dead"]) == 1
    assert list(episode.agent_episodes["dead"].get_rewards()) == [0.0]
