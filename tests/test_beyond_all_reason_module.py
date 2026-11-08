"""BeyondAllReasonModule: its action distribution and its queries, without the engine."""

import pytest

pytest.importorskip("maxalaar_reinforcement_learning")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ray.rllib.core.columns import Columns  # noqa: E402

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import (  # noqa: E402
    BeyondAllReasonModule,
    action_distribution_class,
)

# As beyond_all_reason_environment's action space: a command and a target point.
ACTION_SPACE = gym.spaces.Dict({"command": gym.spaces.Discrete(5), "target": gym.spaces.Box(0, 1, (2,), np.float32)})
DISTRIBUTION_CLASS = action_distribution_class(ACTION_SPACE)


def logits(target_mean: float, target_log_standard_deviation: float, samples: int = 5) -> torch.Tensor:
    """Command logits at random, then (mean x, mean z, log std x, log std z) of the target before squashing."""
    target = torch.tensor([target_mean, target_mean, target_log_standard_deviation, target_log_standard_deviation])
    return torch.cat([torch.randn(samples, 5), target.expand(samples, 4)], dim=-1)


def test_deterministic_actions_have_one_log_probability_per_sample():
    # What RLlib's GetActions does without exploration (evaluation, display_episodes.py): it crashed with
    # "The size of tensor a (5) must match the size of tensor b (2)".
    assert DISTRIBUTION_CLASS.required_input_dim(ACTION_SPACE) == 9
    output = logits(0.0, 0.0)
    distribution = DISTRIBUTION_CLASS.from_logits(output)

    deterministic = distribution.to_deterministic()
    actions = deterministic.sample()
    assert deterministic.logp(actions).shape == (5,)
    assert torch.equal(actions["command"], output[:, :5].argmax(dim=-1))
    assert distribution.logp(distribution.sample()).shape == (5,)  # sampling (training) is unchanged


def test_every_target_drawn_is_a_map_point():
    # The mean where combined_arms_attack_nearest_ppo's policy drifted: (-4.4, -4.3), standard deviation ~5.
    distribution = DISTRIBUTION_CLASS.from_logits(logits(-4.4, np.log(5.0), samples=1000))
    targets = distribution.sample()["target"]
    assert targets.min() >= 0.0 and targets.max() <= 1.0
    assert torch.isfinite(distribution.logp(distribution.sample())).all()
    # Without exploration, the squashed mean: (tanh(0) + 1) / 2 = the map's middle for a mean of 0.
    middle = DISTRIBUTION_CLASS.from_logits(logits(0.0, 0.0)).to_deterministic().sample()["target"]
    assert torch.allclose(middle, torch.full_like(middle, 0.5))


def test_the_target_entropy_falls_when_its_draws_pile_at_the_map_edges():
    # The Gaussian's entropy grows without end with the spread; the squashed distribution's does not.
    def target_entropy(log_standard_deviation: float) -> float:
        distribution = DISTRIBUTION_CLASS.from_logits(logits(0.0, log_standard_deviation, samples=2000))
        command_entropy = torch.distributions.Categorical(logits=distribution._flat_child_distributions[0].logits)
        return float((distribution.entropy() - command_entropy.entropy()).mean())

    assert target_entropy(0.0) > target_entropy(-2.0)  # spreading over the map raises it
    assert target_entropy(0.0) > target_entropy(np.log(5.0))  # piling at the edges lowers it
    assert target_entropy(0.0) < 0.0  # on a map of side 1, at most log(1) = 0 per dimension


def test_the_kl_is_zero_between_equal_distributions_and_positive_otherwise():
    output = logits(0.3, -1.0)
    distribution = DISTRIBUTION_CLASS.from_logits(output)
    assert torch.allclose(distribution.kl(DISTRIBUTION_CLASS.from_logits(output)), torch.zeros(5), atol=1e-6)
    shifted = output.clone()
    shifted[:, 5:7] += 1.0
    assert (distribution.kl(DISTRIBUTION_CLASS.from_logits(shifted)) > 0).all()


def small_module(**model_config):
    from beyond_all_reason_environment.controllers import Agent, AttackNearest
    from beyond_all_reason_environment.game import GameConfiguration
    from beyond_all_reason_environment.reinforcement_learning_environment import ReinforcementLearningEnvironment
    from beyond_all_reason_environment.reinforcement_learning_environment_configuration import (
        ReinforcementLearningEnvironmentConfiguration,
    )
    from beyond_all_reason_environment.scenarios import combined_arms_arena

    controllers = [Agent(), *[AttackNearest()] * (len(combined_arms_arena.teams) - 1)]
    configuration = ReinforcementLearningEnvironmentConfiguration(
        game=GameConfiguration(combined_arms_arena, controllers)
    )
    environment = ReinforcementLearningEnvironment(configuration)
    torch.manual_seed(0)
    module = BeyondAllReasonModule(
        observation_space=environment.observation_space("team0_unit_0"),
        action_space=environment.action_space("team0_unit_0"),
        model_config={
            "d_level1": 16,
            "critic_d": 16,
            "terrain_cnn_channels": [4],
            "action_hidden_sizes": [16],
            **model_config,
        },
    ).cpu()
    return module, environment.observation_space("team0_unit_0")


def batch_of(space, size: int) -> dict:
    space.seed(0)
    samples = [space.sample() for _ in range(size)]
    for sample in samples:
        sample["command_mask"][:] = 1.0
    return {key: torch.from_numpy(np.stack([sample[key] for sample in samples])) for key in space.spaces}


def test_the_units_ask_each_category_with_queries_made_from_their_own_token():
    module, space = small_module(number_queries_per_category=3)
    # No learned query left in the actor; the centralized critic, which observes from no unit, keeps its own.
    actor_queries = [name for name, _ in module.named_parameters() if "learned_query" in name]
    assert actor_queries and all(name.startswith("centralized_critic.") for name in actor_queries)

    batch = batch_of(space, 4)
    with torch.no_grad():
        outputs = module.forward_inference({Columns.OBS: batch})[Columns.ACTION_DIST_INPUTS]
        # The same entities seen by another unit: the category tokens change through the queries alone.
        self_token = module.actor_entity_encoder(
            batch["self_chassis"], batch["self_state"], batch["self_weapons"], batch["self_weapon_mask"]
        )
        queries = module.actor_category_query_enemy_units(self_token).reshape(4, 3, 16)
    assert outputs.shape == (4, 9)
    assert not torch.allclose(queries[0], queries[1])


def test_the_policy_with_several_queries_per_category_exports(tmp_path):
    pytest.importorskip("onnxruntime")
    from maxalaar_reinforcement_learning.export.policy_export import export_policy

    module, _ = small_module(number_queries_per_category=2)
    export_policy(module, tmp_path)
    assert (tmp_path / "policy.onnx").is_file()
