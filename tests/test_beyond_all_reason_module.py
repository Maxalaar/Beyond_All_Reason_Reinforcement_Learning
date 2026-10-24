"""BeyondAllReasonModule's action distribution, without the engine."""

import pytest

pytest.importorskip("maxalaar_reinforcement_learning")

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import (  # noqa: E402
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
