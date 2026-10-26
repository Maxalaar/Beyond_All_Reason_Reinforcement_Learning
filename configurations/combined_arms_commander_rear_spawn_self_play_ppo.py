"""PPO training configuration — combined arms around a commander, in self-play, each side spawned in the rear of
its half: combined_arms_commander_self_play_ppo on the scenario combined_arms_commander_arena_rear_spawn
(spawn_depth 0.5 instead of 1.0), from its weights.

Scenario (beyond_all_reason_environment.scenarios.combined_arms_commander_arena_rear_spawn): the 1v1 arena (flat
1024×1024 elmos, 2 game-minutes), a commander, 3 Shellshockers, 5 Stouts and 10 Pawns per side, each side spawned in
the half of its half nearest its own edge; the game ends when a commander dies.
Everything else as combined_arms_commander_self_play_ppo (see it): Agent() on both sides, one shared policy, a
terminal reward, every command allowed, "fight" for the commander too.

Why: from the rear, the armies have room to form up before the fight, and the draw of the positions weighs less on
who wins. combined_arms_commander_self_play_ppo version_0 at iteration 550, trained with the whole half, won 75 % of
450 games against AttackNearest from the rear (59 % with the whole half): the policy goes on learning where its
manoeuvres count most.
version_0 started from the weights of combined_arms_commander_self_play_ppo version_0 at iteration 550 (2026-10-26:
--initialization_checkpoint_path, a read-only copy of its checkpoint_000110 in
experiments/combined_arms_commander_self_play_ppo/saved_checkpoints/version_0_iteration_550).
What to watch: the draw rate and the episode length (armies further apart, longer episodes), the entropy (-1.4 at
the end of version_0, still falling). No stopping criterion: stopped by hand.

Usage (from this repository root): experiments/combined_arms_commander_rear_spawn_self_play_ppo/supervise_training.sh
(stops and resumes for the RAM; INITIALIZATION_CHECKPOINT_PATH=... for the weights to start from), or
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/combined_arms_commander_rear_spawn_self_play_ppo.py \
        [--initialization_checkpoint_path <checkpoint>]
"""

import torch.nn as nn
from beyond_all_reason_environment import (
    ActionConfiguration,
    Agent,
    GameConfiguration,
    ObservationConfiguration,
    ReinforcementLearningEnvironmentConfiguration,
    RewardWeights,
)
from beyond_all_reason_environment.scenarios import combined_arms_commander_arena_rear_spawn
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

combined_arms_commander_rear_spawn_self_play_ppo = ExperimentationConfiguration(
    experimentation_name="combined_arms_commander_rear_spawn_self_play_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=combined_arms_commander_arena_rear_spawn, controllers=[Agent(), Agent()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# The reinforcement learning part, under a short name.
configuration = combined_arms_commander_rear_spawn_self_play_ppo.reinforcement_learning_configuration

# ── Architecture ──────────────────────────────────────────────────────────────
configuration.architecture = BeyondAllReasonModule
configuration.architecture_configuration = {
    # Actor
    "d_level1": 64,
    "d_level2": 64,
    "d_level3": 64,
    "entity_projector_hidden_sizes": [64],
    "weapon_projector_hidden_sizes": [32],
    "number_entity_attention_heads": 1,
    "number_category_attention_heads": 1,
    "number_global_attention_heads": 1,
    "number_global_attention_layers": 1,
    "global_feedforward_size": 128,
    "context_projector_hidden_sizes": [],
    "terrain_cnn_channels": [16, 32, 32],
    "action_hidden_sizes": [64, 32],
    "activation_function_class": nn.ReLU,
    # Centralized critic
    "critic_d": 64,
    "critic_entity_projector_hidden_sizes": [64],
    "critic_weapon_projector_hidden_sizes": [32],
    "critic_number_entity_attention_heads": 1,
    "critic_number_category_attention_heads": 1,
    "critic_number_global_attention_heads": 1,
    "critic_number_global_attention_layers": 1,
    "critic_global_feedforward_size": 128,
    "critic_context_projector_hidden_sizes": [],
    "critic_terrain_cnn_channels": [16, 32, 32],
}

# ── Ray: an object store of 3 GB (/dev/shm); Ray's default, 30 % of the RAM, let it grow +0.5 GB/hour ────────
combined_arms_commander_rear_spawn_self_play_ppo.ray_object_store_memory = 3 * 1024**3

# ── Multi-agent: 38 agents per episode (19 per side), shared policy ───────────────────────────
configuration.is_multi_agents = True
configuration.policies = {"shared_policy"}
configuration.policy_mapping_function = lambda agent_id, episode, **kwargs: "shared_policy"

# ── Environment ───────────────────────────────────────────────────────────────
configuration.disable_environment_checking = True
configuration.normalize_actions = False
configuration.flatten_observations = False
configuration.compress_observations = False

# ── PPO hyperparameters ───────────────────────────────────────────────────────
configuration.training_name = "base"
configuration.gamma = 0.995
configuration.lambda_gae = 0.95
configuration.learning_rate = 3e-4
configuration.gradient_clip = 0.5
configuration.clip_policy_parameter = 0.2
# 0.005 let the entropy drift up in five_versus_one_to_five_attack_nearest_ppo (4.8 -> 6.7 over 3 hours).
configuration.entropy_coefficient = 0.001
configuration.value_function_loss_coefficient = 1.0
configuration.number_epochs = 10
# A batch of a fixed size, in agent steps: 12288 (24 minibatches of 512), to a step of the 38 agents per env runner.
# Counted in env steps with complete episodes (600), it ranged from 10280 to 20387 agent steps over 848 iterations
# (longer episodes, fewer deaths), until the learner ran out of GPU memory (8 GB) on 2026-10-25. Truncated episodes go
# on at the next iteration; PPO bootstraps their value with the critic.
configuration.count_steps_by = "agent_steps"
configuration.batch_mode = "truncate_episodes"
configuration.rollout_fragment_length = "auto"
configuration.train_batch_size = 12288
configuration.minibatch_size = 512
configuration.sample_timeout_s = 180

# ── Resources ─────────────────────────────────────────────────────────────────
# Each engine process uses ~2.3 CPU cores and ~4.7 GB RAM.
# 3 runners: ~14 GB RAM, ~7 cores for engines. 4 runners (~19 GB) froze the 31 GB machine (no swap) on 2026-10-18.
# 2 runners: with 3, the 36 agents took the machine over 88 % RAM within a minute (2026-10-25), before any iteration.
configuration.number_environment_runners = 2
configuration.number_cpus_per_environment_runners = 3
configuration.number_gpus_per_environment_runners = 0
configuration.number_learners = 1
configuration.number_cpus_per_learner = 1
configuration.number_gpus_per_learner = 1

# ── Evaluation ────────────────────────────────────────────────────────────────
configuration.evaluation_interval = None

# ── No stopping criterion: stopped by hand ───────────────────────────────────
# VictoryCallback still reports the rates (seen from one side) and the draws.
configuration.callback = VictoryCallback
configuration.stopping_criterion = None

# ── Checkpoints: the 10 latest (no score to rank them in self-play) ─────────
# Every 5 iterations: a restart for the RAM (supervise_training.sh) loses at most 5.
configuration.checkpoint_frequency = 5
configuration.number_checkpoint_to_keep = 10
configuration.checkpoint_score_attribute = None
# display_episodes.py shows the latest checkpoint (it needs a metric to choose one).
combined_arms_commander_rear_spawn_self_play_ppo.display_episodes_configuration.checkpoint_score_attribute = (
    "training_iteration"
)
combined_arms_commander_rear_spawn_self_play_ppo.display_episodes_configuration.checkpoint_score_order = "max"
