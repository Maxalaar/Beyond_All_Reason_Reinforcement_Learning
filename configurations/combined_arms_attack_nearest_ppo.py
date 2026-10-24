"""PPO training configuration — combined arms: 3 Shellshockers, 5 Stouts and 10 Pawns against 3 to 6 Stouts and 15 to 30
AttackNearest Pawns (curriculum step 3).

Scenario (beyond_all_reason_environment.scenarios.combined_arms_arena): the 1v1 arena (flat 1024×1024 elmos,
2 game-minutes) with 3 Shellshockers (artillery: range 710, slow shells), 5 Stouts (light tanks: 1800 health, range
350) and 10 Pawns on the west side, 3 to 6 Stouts and 15 to 30 Pawns on the east side, drawn for each episode.
Chosen here, for this experiment: the 18 agents (one shared policy for every unit: a unit is known by its chassis and
weapons) against AttackNearest opponents, one weapon slot per unit, every command allowed, and a terminal reward
(+1 victory, -1 defeat).

Why: Pawn against Pawn left little to learn beyond numbers (five_versus_one_to_five_attack_nearest_ppo plateaued at
~80 %: 100 % against 1 or 2, 35 % against 5). Here a line of Stouts in front, which the opponents attack first, lets
the Pawns and the Shellshockers fire safely: the formation pays against the larger numbers. Against Pawns only, the
west Stouts won every game, even for the stage 2 policy (100 % at the first iteration): the opponents have Stouts too.
VictoryCallback reports the victory rate against each number of opponents (env_runners/victory_rate_against_<number>).
Stops at 80 % victories: with up to 30 opponents, 90 % may be out of reach.
In pawn_curriculum.py it starts from five_versus_five_passive_ppo.py.

Starting point (40 games each, 2026-10-23): the stage 2 policy (actions sampled) wins 50 % (58 % up to 26 opponents,
33 % from 32), a script attacking the nearest enemy 70 %; 80 % asks for better than the script.
Expected episode length: at most ~113 decisions (2 game-minutes at 32 frames/decision); ~50 in practice.

Usage (from this repository root, with this package installed: pip install -e .):
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/combined_arms_attack_nearest_ppo.py
"""

import torch.nn as nn
from beyond_all_reason_environment import (
    ActionConfiguration,
    Agent,
    AttackNearest,
    GameConfiguration,
    ObservationConfiguration,
    ReinforcementLearningEnvironmentConfiguration,
    RewardWeights,
)
from beyond_all_reason_environment.scenarios import combined_arms_arena
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

combined_arms_attack_nearest_ppo = ExperimentationConfiguration(
    experimentation_name="combined_arms_attack_nearest_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=combined_arms_arena, controllers=[Agent(), AttackNearest()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# The reinforcement learning part, under a short name.
configuration = combined_arms_attack_nearest_ppo.reinforcement_learning_configuration

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

# ── Multi-agent: 18 agents per episode, shared policy ───────────────────────────
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
# 600 env steps (~730 sampled, complete episodes) = ~5700 agent steps (18 agents, the dead ones stop acting; measured in
# pawn_curriculum version_3). 4000 env steps of 5 agents kept the 31 GB machine at 89-90 % RAM (2026-10-18).
configuration.train_batch_size = 600
configuration.minibatch_size = 512
configuration.batch_mode = "complete_episodes"
configuration.sample_timeout_s = 180

# ── Resources ─────────────────────────────────────────────────────────────────
# Each engine process uses ~2.3 CPU cores and ~4.7 GB RAM.
# 3 runners: ~14 GB RAM, ~7 cores for engines. 4 runners (~19 GB) froze the 31 GB machine (no swap) on 2026-10-18.
configuration.number_environment_runners = 3
configuration.number_cpus_per_environment_runners = 3
configuration.number_gpus_per_environment_runners = 0
configuration.number_learners = 1
configuration.number_cpus_per_learner = 1
configuration.number_gpus_per_learner = 1

# ── Evaluation ────────────────────────────────────────────────────────────────
configuration.evaluation_interval = None

# ── Stopping criterion: 80 % victories ────────────────────────────────────────
# VictoryCallback averages the victories over the last 100 episodes of each env runner (0 before 100 episodes).
# Tune saves a last checkpoint when it stops: the next stage of a curriculum (pawn_curriculum.py) starts from it.
configuration.callback = VictoryCallback
configuration.stopping_criterion = {"env_runners/victory_rate_full_window": 0.8}

# ── Checkpoints ──────────────────────────────────────────────────────────────
configuration.checkpoint_frequency = 20
configuration.number_checkpoint_to_keep = 10
configuration.checkpoint_score_attribute = "env_runners/victory_rate"
configuration.checkpoint_score_order = "max"
