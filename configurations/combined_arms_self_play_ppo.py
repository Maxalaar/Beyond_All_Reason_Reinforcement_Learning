"""PPO training configuration — combined arms in self-play: one policy plays both sides, 3 Shellshockers, 5 Stouts and
10 Pawns each.

Scenario (beyond_all_reason_environment.scenarios.combined_arms_symmetric_arena): the 1v1 arena (flat 1024×1024 elmos,
2 game-minutes) with the same army on each side: 3 Shellshockers (artillery: range 710, slow shells), 5 Stouts (light
tanks: 1800 health, range 350) and 10 Pawns.
Chosen here, for this experiment: Agent() on both sides, the 36 agents with one shared policy (each sees its own
alliance as alliance 0), one weapon slot per unit, every command allowed, and a terminal reward (+1 victory,
-1 defeat; a draw at the time limit is worth 0). Trained from scratch, without the Pawn curriculum.

Why: against scripted opponents (combined_arms_self_play_ppo), the policy degenerated (every unit moved to the
north-west corner, entropy 4.0 -> 6.2): does an opponent that learns too keep it from degenerating?
What to watch: the victory rate means little here (the policy against itself: ~50 %, seen from one side); a rising
draw rate (1 - victory_rate - defeat_rate) shows both sides avoiding the fight, the entropy shows the policy spreading.
No stopping criterion: stopped by hand. The 10 latest checkpoints are kept (no score to rank them).

Expected episode length: at most ~113 decisions (2 game-minutes at 32 frames/decision); ~60 for scripted sides.

Usage (from this repository root, with this package installed: pip install -e .):
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/combined_arms_self_play_ppo.py
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
from beyond_all_reason_environment.scenarios import combined_arms_symmetric_arena
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

combined_arms_self_play_ppo = ExperimentationConfiguration(
    experimentation_name="combined_arms_self_play_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=combined_arms_symmetric_arena, controllers=[Agent(), Agent()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# The reinforcement learning part, under a short name.
configuration = combined_arms_self_play_ppo.reinforcement_learning_configuration

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

# ── Multi-agent: 36 agents per episode (18 per side), shared policy ───────────────────────────
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
# Counted in env steps (complete episodes: ~730 sampled for 600). The dead agents stop acting:
# combined_arms_attack_nearest_ppo sampled ~5700 agent steps per iteration with its 18 agents, so ~11000 here with 36.
# 4000 env steps of 5 agents kept the 31 GB machine at 89-90 % RAM (2026-10-18).
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

# ── No stopping criterion: stopped by hand ───────────────────────────────────
# VictoryCallback still reports the rates (seen from one side) and the draws.
configuration.callback = VictoryCallback
configuration.stopping_criterion = None

# ── Checkpoints: the 10 latest (no score to rank them in self-play) ─────────
configuration.checkpoint_frequency = 20
configuration.number_checkpoint_to_keep = 10
configuration.checkpoint_score_attribute = None
