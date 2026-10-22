"""PPO training configuration — a Shellshocker and 10 Pawns against 10 to 20 AttackNearest Pawns.

Scenario (beyond_all_reason_environment.scenarios.artillery_and_ten_versus_ten_to_twenty_arena): the 1v1 arena (flat
1024×1024 elmos, 2 game-minutes) with a Shellshocker (armart, artillery: range 710, 182 damage per shell, 4.3 s
reload, 620 health) and 10 Pawns on the west side, 10 to 20 Pawns on the east side, drawn for each episode.
Chosen here, for this experiment: the 11 agents (one shared policy, the Shellshocker included) against AttackNearest
Pawns, one weapon slot per unit, every command allowed, and a terminal reward (+1 victory, -1 defeat).

Why: Pawn against Pawn left little to learn beyond numbers (five_versus_one_to_five_attack_nearest_ppo plateaued at
~80 %: 100 % against 1 or 2, 35 % against 5). The Shellshocker outranges the Pawns but dies at close range: in the
uneven games, the west side wins by screening it while it fires, not by charging.
VictoryCallback reports the victory rate against each number of opponents (env_runners/victory_rate_against_<number>).
Stops at 80 % victories: with up to twice as many opponents, 90 % may be out of reach.

Starts from the weights of five_versus_five_passive_ppo.py (curriculum step 2), adapted by
scripts/add_unit_types_to_checkpoint.py: this scenario adds the armart type to the chassis one-hot (24 -> 25 chassis
features), a zero column in the first layer of each entity encoder, so the Pawns keep their weights.

Expected episode length: at most ~113 decisions (2 game-minutes at 32 frames/decision); ~25 in practice.

Usage (from this repository root, with this package installed: pip install -e .):
    python scripts/add_unit_types_to_checkpoint.py \
        --checkpoint_path <five_versus_five_passive_ppo checkpoint> \
        --old_unit_types armpw --new_unit_types armart armpw \
        --output_path experiments/artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo/initialization_checkpoint
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo.py
"""

from pathlib import Path

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
from beyond_all_reason_environment.scenarios import artillery_and_ten_versus_ten_to_twenty_arena
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo = ExperimentationConfiguration(
    experimentation_name="artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(
                scenario=artillery_and_ten_versus_ten_to_twenty_arena, controllers=[Agent(), AttackNearest()]
            ),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# The reinforcement learning part, under a short name.
configuration = artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo.reinforcement_learning_configuration

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

# ── Initialization: the weights of five_versus_five_passive_ppo.py, with the armart type added ──────────────
configuration.initialization_checkpoint_path = Path(
    "experiments/artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo/initialization_checkpoint"
)

# ── Multi-agent: 11 agents per episode, shared policy ───────────────────────────
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
# 1000 env steps = ~11000 agent steps (11 agents), as many as the 5-agent stages (2000 env steps): 4000 kept the 31 GB
# machine at 89-90 % RAM (2026-10-18).
configuration.train_batch_size = 1000
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
# VictoryCallback averages the victories over the last 100 episodes of each env runner. Tune saves a
# last checkpoint when it stops: the next stage of a curriculum (pawn_curriculum.py) starts from it.
configuration.callback = VictoryCallback
configuration.stopping_criterion = {"env_runners/victory_rate": 0.8}

# ── Checkpoints ──────────────────────────────────────────────────────────────
configuration.checkpoint_frequency = 20
configuration.number_checkpoint_to_keep = 10
configuration.checkpoint_score_attribute = "env_runners/victory_rate"
configuration.checkpoint_score_order = "max"
