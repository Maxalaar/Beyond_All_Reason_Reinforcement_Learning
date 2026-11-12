"""PPO training configuration — combined arms around a commander, in self-play: one policy plays both sides, a
commander, 3 Shellshockers, 5 Stouts and 10 Pawns each; the side whose commander dies loses.

Scenario (beyond_all_reason_environment.scenarios.combined_arms_commander_arena): the 1v1 arena (flat 1024×1024 elmos,
2 game-minutes), each side spawned anywhere in its half, the game ending when a commander dies
(commander_death_ends_game).
Chosen here, for this experiment: Agent() on both sides, the 38 agents with one shared policy, one weapon slot per unit,
and a terminal reward (+1 victory, -1 defeat, 0 for a draw at the time limit). Every command allowed, "fight" for
the commander too (decided with the user): a builder under it attacks the enemies within ~450 elmos first, and repairs
damaged allies or reclaims wrecks only when no enemy is near. Its D-gun needs a manual fire command, which it does
not have.
version_0 was trained from scratch (observation version 1, learned level 2 queries).
version_1 (decided with the user, 2026-11-10) starts from the weights of combined_arms_commander_rear_spawn_self_play_ppo
version_3 at its checkpoint_001005 (iteration ~5030; a read-only copy in
experiments/combined_arms_commander_rear_spawn_self_play_ppo/saved_checkpoints/version_3_checkpoint_001005), with its
observation (version 2) and architecture (queries from the observing unit's token) and the team outcome. Why: trained
from the rear only, version_3 played as well as version_0 and version_1 against AttackNearest (74.5 % of 200 games)
but lost to them head to head (35 % against version_0, 40.5 % against version_1), with an entropy of -5.7 against -1.8:
the rear spawn gives starts that look alike, and the policy may have narrowed to one routine. Spawned anywhere in its
half, it sees more varied situations.
Result (stopped 2026-11-11 at iteration ~1565, checkpoint_000346, evaluated from the rear, 200 games each): nothing
changed, 42 % victories against version_1 of the rear spawn, 33.5 % against its version_0 (18 % draws), 70 % against
AttackNearest; the entropy stayed at about -6. The spawn alone does not explain the gap: the queries from the unit's
token are the main suspect left.

Why: in combined_arms_self_play_ppo (spawn bands at the edges, every unit to destroy), the draws rose to 25-38 % for
hours, episodes reaching the time limit. Here a side can lose at any moment by its commander, and the armies start
scattered over their halves, sometimes in contact: 20 scripted games gave no draw, ~26 decisions per game (2026-10-25).
A strategy has a clear object: protect the own commander, kill the other one.
What to watch: the draw rate (1 - victory_rate - defeat_rate) and the episode length; the victory rate means little in
self-play (~50 %, seen from one side). No stopping criterion: stopped by hand.

Usage (from this repository root, with this package installed: pip install -e .):
experiments/combined_arms_commander_self_play_ppo/start_training.sh (stops and resumes for the RAM;
TRAINING_PATH=<version_N> resumes that version, INITIALIZATION_CHECKPOINT_PATH=<checkpoint> starts from those weights),
or
    python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
        --experimentation_configuration_file configurations/combined_arms_commander_self_play_ppo.py \
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
from beyond_all_reason_environment.scenarios import combined_arms_commander_arena
from maxalaar_reinforcement_learning.configuration.experimentation_configuration import ExperimentationConfiguration

from beyond_all_reason_reinforcement_learning.architecture.beyond_all_reason_module import BeyondAllReasonModule
from beyond_all_reason_reinforcement_learning.victory_callback import VictoryCallback

combined_arms_commander_self_play_ppo = ExperimentationConfiguration(
    experimentation_name="combined_arms_commander_self_play_ppo",
    environment_name="beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment",
    environment_configuration={
        "configuration": ReinforcementLearningEnvironmentConfiguration(
            game=GameConfiguration(scenario=combined_arms_commander_arena, controllers=[Agent(), Agent()]),
            observation=ObservationConfiguration(maximum_weapons_per_unit=1, version=2),
            action=ActionConfiguration(),
            rewards=RewardWeights(damage_dealt=0.0, damage_received=0.0, kill=0.0, victory=1.0, defeat=-1.0),
        )
    },
)

# The reinforcement learning part, under a short name.
configuration = combined_arms_commander_self_play_ppo.reinforcement_learning_configuration

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
    # The queries of each category, made from the observing unit's token (since version_1 of this training).
    "number_queries_per_category": 1,
    "category_query_hidden_sizes": [64],
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
combined_arms_commander_self_play_ppo.ray_object_store_memory = 3 * 1024**3

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
# Complete episodes: a dead unit's last action waits for the end of the game, which pays it the team's outcome, and
# MultiAgentEpisode.cut() does not hand waiting actions over (cut by a batch, the unit would lose that outcome). In agent
# steps: 12288, plus up to a game per env runner (1839 agent steps at most over 115 games, 2026-11-05), under the 20387
# that ran the learner out of GPU memory (8 GB) on 2026-10-25, when the count was in env steps.
configuration.count_steps_by = "agent_steps"
configuration.batch_mode = "complete_episodes"
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

# ── Checkpoints: all of them since version_1 (11 MB each; version_0 kept the 10 latest) ─────────
# Every 5 iterations: a restart for the RAM (start_training.sh) loses at most 5.
configuration.checkpoint_frequency = 5
# Each checkpoint with its policy exported to ONNX, playable whatever the code becomes (Checkpoint).
configuration.export_policies_at_checkpoint = True
configuration.number_checkpoint_to_keep = None
configuration.checkpoint_score_attribute = None
# display_episodes.py shows the latest checkpoint (it needs a metric to choose one).
combined_arms_commander_self_play_ppo.display_episodes_configuration.checkpoint_score_attribute = "training_iteration"
combined_arms_commander_self_play_ppo.display_episodes_configuration.checkpoint_score_order = "max"
