# Beyond_All_Reason_Reinforcement_Learning

Agents trained on Beyond All Reason: the bridge between the environment and the user's reinforcement learning library.
Created on 2026-10-22 from code that was in Maxalaar_Reinforcement_Learning and in the environment repository.

## Conventions
- Discuss with the user in French. Names in full, no acronyms in file, folder and identifier names.
- Conda environment `Beyond_All_Reason_Reinforcement_Learning` (/media/SSD_1To/conda/envs/; cloned from
  Maxalaar_Reinforcement_Learning's on 2026-10-22), where trainings run: this package, the environment and the library
  installed with `pip install -e`; conda variable `TORCH_DISABLE_NATIVE_JIT=1` (NixOS, see the library's CLAUDE.md).
  The library's own environment no longer has the Beyond All Reason packages (the user's rule: it knows no game).
- Scripts run from this repository root (experiments/ is relative to the working directory); trainings use the
  scripts of Maxalaar_Reinforcement_Learning (`python ../Maxalaar_Reinforcement_Learning/scripts/...`).
- Configuration files define a variable named like the file.

## Repositories (decided with the user, 2026-10-22)
- Three repositories, one direction of dependency:
  - Beyond_All_Reason_Environment (/media/SSD_1To/Programing_Projects/Python/Beyond_All_Reason_Environment): the game
    only (scenarios, environment, engine); no reinforcement learning library;
  - Maxalaar_Reinforcement_Learning (/media/SSD_1To/Programing_Projects/Python/Maxalaar_Reinforcement_Learning): the
    generic library; knows no game;
  - this one: depends on both. Game code goes to the environment, generic code to the library, the rest here.
- The environment is named by class path (`environment_name` =
  "beyond_all_reason_reinforcement_learning.ray_multi_agent_environment.RayMultiAgentEnvironment"), registered by the
  library (register_environments.register_environment).

## State of the experiments (2026-10-23)
- pawn_curriculum (stages numbered from 1): 1v1 and 5v5 against Passive reach 90 % victories in ~30 min.
  5v5 against HoldAndFire got stuck in avoiding the fight (draws 0, defeats -1); 5 against 1 to 5 AttackNearest
  (version_1, 2026-10-22) plateaued at ~80 % for 900 iterations (100 % against 1-2, 35 % against 5: the Pawn fight
  leaves little beyond numbers), with the entropy drifting up (4.8 -> 6.7) at coefficient 0.005.
  Decided with the user (2026-10-23): stage 3 is combined arms (3 Shellshockers, 5 Stouts, 10 Pawns against 3 to 6
  Stouts and 15 to 30 AttackNearest Pawns, until 80 %, entropy 0.001); the Pawn-only AttackNearest stages left the
  curriculum. Against Pawns only (version_2), the west Stouts won 100 % at once; with east Stouts the stage 2 policy
  starts at 50 %, a scripted attack of the nearest enemy at 70 %.
- Stopping criteria use env_runners/victory_rate_full_window (0 until 100 episodes per env runner): on victory_rate,
  stages 2 and 3 of version_2 stopped after 15 to 30 episodes.
  A Shellshocker alone was judged uninteresting by the user (slow shells, hard to use).
- No unit type one-hot in the observation (user's choice, 2026-10-23): a unit is known by its chassis and weapons,
  every unit uses the same policy, the observation has the same size in every scenario. Checkpoints made before
  (42 inputs to the entity encoders, 41 now) do not load anymore.
- The target point was drawn from RLlib's default for a Box space, a Gaussian clipped to the map by the environment:
  stage 3 of version_3 drifted to a mean of (-4.4, -4.3), standard deviation ~5 (97 % of the draws off the map, every
  unit sent to the corner (0, 0)); nothing holds the spread once off the map, hence the entropy drifts of every run.
  Since 2026-10-25 the target is a Gaussian squashed into the map (BeyondAllReasonModule, _BoundedTarget); checkpoints
  made before have the same shapes but another meaning: do not resume them.
- The user does not see the attack orders' aim as a problem: "fight" (attack-move) does the offensive moves.
- combined_arms_self_play_ppo (version_2, 848 iterations, 2026-10-25): the entropy fell steadily with the bounded
  target, but the draws rose to 25-38 % for hours (episodes near the time limit). Decided with the user: the commander
  arena (combined_arms_commander_arena: a commander per side, spawn anywhere in the own half, the game ends when a
  commander dies) for combined_arms_commander_self_play_ppo; 20 scripted games gave no draw, ~26 decisions per game.
  The commander keeps "fight" (the user's choice): under it, a builder attacks enemies within ~450 elmos first, and
  repairs or reclaims only when none is near (CBuilderCAI::ExecuteFight; idle, it does nothing on its own). Making it
  a pure fighter would need its unit definition changed in the game extension (canRepair, canReclaim are static). units.armada.commander is armcom; the old
  units...t1.commander was the decoy (armdecom), now decoy_commander.
- RAM (31 GB, no swap), measured on combined_arms_self_play_ppo (2 env runners, 36 agents, 2026-10-25): the training
  takes ~19 GB from its first minute (engines 8.5, learner 3.9, Ray object store 1.9, env runners 1.6, algorithm 1.3),
  then +2 GB/hour spread over engines (+0.27 each, all anonymous heap), object store (+0.5), learner (+0.5) and env
  runners (+0.2 each). The desktop applications take 7-10 GB (a Firefox tab 4.25 GB): with them, the machine starts at
  ~88 %, which put the supervisor in a restart loop. Ray's object store is now capped (ray_object_store_memory, 3 GB).
- Batch size: counted in env steps with complete episodes, it varied from 10280 to 20387 agent steps (36 agents, dying,
  episodes getting longer) until the learner ran out of GPU memory (8 GB). count_steps_by "agent_steps" with
  batch_mode "truncate_episodes" gives a fixed size (to a step of every agent per env runner).
- The training scripts exit with 0 even when the Tune trial errored: start_training.sh looks for "Number of
  errored trials" in the log to resume.
- Team outcome (decided with the user, 2026-11-05): a whole team wins (+1) or loses (-1) together, its dead units too.
  Before, a unit dead before the end got 0: in self-play at iteration 1270, 55 % of the unit steps never saw the outcome,
  and a unit of a losing side was better off dead. Now a dead unit has no observation but is not done: RLlib keeps its
  last action waiting and completes it at the end of the game, so no step of a dead unit enters the batches.
  MultiAgentEpisode.cut() drops waiting actions: batch_mode must be "complete_episodes" (a test checks it). Checkpoints
  made before load, but their critics learned the old returns.
- combined_arms_commander_rear_spawn_self_play_ppo version_0, old reward (evaluated 2026-11-05 from the rear): iteration
  1270 won 75.8 % of 817 games against AttackNearest, iteration 550 74.7 % of 427: no measurable gain against the script,
  though 1270 beats 550 (58 % / 36 %). RAM, measured by process (2026-11-06):
  - Evaluation: EnvRunner.sample(num_episodes=...) keeps its finished episodes until it returns, ~65 kB per agent step
    for the terrain alone: a runner grew to 17 GB, and Ray's memory monitor killed workers at 95 %. It kills the most
    recently scheduled first, so 19 fresh replacements of 0.4 GB died before the runner of 17 GB, and "225 episodes" were
    reported for ~817 played. The evaluation scripts now play rounds of 20 episodes (evaluation_rounds.py): 1.1 GB.
  - Training: no explosion, 85 % and at most 87.6 % over 10 minutes: engines 8.5 GB, desktop 7.6, learner 4.9, env runners
    2, driver 1.7. No worker was lost in the 28 restarts of version_0: spikes crossed the supervisor's 88 %, now 91 %.
- combined_arms_commander_rear_spawn_self_play_ppo version_1 (decided with the user, 2026-11-06): the same scenario as
  version_0, from scratch, with the team outcome. Checked before starting, on a run deleted after: episode returns exactly
  0 in self-play, batches of 13200 to 14400 agent steps, a resume by the supervisor from its checkpoint, 7 GB for the 2
  engines. To start: experiments/combined_arms_commander_rear_spawn_self_play_ppo/start_training.sh.

