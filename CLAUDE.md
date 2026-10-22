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
  Decided with the user (2026-10-23): stage 3 is combined arms (3 Shellshockers, 5 Stouts, 10 Pawns against 15 to 30
  AttackNearest Pawns, until 80 %, entropy 0.001); the Pawn-only AttackNearest stages left the curriculum.
  A Shellshocker alone was judged uninteresting by the user (slow shells, hard to use).
- No unit type one-hot in the observation (user's choice, 2026-10-23): a unit is known by its chassis and weapons,
  every unit uses the same policy, the observation has the same size in every scenario. Checkpoints made before
  (42 inputs to the entity encoders, 41 now) do not load anymore.
- The agents do not aim yet: in display_episodes (explore=False) their attack orders fall where no enemy is within
  attack_snap_radius (60 elmos), so they wait, and the engine's fire at will engages when enemies come in range.
  The user chose not to change the targeting for now (bigger snap radius or target relative to the unit discussed).
- RAM (31 GB, no swap): 3 env runners and the other applications leave little margin; each engine takes ~4.3 GB
  and grows ~0.3 GB/hour. To investigate in the environment (what the headless engine loads).
