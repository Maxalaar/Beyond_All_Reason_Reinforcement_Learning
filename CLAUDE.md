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

## State of the experiments (2026-10-22)
- pawn_curriculum: 1v1 and 5v5 against Passive reached 90 % victories; 5v5 against HoldAndFire got stuck in avoiding
  the fight (draws are worth 0, defeats -1), its entropy diverged. Next idea agreed in principle: an intermediate
  stage, 5 agents against 2 HoldAndFire Pawns. Checkpoints made before 2026-10-22 do not load anymore (class paths
  changed with the move between repositories).
- RAM (31 GB, no swap): 3 env runners and the other applications leave little margin; each engine takes ~4.3 GB
  and grows ~0.3 GB/hour. To investigate in the environment (what the headless engine loads).
