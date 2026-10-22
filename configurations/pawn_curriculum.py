"""Pawn curriculum on the 1v1 arena (flat 1024×1024 elmos): each stage until 90 % victories, then the next one from
its weights.

    1. one_versus_one_ppo.py                           1 agent against 1 Passive Pawn
    2. five_versus_five_passive_ppo.py                 5 agents against 5 Passive Pawns
    3. combined_arms_attack_nearest_ppo.py             3 Shellshockers, 5 Stouts and 10 Pawns against 15 to 30
                                                       AttackNearest Pawns (drawn each episode), until 80 %

Every unit uses the same policy, known by its chassis and weapons (no unit type in the observation), so the weights
pass from the Pawns of steps 1 and 2 to the tanks and the artillery of step 3.
Against still opponents that shoot back (HoldAndFire), the agents learned to avoid the fight (draws); 5 Pawns against
1 to 5 AttackNearest Pawns plateaued at ~80 % (the Pawn fight leaves little beyond numbers): step 3 opposes
opponents that come to fight with an army whose formation matters.

Usage (from this repository root; the same command resumes the curriculum where it stopped, --new starts a new run):
    python ../Maxalaar_Reinforcement_Learning/scripts/curriculum_training.py \
        --curriculum_configuration_file configurations/pawn_curriculum.py
"""

from maxalaar_reinforcement_learning.configuration.curriculum_configuration import CurriculumConfiguration

pawn_curriculum = CurriculumConfiguration(
    curriculum_name="pawn_curriculum",
    stages=[
        "configurations/one_versus_one_ppo.py",
        "configurations/five_versus_five_passive_ppo.py",
        "configurations/combined_arms_attack_nearest_ppo.py",
    ],
)
