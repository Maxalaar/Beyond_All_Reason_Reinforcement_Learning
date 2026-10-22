"""Pawn curriculum on the 1v1 arena (flat 1024×1024 elmos): each stage until 90 % victories, then the next one from
its weights.

    1. one_versus_one_ppo.py                           1 agent against 1 Passive Pawn
    2. five_versus_five_passive_ppo.py                 5 agents against 5 Passive Pawns
    3. five_versus_one_to_five_attack_nearest_ppo.py   5 agents against 1 to 5 AttackNearest Pawns (drawn each episode)
    4. five_versus_five_attack_nearest_ppo.py          5 agents against 5 AttackNearest Pawns

Against still opponents that shoot back (HoldAndFire), the agents learned to avoid the fight (draws): from step 3 on,
the opponents come to fight.

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
        "configurations/five_versus_one_to_five_attack_nearest_ppo.py",
        "configurations/five_versus_five_attack_nearest_ppo.py",
    ],
)
