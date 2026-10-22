# Beyond All Reason reinforcement learning

Agents trained on the real-time strategy game [Beyond All Reason](https://www.beyondallreason.info/), with
[Maxalaar_Reinforcement_Learning](https://github.com/Maxalaar/Maxalaar_Reinforcement_Learning) (a layer over Ray RLlib)
and the environment of [Beyond_All_Reason_Environment](https://github.com/Maxalaar/Beyond_All_Reason_Environment).

| Repository | Role |
|---|---|
| Beyond_All_Reason_Environment | the game as an environment, without any reinforcement learning library |
| Maxalaar_Reinforcement_Learning | generic training, curriculum, evaluation and export; knows no game |
| **this one** | what joins them: RLlib adapter, network, victory metric, experiments |

## Contents

- `src/beyond_all_reason_reinforcement_learning/`
  - `ray_multi_agent_environment.py`: `RayMultiAgentEnvironment`, the RLlib `MultiAgentEnv` of the game (one agent per
    unit; a `render_mode` opens the game window);
  - `architecture/`: `BeyondAllReasonModule`, attention over the entities (each unit and its weapons, then each
    category, then the whole situation), and its centralized critic, which sees every alliance;
  - `victory_callback.py`: `VictoryCallback`, victory and defeat rates over the last 100 episodes of each env runner.
- `configurations/`: the experiments (`environment_name` is the class path of `RayMultiAgentEnvironment`):
  - `one_versus_one_ppo.py`: 1 agent against 1 passive Pawn;
  - `five_versus_five_passive_ppo.py`: 5 agents against 5 passive Pawns;
  - `five_versus_two_hold_and_fire_ppo.py`: 5 agents against 2 Pawns that shoot back;
  - `five_versus_five_hold_and_fire_ppo.py`: 5 agents against 5 Pawns that shoot back;
  - `pawn_curriculum.py`: the four, each until 90 % victories, every stage starting from the weights of the previous.
- `experiments/`: the results (checkpoints, logs), not versioned.

## Installation

The three repositories side by side, in this project's conda environment (Python 3.13):

```bash
conda create -n Beyond_All_Reason_Reinforcement_Learning python=3.13
conda activate Beyond_All_Reason_Reinforcement_Learning
git clone https://github.com/Maxalaar/Beyond_All_Reason_Environment.git
git clone https://github.com/Maxalaar/Maxalaar_Reinforcement_Learning.git
git clone https://github.com/Maxalaar/Beyond_All_Reason_Reinforcement_Learning.git
pip install -e Beyond_All_Reason_Environment -e Maxalaar_Reinforcement_Learning -e Beyond_All_Reason_Reinforcement_Learning
python Beyond_All_Reason_Environment/scripts/download_game_data.py    # engine, game and maps
```

## Usage

From this repository root, with the scripts of Maxalaar_Reinforcement_Learning:

```bash
# The curriculum (1v1, then 5v5): the same command resumes it where it stopped
python ../Maxalaar_Reinforcement_Learning/scripts/curriculum_training.py \
    --curriculum_configuration_file configurations/pawn_curriculum.py

# One experiment
python ../Maxalaar_Reinforcement_Learning/scripts/reinforcement_learning_training.py \
    --experimentation_configuration_file configurations/one_versus_one_ppo.py

# Watch a trained policy in the game window
python ../Maxalaar_Reinforcement_Learning/scripts/display_episodes.py \
    --experimentation_configuration_file configurations/five_versus_five_passive_ppo.py \
    --reinforcement_learning_path experiments/pawn_curriculum/version_0/stage_1_five_versus_five_passive_ppo/reinforcement_learning/base/version_0

python scripts/try_ray_multi_agent_environment.py   # a few random episodes through the RLlib adapter
pytest                                              # tests, without the engine
```
