"""Adapt the weights of a checkpoint to a scenario with other unit types.

The chassis features of an entity start with a one-hot of its unit type, over the scenario's unit types (sorted:
ScenarioConfiguration.unit_types); a scenario with another set of types changes the input size of the first layer of
each entity encoder (actor and centralized critic). This copies the checkpoint's modules with that layer remapped: each
type found in both lists keeps its column, a new type gets a zero column (it then looks like a unit without type, known
by its other chassis features), a type dropped loses its column. The other weights are copied as they are.

The output holds learner_group/learner/rl_module only, what initialization_checkpoint_path reads (weights only).

Example (from this repository root), the 5v5 Pawn weights for a scenario that adds the Shellshocker:
    python scripts/add_unit_types_to_checkpoint.py \
        --checkpoint_path <a five_versus_five_passive_ppo checkpoint> \
        --old_unit_types armpw --new_unit_types armart armpw \
        --output_path experiments/artillery_and_ten_versus_ten_to_twenty_attack_nearest_ppo/initialization_checkpoint
"""

import argparse
import pickle
import shutil
from pathlib import Path

import numpy as np

MODULES_PATH = Path("learner_group") / "learner" / "rl_module"
# First layer of each entity encoder: its input is chassis ∥ state, the chassis starting with the unit type one-hot.
CHASSIS_LAYER_SUFFIX = "entity_encoder.entity_projector.0.weight"


def remap_type_columns(weight: np.ndarray, old_unit_types: list[str], new_unit_types: list[str]) -> np.ndarray:
    """``weight`` (output, len(old_unit_types) + other inputs) -> (output, len(new_unit_types) + other inputs)."""
    number_old = len(old_unit_types)
    type_columns = np.zeros((weight.shape[0], len(new_unit_types)), dtype=weight.dtype)
    for new_index, unit_type in enumerate(new_unit_types):
        if unit_type in old_unit_types:
            type_columns[:, new_index] = weight[:, old_unit_types.index(unit_type)]
    return np.concatenate([type_columns, weight[:, number_old:]], axis=1)


def add_unit_types_to_checkpoint(
    checkpoint_path: Path, old_unit_types: list[str], new_unit_types: list[str], output_path: Path
) -> None:
    if old_unit_types != sorted(old_unit_types) or new_unit_types != sorted(new_unit_types):
        raise ValueError("unit types go in the order of the observation: sorted, as ScenarioConfiguration.unit_types")
    source = checkpoint_path / MODULES_PATH
    if not source.exists():
        raise FileNotFoundError(f"no modules in {source}")
    destination = output_path / MODULES_PATH
    if output_path.exists():
        raise FileExistsError(f"{output_path} exists already")
    shutil.copytree(source, destination)

    for state_path in sorted(destination.glob("*/module_state.pkl")):
        with open(state_path, "rb") as file:
            state = pickle.load(file)
        remapped = [key for key in state if key.endswith(CHASSIS_LAYER_SUFFIX)]
        if not remapped:
            raise ValueError(f"{state_path}: no layer ending with {CHASSIS_LAYER_SUFFIX}")
        for key in remapped:
            weight = np.asarray(state[key])
            state[key] = remap_type_columns(weight, old_unit_types, new_unit_types)
            print(f"{state_path.parent.name}: {key} {weight.shape} -> {state[key].shape}")
        with open(state_path, "wb") as file:
            pickle.dump(state, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--old_unit_types", nargs="+", required=True, help="the unit types of the checkpoint, sorted")
    parser.add_argument("--new_unit_types", nargs="+", required=True, help="the unit types of the new scenario, sorted")
    parser.add_argument("--output_path", type=Path, required=True)
    arguments = parser.parse_args()
    add_unit_types_to_checkpoint(
        arguments.checkpoint_path, arguments.old_unit_types, arguments.new_unit_types, arguments.output_path
    )
