from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .imputation import ImputationStrategy
from .scaling import ScalingStrategy


def load_json_data(file_path: Path) -> dict[str, float]:
    with open(file_path) as f:
        return json.load(f)


def preprocess_data(
    data_files: list[Path],
    imputation_strategy: ImputationStrategy,
    scaling_strategy: ScalingStrategy,
) -> np.ndarray:
    all_keys: set[str] = set().union(*(load_json_data(file).keys() for file in data_files))

    # Create a dictionary to store data for each key across all files
    data_by_key: dict[str, list[float]] = {key: [] for key in all_keys}
    print("Loading data...")
    for file in tqdm(data_files):
        file_data = load_json_data(file)
        for key in all_keys:
            data_by_key[key].append(file_data.get(key, np.nan))

    # Impute and scale each key's data separately
    print("Imputing and scaling fields...")
    for key in tqdm(all_keys):
        data_array = np.array(data_by_key[key])
        data_array = imputation_strategy.impute(data_array)
        data_by_key[key] = scaling_strategy.scale(data_array.reshape(-1, 1)).flatten()

    # Construct the final data matrix
    data_matrix = np.array(
        [[data_by_key[key][i] for key in all_keys] for i in range(len(data_files))]
    )

    return data_matrix


def compute_distances(data_matrix: np.ndarray, target_index: int, ord: float) -> np.ndarray:
    target_row = data_matrix[target_index]
    return np.linalg.norm(data_matrix - target_row, ord=ord, axis=1)


@dataclass
class Args:
    target_file: Path
    data_dir: Path
    n: int
    imputation: str
    scaling: str
    ord: float


def float_or_none(s: str) -> float | None:
    try:
        return float(s)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Find the closest and furthest 'JSON neighbors'.")
    parser.add_argument("target_file", type=Path, help="The JSON file to compare against")
    parser.add_argument("data_dir", type=Path, help="Directory containing JSON data files")
    parser.add_argument("-n", type=int, default=5, help="Number of closest/furthest files to print")
    parser.add_argument(
        "--imputation", choices=["median"], default="median", help="Imputation strategy"
    )
    parser.add_argument(
        "--scaling", choices=["standard"], default="standard", help="Scaling strategy"
    )
    parser.add_argument(
        "--ord",
        default=2,
        type=float,
        help="Order of the norm to use. Controls how the distance is computed. "
        "Is passed to numpy.linalg.norm.",
    )
    args = parser.parse_args(namespace=Args)

    data_files = list(args.data_dir.glob("*.json"))
    if args.target_file not in data_files:
        raise ValueError("Target file not found in the data directory")

    imputation_strategy = ImputationStrategy.from_string(args.imputation)
    scaling_strategy = ScalingStrategy.from_string(args.scaling)

    data_matrix = preprocess_data(data_files, imputation_strategy, scaling_strategy)
    target_index = data_files.index(args.target_file)
    distances = compute_distances(data_matrix, target_index, ord=args.ord)

    closest_indices = np.argsort(distances)[1 : args.n + 1]  # Exclude the target file itself
    furthest_indices = np.argsort(distances)[-args.n :][::-1]

    print("Closest files:")
    for index in closest_indices:
        print(f"- {data_files[index]} (distance: {distances[index]:.3f})")

    print("\nFurthest files:")
    for index in furthest_indices:
        print(f"- {data_files[index]} (distance: {distances[index]:.3f})")


if __name__ == "__main__":
    main()
