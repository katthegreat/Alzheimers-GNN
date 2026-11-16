from pathlib import Path

import numpy as np
from scipy.io import loadmat


DATA_DIR = Path("data/raw/MNMI_AD/DATA/FC")


def load_fc_matrix(file_path):
    data = loadmat(file_path)

    matrices = [
        value
        for key, value in data.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and value.ndim == 2
        and value.shape[0] == value.shape[1]
        and np.issubdtype(value.dtype, np.number)
    ]

    if len(matrices) != 1:
        raise ValueError(
            f"Expected one square matrix in {file_path}, found {len(matrices)}"
        )

    matrix = np.asarray(matrices[0], dtype=np.float32)

    if not np.isfinite(matrix).all():
        raise ValueError(f"Invalid values in {file_path}")

    if matrix.shape != (46, 46):
        raise ValueError(
            f"Unexpected matrix shape {matrix.shape} in {file_path}"
        )

    return matrix


def load_dataset():
    matrices = []
    labels = []
    filenames = []

    for file_path in sorted(DATA_DIR.glob("FC_*.mat")):
        name = file_path.name

        if name.startswith("FC_AD"):
            label = 1
        elif name.startswith("FC_NC"):
            label = 0
        else:
            continue

        matrix = load_fc_matrix(file_path)

        matrices.append(matrix)
        labels.append(label)
        filenames.append(name)

    if not matrices:
        raise RuntimeError("No AD or NC connectivity matrices found")

    return (
        np.stack(matrices),
        np.asarray(labels, dtype=np.int64),
        filenames,
    )


if __name__ == "__main__":
    X, y, filenames = load_dataset()

    print("Total subjects:", len(y))
    print("Alzheimer's subjects:", np.sum(y == 1))
    print("Normal control subjects:", np.sum(y == 0))
    print("Dataset shape:", X.shape)
    print("Label shape:", y.shape)
    print("First five files:", filenames[:5])
    print("Minimum correlation:", X.min())
    print("Maximum correlation:", X.max())
