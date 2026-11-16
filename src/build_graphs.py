import numpy as np
import torch
from torch_geometric.data import Data

from load_dataset import load_dataset


def make_graph(matrix, mask, label):
    matrix = matrix.copy()
    np.fill_diagonal(mask, False)

    source, target = np.where(np.triu(mask, k=1))

    edges = np.concatenate(
        [
            np.stack([source, target]),
            np.stack([target, source])
        ],
        axis=1
    )

    weights = np.concatenate(
        [matrix[source, target], matrix[source, target]]
    )

    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)

    x = torch.eye(matrix.shape[0], dtype=torch.float32)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        y=torch.tensor([label], dtype=torch.long)
    )


def construct_graphs(matrix, label):
    regular = make_graph(
        matrix, matrix != 0, label
    )

    positive = make_graph(
        matrix, matrix > 0, label
    )

    negative = make_graph(
        matrix, matrix < 0, label
    )

    high_positive = make_graph(
        matrix, matrix > 0.5, label
    )

    low_positive = make_graph(
        matrix, (matrix > 0) & (matrix <= 0.5), label
    )

    high_negative = make_graph(
        matrix, matrix < -0.5, label
    )

    low_negative = make_graph(
        matrix, (matrix < 0) & (matrix >= -0.5), label
    )

    return {
        "regular": [regular],
        "two_graphs": [positive, negative],
        "four_graphs": [
            high_positive,
            low_positive,
            high_negative,
            low_negative
        ]
    }


if __name__ == "__main__":
    matrices, labels, filenames = load_dataset()

    example = construct_graphs(matrices[0], labels[0])

    print("Subject:", filenames[0])
    print("Diagnosis:", "AD" if labels[0] == 1 else "NC")

    for representation, graphs in example.items():
        print("\n", representation)

        for i, graph in enumerate(graphs):
            print(
                "Graph", i + 1,
                "| Nodes:", graph.num_nodes,
                "| Directed edges:", graph.num_edges
            )
