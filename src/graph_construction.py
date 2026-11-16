
import numpy as np


def construct_four_graphs(correlation_matrix):
    """
    This function takes one subject's correlation matrix
    and separates it into four graphs based on the
    strength and sign of the correlations.

    Each graph has the same brain regions (nodes),
    but different connections (edges).
    """

    matrix = np.asarray(correlation_matrix, dtype=float)

    # First, make sure the matrix is valid.
    # It should be square because we are comparing
    # the same brain regions with each other.
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Correlation matrix must be square.")

    # Check for missing or invalid values.
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains NaN or infinite values.")

    # Correlation matrices should be symmetric,
    # meaning the correlation from ROI A to ROI B
    # should equal the correlation from ROI B to ROI A.
    if not np.allclose(matrix, matrix.T, atol=1e-5):
        raise ValueError("Correlation matrix must be symmetric.")

    # Correlation values should always be between -1 and 1.
    if np.any((matrix < -1) | (matrix > 1)):
        raise ValueError("Correlations must be between -1 and 1.")

    num_nodes = matrix.shape[0]

    # We don't want a brain region connecting to itself,
    # so we exclude the diagonal values from our graphs.
    valid_edges = ~np.eye(num_nodes, dtype=bool)

    # This is the main part of our project.
    # We split the connections into four categories
    # so the GNN can learn from each type separately
    masks = {
        "high_positive": matrix > 0.5,
        "low_positive": (matrix > 0) & (matrix <= 0.5),
        "high_negative": matrix < -0.5,
        "low_negative": (matrix >= -0.5) & (matrix < 0),
    }

    # Store all four graphs in one dictionary.
    graphs = {}

    for graph_name, mask in masks.items():

        # Remove self-connections from the current graph.
        mask = mask & valid_edges

        # Find which brain regions are connected
        # based on the correlation category.
        source_nodes, target_nodes = np.where(mask)

        # Store the connections as pairs of node indices.
        # This format will also be useful when we
        # convert the graphs to PyTorch Geometric.
        edge_index = np.vstack(
            (source_nodes, target_nodes)
        ).astype(np.int64)

        # Keep the original correlation values as weights.
        # This lets us preserve how strong each connection is,
        # instead of only recording whether an edge exists.
        edge_weight = matrix[
            source_nodes, target_nodes
        ].astype(np.float32)

        # Save the connections, weights, and number of ROIs
        # for the current graph.
        graphs[graph_name] = {
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "num_nodes": num_nodes,
        }

    # Return all four graphs for this subject.
    return graphs


# Testing the graph construction before using real data.
# We are using a small example with five brain regions
# to make sure the splitting method works correctl
if __name__ == "__main__":

    # Example correlation matrix (not actual OASIS-3 data).
    # Positive values represent positive correlations,
    # while negative values represent negative correlations.
    example_matrix = np.array([
        [1.0,  0.8,  0.3, -0.7, -0.2],
        [0.8,  1.0, -0.4,  0.6,  0.1],
        [0.3, -0.4,  1.0, -0.8,  0.9],
        [-0.7, 0.6, -0.8,  1.0, -0.3],
        [-0.2, 0.1,  0.9, -0.3,  1.0],
    ])

    # Call our function to create the four graphs.
    graphs = construct_four_graphs(example_matrix)

    print("Number of brain ROIs:", example_matrix.shape[0])

    # Print each graph so we can check that the
    # connections were separated correctly.
    for name, graph in graphs.items():
        print(f"\nGraph: {name}")
        print(
            "Number of directed edges:",
            graph["edge_index"].shape[1]
        )
        print("Edge indices:")
        print(graph["edge_index"])
        print("Edge weights:")
        print(graph["edge_weight"])
