
import numpy as np
from load_dataset import load_dataset
from build_graphs import construct_graphs

# loading the matrices and labels
matrices, labels, filenames = load_dataset()

# keeping track of how many graphs have no edges
empty_counts = {
    "regular": 0,
    "positive": 0,
    "negative": 0,
    "high_positive": 0,
    "low_positive": 0,
    "high_negative": 0,
    "low_negative": 0,
}

asymmetric = []
invalid_ranges = []
edge_mismatches = []

# going through each subject to check their graphs
for matrix, label, filename in zip(matrices, labels, filenames):

    # checking if the matrix is symmetric
    if not np.allclose(matrix, matrix.T, atol=1e-5):
        asymmetric.append(filename)

    # making sure the correlations are between -1 and 1
    if matrix.min() < -1.00001 or matrix.max() > 1.00001:
        invalid_ranges.append(filename)

    # creating the different graph types
    graphs = construct_graphs(matrix, int(label))

    regular = graphs["regular"][0]
    positive, negative = graphs["two_graphs"]
    high_pos, low_pos, high_neg, low_neg = graphs["four_graphs"]

    all_graphs = {
        "regular": regular,
        "positive": positive,
        "negative": negative,
        "high_positive": high_pos,
        "low_positive": low_pos,
        "high_negative": high_neg,
        "low_negative": low_neg,
    }

    # checking if any of the graphs have zero edges
    for name, graph in all_graphs.items():
        if graph.num_edges == 0:
            empty_counts[name] += 1

    # checking if the positive and negative edges add up to the regular graph
    if regular.num_edges != positive.num_edges + negative.num_edges:
        edge_mismatches.append(filename)

    # checking if the high and low positive edges add up
    if positive.num_edges != high_pos.num_edges + low_pos.num_edges:
        edge_mismatches.append(filename)

    # doing the same thing for the negative edges
    if negative.num_edges != high_neg.num_edges + low_neg.num_edges:
        edge_mismatches.append(filename)

# printing the results from checking all the subjects
print("Subjects checked:", len(filenames))
print("Asymmetric matrices:", len(asymmetric))
print("Invalid correlation ranges:", len(invalid_ranges))
print("Edge partition mismatches:", len(set(edge_mismatches)))

# seeing how many empty graphs we have for each type
print("\nEmpty graph counts:")
for name, count in empty_counts.items():
    print(f"{name}: {count}")

# checking how many ad and normal control subjects we have
print("\nDataset balance:")
print("AD:", int(np.sum(labels == 1)))
print("NC:", int(np.sum(labels == 0)))
