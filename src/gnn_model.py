import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


# using two gcn layers to get information from the brain graphs
class GraphEncoder(nn.Module):
    def __init__(self, input_dim=46, hidden_dim=32):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, graph):
        # getting the node features and connections
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight.abs()

        # making a batch if we are only using one graph
        batch = getattr(graph, "batch", None)
        if batch is None:
            batch = torch.zeros(
                x.size(0),
                dtype=torch.long,
                device=x.device
            )

        # passing the graph through the gcn layers
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)

        # combining the node features into one graph representation
        return global_mean_pool(x, batch)


# model for predicting alzheimers or normal control
class AlzheimerGNN(nn.Module):
    def __init__(self, representation="regular", hidden_dim=32):
        super().__init__()

        # number of graphs depends on which method we are testing
        graph_counts = {
            "regular": 1,
            "two_graphs": 2,
            "four_graphs": 4
        }

        if representation not in graph_counts:
            raise ValueError("Unknown graph representation")

        self.representation = representation
        self.num_graphs = graph_counts[representation]

        # using a separate encoder for each graph
        self.encoders = nn.ModuleList([
            GraphEncoder(hidden_dim=hidden_dim)
            for _ in range(self.num_graphs)
        ])

        # using the graph features to predict the class
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_graphs, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

    def forward(self, graphs):
        if len(graphs) != self.num_graphs:
            raise ValueError(
                f"Expected {self.num_graphs} graphs, got {len(graphs)}"
            )

        # getting the embeddings from each graph
        embeddings = [
            encoder(graph)
            for encoder, graph in zip(self.encoders, graphs)
        ]

        # putting all the graph embeddings together
        combined = torch.cat(embeddings, dim=1)

        return self.classifier(combined)


if __name__ == "__main__":
    from build_graphs import construct_graphs
    from load_dataset import load_dataset

    # loading the data to test the model
    matrices, labels, filenames = load_dataset()

    example = construct_graphs(
        matrices[0],
        int(labels[0])
    )

    # checking if all three graph methods work
    for representation, graphs in example.items():
        model = AlzheimerGNN(representation)
        output = model(graphs)
        print(
            representation,
            "| Output shape:",
            tuple(output.shape)
        )