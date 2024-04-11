import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data


class GATRatingPrediction(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GATRatingPrediction, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        # Add a fully connected layer for rating prediction
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 to account for concatenated embeddings

    def forward(self, data, user_indices, movie_indices):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Gather specific user and movie embeddings
        user_embeddings = x[user_indices]
        movie_embeddings = x[movie_indices]

        # Combine user and movie embeddings (e.g., concatenation)
        combined_embeddings = torch.cat((user_embeddings, movie_embeddings), dim=1)

        # Predict ratings
        ratings_pred = self.fc(combined_embeddings)
        return ratings_pred
