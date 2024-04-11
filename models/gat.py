import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATRatingPrediction(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GATRatingPrediction, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))

        user_features = x[edge_index[0]]
        movie_features = x[edge_index[1]]
        combined_features = torch.cat([user_features, movie_features], dim=1)

        return self.fc(combined_features)
