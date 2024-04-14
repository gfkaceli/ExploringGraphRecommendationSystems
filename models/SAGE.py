import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


class SAGERecommendation(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(SAGERecommendation, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, hidden_dim)
        # Add a fully connected layer for rating prediction
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 to account for concatenated embeddings

    def forward(self, x, edge_index):

        # Forward pass through GraphSAGE layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        # Using edge_index to pick relevant node features for ratings prediction
        edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)
        ratings = self.fc(edge_features).squeeze()
        ratings = torch.sigmoid(ratings) * 4 + 1

        return ratings
