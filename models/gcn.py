import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class GCNRatingPrediction(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GCNRatingPrediction, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # Add a fully connected layer for rating prediction
        self.fc = torch.nn.Linear(hidden_dim * 2, 1)  # hidden_dim * 2 to account for concatenated embeddings

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # Use edge_index to select node features for source and destination
        edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)

        # Predict the rating from edge features
        ratings = self.fc(edge_features)
        ratings = torch.sigmoid(ratings) * 4 + 1 # compress ratings into range of (1, 5)
        return ratings.squeeze()  # Remove extra dimensions'
