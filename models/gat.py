import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class GATRatingPrediction(torch.nn.Module):
    def __init__(self, num_features, hidden_dim):
        super(GATRatingPrediction, self).__init__()
        # Multiple attention heads in the first layer
        self.conv1 = GATConv(num_features, hidden_dim, heads=4, concat=True, dropout=0.6)
        # Single attention head in the second layer, may reduce dimensionality
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=0.6)
        # Add a linear layer to predict a single rating from the concatenated features of two nodes
        self.predictor = torch.nn.Linear(2 * hidden_dim, 1)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        # Use edge_index to select node features for source and destination
        edge_features = torch.cat((x[edge_index[0]], x[edge_index[1]]), dim=1)

        # Predict the rating from edge features
        ratings = self.predictor(edge_features)
        return ratings.squeeze()  # Remove extra dimensions
