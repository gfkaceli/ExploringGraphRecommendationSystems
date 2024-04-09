import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn.to_hetero_transformer import to_hetero
from torch_geometric.nn import SAGEConv


class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, conv):
        super().__init__()
        self.conv1 = conv((-1, -1), hidden_channels)
        self.conv2 = conv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, conv=SAGEConv):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels, conv)
        self.encoder = to_hetero(self.encoder, Data.metadata(), aggr='sum')
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict, edge_label_index)
