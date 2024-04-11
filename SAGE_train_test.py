import torch
from models.SAGE import SAGERecommendation
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch.nn.functional as F
from tqdm import tqdm

ratings_data = pd.read_csv("datasets/encoded/ratings.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create mappings for user IDs and movie IDs to consecutive integers

# Unique nodes
user_ids = ratings_data['UID'].unique()
movie_ids = ratings_data['MID'].unique()

# Node mapping to indices
user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
movie_mapping = {mid: idx + len(user_ids) for idx, mid in enumerate(movie_ids)}

# Create edge index
edge_index = torch.tensor([
    [user_mapping[uid], movie_mapping[mid]]
    for uid, mid in zip(ratings_data['UID'], ratings_data['MID'])
], dtype=torch.long).t().contiguous()

# Ratings as edge attributes
edge_attr = torch.tensor(ratings_data['rating'], dtype=torch.float).unsqueeze(1)

# Random node features
num_users = len(user_ids)
num_movies = len(movie_ids)
num_features = 10  # Random choice, can be tuned
node_features = torch.randn((num_users + num_movies, num_features), dtype=torch.float)

# Graph data object
data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

# Model initialization
model = SAGERecommendation(num_features=10, hidden_dim=32)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Training loop
model.train()
for epoch in range(120):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.edge_attr)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')
