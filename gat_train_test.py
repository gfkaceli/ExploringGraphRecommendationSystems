import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
from models.gat import GATRatingPrediction  # Ensure this path is correct

# Load datasets
users_df = pd.read_csv('datasets/encoded/0.01users.csv')
movies_df = pd.read_csv('datasets/encoded/movies.csv')
ratings_df = pd.read_csv('datasets/encoded/ratings.csv')

# Prepare user and movie mappings
user_ids = users_df['UID'].unique()
movie_ids = movies_df['MID'].unique()
user_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
movie_mapping = {movie_id: i + len(user_ids) for i, movie_id in enumerate(movie_ids)}

# Prepare edges and edge attributes
edges = torch.tensor([(user_mapping[row['UID']], movie_mapping[row['MID']]) for index, row in ratings_df.iterrows()], dtype=torch.long).t().contiguous()
edge_attrs = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

# Node features
user_features = torch.tensor(users_df.drop('UID', axis=1).values, dtype=torch.float)
movie_features = torch.tensor(movies_df.drop('MID', axis=1).values, dtype=torch.float)
if user_features.shape[1] != movie_features.shape[1]:
    # Pad the smaller one
    padding = torch.zeros(abs(user_features.shape[1] - movie_features.shape[1]))
    if user_features.shape[1] < movie_features.shape[1]:
        user_features = torch.cat([user_features, padding.expand(user_features.size(0), -1)], dim=1)
    else:
        movie_features = torch.cat([movie_features, padding.expand(movie_features.size(0), -1)], dim=1)
node_features = torch.cat([user_features, movie_features], dim=0)

# Data split
train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
train_edges = torch.tensor([(user_mapping[row['UID']], movie_mapping[row['MID']]) for index, row in train_df.iterrows()],
                           dtype=torch.long).t().contiguous()
train_edge_attrs = torch.tensor(train_df['rating'].values, dtype=torch.float)
test_edges = torch.tensor([(user_mapping[row['UID']], movie_mapping[row['MID']]) for index, row in test_df.iterrows()],
                          dtype=torch.long).t().contiguous()
test_edge_attrs = torch.tensor(test_df['rating'].values, dtype=torch.float)

# Graph data
train_data = Data(x=node_features, edge_index=train_edges, edge_attr=train_edge_attrs)
test_data = Data(x=node_features, edge_index=test_edges, edge_attr=test_edge_attrs)

# Model and optimization
model = GATRatingPrediction(num_features=node_features.shape[1], hidden_dim=64)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Training loop
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out, train_data.edge_attr)  # Ensure dimensions match
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'models/gat_rating_prediction.pth')

# Evaluate the model
model.eval()
with torch.no_grad():
    out = model(test_data.x, test_data.edge_index)
    loss = criterion(out, test_data.edge_attr)
    print(f'Test Loss: {loss.item()}')

    # Calculate MSE and RMSE
    mse = F.mse_loss(out, test_data.edge_attr)
    rmse = torch.sqrt(mse)
    print(f'MSE: {mse.item()}, RMSE: {rmse.item()}')

    # Convert predictions to binary
    predicted_labels = (out > 3.5).float()  # Assuming 3.5 as the threshold

    # True labels are your actual ratings turned into binary (1 if rating > 3.5 else 0)
    true_labels = (test_data.edge_attr > 3.5).float()

    # Calculate precision and recall
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
