import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch.optim import Adam
import torch.nn.functional as F
from models.gcn import GCNRatingPrediction  # Ensure this path is correct

# Load datasets
users_df = pd.read_csv('datasets/encoded/0.01users.csv')
movies_df = pd.read_csv('datasets/encoded/movies.csv')
ratings_df = pd.read_csv('datasets/encoded/ratings.csv')

user_cnt = ratings_df['UID'].nunique()
movie_cnt = ratings_df['MID'].nunique()

# Prepare user and movie mappings
user_ids = users_df['UID'].unique()
movie_ids = movies_df['MID'].unique()
user_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
movie_mapping = {movie_id: i + len(user_ids) for i, movie_id in enumerate(movie_ids)}

# Prepare edges and edge attributes
edges = torch.tensor([(user_mapping[row['UID']], movie_mapping[row['MID']]) for index, row in ratings_df.iterrows()],
                     dtype=torch.long).t().contiguous()
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
model = GCNRatingPrediction(num_features=node_features.shape[1], hidden_dim=64)
optimizer = Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training loop
losses = []
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out, train_data.edge_attr)  # Ensure dimensions match
    loss.backward()
    # Record the loss
    losses.append(loss.item())
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# Smooth the losses
smoothed_losses = smooth_curve(losses)

# Plot the smoothed curve
plt.figure(figsize=(10, 5))
plt.plot(smoothed_losses, label='Smoothed Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Smoothed Training Loss Curve')
plt.legend()
plt.show()

# Save the model
torch.save(model.state_dict(), 'models/gcn_rating_prediction.pth')

# Evaluate the model
model.eval()
with torch.no_grad():
    out = model(test_data.x, test_data.edge_index)
    loss = criterion(out.squeeze(), test_data.edge_attr)
    print(f'Test Loss: {loss.item()}')

    # Calculate MSE and RMSE
    mse = F.mse_loss(out.squeeze(), test_data.edge_attr, reduction='mean')
    rmse = torch.sqrt(mse)
    print(f'MSE: {mse.item()}, RMSE: {rmse.item()}')