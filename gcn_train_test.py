from models.gcn import GCNRatingPrediction
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import torch.nn.functional as F

ratings_df = pd.read_csv("datasets/encoded/ratings.csv")

# Encode user IDs and movie IDs to integer indices
user_encoder = LabelEncoder()
movie_encoder = LabelEncoder()

ratings_df['UID'] = user_encoder.fit_transform(ratings_df['UID'])
ratings_df['MID'] = movie_encoder.fit_transform(ratings_df['MID']) + ratings_df['UID'].max() + 1

# Prepare the graph data
num_users = ratings_df['UID'].max() + 1
num_movies = ratings_df['MID'].max() + 1
num_nodes = num_users + num_movies

# Dummy node features
x = torch.eye(num_nodes, num_nodes)

# Edge index and weight
edge_index = torch.tensor([ratings_df['UID'], ratings_df['MID'] + num_users], dtype=torch.long)
edge_weight = torch.tensor(ratings_df['rating'], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

# Model, optimizer, and loss function
model = GCNRatingPrediction(num_features=num_nodes, hidden_dim=64)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

# Train/test split
train_df, test_df = train_test_split(ratings_df, test_size=0.2)


def train():
    model.train()
    optimizer.zero_grad()

    # Indices for users and movies in the training set
    user_indices = torch.tensor(train_df['UID'], dtype=torch.long)
    movie_indices = torch.tensor(train_df['MID'] + num_users, dtype=torch.long)  # Offset movie indices

    # Actual ratings
    ratings = torch.tensor(train_df['rating'].values, dtype=torch.float)

    # Forward pass and loss calculation
    ratings_pred = model(data, user_indices, movie_indices).squeeze()
    loss = criterion(ratings_pred, ratings)
    loss.backward()
    optimizer.step()
    return loss.item()


# Training loop
for epoch in range(200):  # Number of epochs
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

torch.save(model.state_dict(), 'models/gcn_rating_prediction.pth')

model.eval()  # Set the model to evaluation mode

# Assuming test_user_indices and test_movie_indices contain indices for test data
with torch.no_grad():  # No need to track gradients for testing
    test_user_indices = torch.tensor(test_df['UID'].values, dtype=torch.long)
    test_movie_indices = torch.tensor(test_df['MID'].values + num_users, dtype=torch.long)  # Offset movie indices
    predicted_ratings = model(data, test_user_indices, test_movie_indices).squeeze()

# You can then compare these predicted ratings with the actual ratings from your test set
actual_ratings = torch.tensor(test_df['rating'].values, dtype=torch.float)

# Calculate some metric to evaluate performance, e.g., Mean Squared Error
mse = F.mse_loss(predicted_ratings, actual_ratings)
print(f'Test MSE: {mse.item()}')
