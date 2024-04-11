import torch
from models.SAGE import SAGERecommendation
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F


ratings_data = pd.read_csv("datasets/encoded/ratings.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create mappings for user IDs and movie IDs to consecutive integers

# Unique nodes
user_ids = ratings_data['UID'].unique()
movie_ids = ratings_data['MID'].unique()

# Node mapping to indices
user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
movie_mapping = {mid: idx + len(user_ids) for idx, mid in enumerate(movie_ids)}

train_data, val_data = train_test_split(ratings_data, test_size=0.2, random_state=42)

# Random node features
num_users = len(user_ids)
num_movies = len(movie_ids)
num_features = 10  # Random choice, can be tuned
node_features = torch.randn((num_users + num_movies, num_features), dtype=torch.float)


# Create Data objects for training and validation
def create_pyg_data(data_frame):
    edge_index = torch.tensor([
        [user_mapping[uid], movie_mapping[mid]] for uid, mid in zip(data_frame['UID'], data_frame['MID'])
    ], dtype=torch.long).t().contiguous()

    edge_attr = torch.tensor(data_frame['rating'].values, dtype=torch.float).unsqueeze(1)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


train_dataset = create_pyg_data(train_data)
val_dataset = create_pyg_data(val_data)


# Model initialization
model = SAGERecommendation(num_features=10, hidden_dim=32)
optimizer = Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Training loop
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(train_dataset)
    loss = criterion(out, train_dataset.edge_attr)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}: Loss {loss.item()}')

torch.save(model.state_dict(), 'models/SAGE_rating_prediction.pth')

# DataLoader for the validation set
val_loader = DataLoader([val_dataset], batch_size=1)

# Evaluate the model
model.eval()
with torch.no_grad():
    for data in val_loader:
        out = model(data)
        loss = criterion(out, data.edge_attr)
        print(f'Validation Loss: {loss.item()}')

        # Calculate MSE and RMSE
        mse = F.mse_loss(out, data.edge_attr, reduction='mean')
        rmse = torch.sqrt(mse)
        print(f'MSE: {mse.item()}, RMSE: {rmse.item()}')
