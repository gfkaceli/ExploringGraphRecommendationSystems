import torch
from models.SAGE import SAGERecommendation
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.optim import Adam
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from log_to_csv import log_metrics_to_csv

ratings_df = pd.read_csv("datasets/encoded/ratings.csv")
user_df = pd.read_csv("datasets/encoded/0.01users.csv")
movies_df = pd.read_csv("datasets/encoded/movies.csv")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Create mappings for user IDs and movie IDs to consecutive integers

# Unique nodes
user_ids = ratings_df['UID'].unique()
movie_ids = ratings_df['MID'].unique()

# Node mapping to indices
user_mapping = {uid: idx for idx, uid in enumerate(user_ids)}
movie_mapping = {mid: idx + len(user_ids) for idx, mid in enumerate(movie_ids)}

# Prepare edges and edge attributes
edges = torch.tensor([(user_mapping[row['UID']], movie_mapping[row['MID']]) for index, row in ratings_df.iterrows()],
                     dtype=torch.long).t().contiguous()
edge_attrs = torch.tensor(ratings_df['rating'].values, dtype=torch.float)

# Node features
user_features = torch.tensor(user_df.drop('UID', axis=1).values, dtype=torch.float)
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


# Model initialization
model = SAGERecommendation(num_features=node_features.shape[1], hidden_dim=64)
optimizer = Adam(model.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss

# Training loop
losses = []
model.train()
for epoch in range(350):
    optimizer.zero_grad()
    out = model(train_data.x, train_data.edge_index)
    loss = criterion(out, train_data.edge_attr)
    loss.backward()
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
plt.title('SAGE Training Loss Curve')
plt.legend()
plt.savefig("metrics/SAGE_train_loss")
plt.clf()

torch.save(model.state_dict(), 'models/SAGE_rating_prediction.pth')

# DataLoader for the validation set
val_loader = DataLoader([test_data], batch_size=1)

# Evaluate the model
model.eval()
with torch.no_grad():
    for data in val_loader:
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.edge_attr)
        test_loss = loss.item()
        print(f'Validation Loss: {test_loss}')

        # Calculate MSE and RMSE
        mse = F.mse_loss(out, data.edge_attr, reduction='mean')
        rmse = torch.sqrt(mse)
        print(f'MSE: {mse.item()}, RMSE: {rmse.item()}')

        # Convert predictions to binary
        predicted_labels = (out > 3.0).float()  # Assuming 3.0 as the threshold

        # True labels are your actual ratings turned into binary (1 if rating > 3.0 else 0)
        true_labels = (test_data.edge_attr > 3.0).float()

        # Calculate precision and recall
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

metrics = {"Precision": precision, "Recall": recall, "Test Loss": test_loss, "MSE": mse.item(), "RMSE"
           : rmse.item()}

log_metrics_to_csv("metrics/SAGE_train_metric.csv", metrics)
