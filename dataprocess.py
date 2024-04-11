import pandas as pd

movies_data = pd.read_csv('datasets/encoded/movies.csv')
users_data  = pd.read_csv('datasets/encoded/0.01users.csv')

print(movies_data.head())
print(users_data.head())
