import pandas as pd
from datetime import datetime
from convert import convert_categorical


ratings_path = 'datasets/ml-100k/u.data'
users_path = 'datasets/ml-100k/u.user'
items_path = 'datasets/ml-100k/u.item'
# Load ratings
ratings = pd.read_csv(ratings_path, sep='\t', names=['UID', 'MID', 'rating', 'timestamp'],
                      encoding='latin-1')

# Load users
users = pd.read_csv(users_path, sep='|', names=['user_id', 'age', 'gender', 'job', 'zip'],
                    encoding='latin-1')

# Load movies (items)
movies = pd.read_csv(items_path, sep='|', names=['movie_id', 'title', 'release_date', 'video_release_date',
                                                 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                                                 "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
                                                 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                                                 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'], encoding='latin-1')

users = convert_categorical(users, 'job')
users = convert_categorical(users, 'gender')
users['bin'] = pd.cut(users['age'], [0, 10, 20, 30, 40, 50, 100], labels=['1', '2', '3', '4', '5', '6'])
""" 
    the above line groups the ages together, so ages
    0-10 for age1 group
    10-20 for age2 
    20-30 for age3
    30-40 for age4
    40-50 fro age5
    50-100 for age6
    we use this to define the boundaries when we encode the data
"""
users['age'] = users['bin']

users = users.drop('bin', axis=1)
users = convert_categorical(users, 'age')
users = users.drop('zip', axis=1)
movies = movies.drop(['video_release_date', 'IMDb_URL'], axis=1)
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['timestamp'] = ratings['datetime']
ratings = ratings.drop('datetime', axis=1)
ratings = ratings.drop('timestamp', axis=1)
print(movies.columns)
print(ratings.head())
print(users.columns)
ratings.to_csv("datasets/encoded/ratings.csv", index=False)


