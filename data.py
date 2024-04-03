import pandas as pd
from datetime import datetime

# pass in column names for each CSV
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('datasets/ml-100k/u.user', sep='|', names=u_cols,
                    encoding='latin-1')

r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('datasets/ml-100k/u.data', sep='\t', names=r_cols,
                      encoding='latin-1')

# the movies file contains columns indicating the movie's genres
# let's only load the first five columns of the file with usecols
m_cols = ['movie_id', 'title', 'release_date']
movies = pd.read_csv('datasets/ml-100k/u.item', sep='|', names=m_cols, usecols=range(3),
                     encoding='latin-1')

print(movies.head())
print(ratings.head())
print(users.head())

# Example: Check for missing values
print(ratings.isnull().sum())
print(users.isnull().sum())
print(movies.isnull().sum())