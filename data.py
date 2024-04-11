import pandas as pd
from convert import convert_categorical
import networkx as nx
import itertools
import collections

ratings_path = 'datasets/ml-100k/ua.base'
users_path = 'datasets/ml-100k/u.user'
items_path = 'datasets/ml-100k/u.item'
# Load ratings
ratings = pd.read_csv(ratings_path, sep='\t', names=['UID', 'MID', 'rating', 'timestamp'],
                      encoding='latin-1')

# Load users
users = pd.read_csv(users_path, sep='|', names=['UID', 'age', 'gender', 'job', 'zip'],
                    encoding='latin-1')

# Load movies (items)
movies = pd.read_csv(items_path, sep='|', names=['MID', 'title', 'release_date', 'video_release_date',
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
movies = movies.drop(['video_release_date', 'IMDb_URL', 'title', 'release_date'], axis=1) # drop these axes
ratings['datetime'] = pd.to_datetime(ratings['timestamp'], unit='s')
ratings['timestamp'] = ratings['datetime']
ratings = ratings.drop('datetime', axis=1)
ratings = ratings.drop('timestamp', axis=1)
print(movies.columns)
print(ratings.head())
print(users.columns)
ratings.to_csv("datasets/encoded/ratings.csv", index=False)
movies.to_csv("datasets/encoded/movies.csv", index=False)
alpha_coefs = [0.01]
G = nx.Graph()
for alpha_coef in alpha_coefs:
    pairs = []
    grouped = ratings.groupby(['MID', 'rating'])
    for key, group in grouped:
        pairs.extend(list(itertools.combinations(group['UID'], 2)))
    counter = collections.Counter(pairs)
    alpha = alpha_coef * 1682  # param*i_no
    edge_list = map(list, collections.Counter(el for el in counter.elements() if counter[el] >= alpha).keys())

    for el in edge_list:
        G.add_edge(el[0], el[1], weight=1)
        G.add_edge(el[0], el[0], weight=1)
        G.add_edge(el[1], el[1], weight=1)

    pr = nx.pagerank(G.to_directed())
    users['PR'] = users['UID'].map(pr)
    users['PR'] /= float(users['PR'].max())
    dc = nx.degree_centrality(G)
    users['CD'] = users['UID'].map(dc)
    users['CD'] /= float(users['CD'].max())
    cc = nx.closeness_centrality(G)
    users['CC'] = users['UID'].map(cc)
    users['CC'] /= float(users['CC'].max())
    bc = nx.betweenness_centrality(G)
    users['CB'] = users['UID'].map(bc)
    users['CB'] /= float(users['CB'].max())
    lc = nx.load_centrality(G)
    users['LC'] = users['UID'].map(lc)
    users['LC'] /= float(users['LC'].max())
    nd = nx.average_neighbor_degree(G, weight='weight')
    users['AND'] = users['UID'].map(nd)
    users['AND'] /= float(users['AND'].max())
    X_train = users[users.columns[0:]]
    X_train.fillna(0, inplace=True)
    X_train.to_csv(f"datasets/encoded/{alpha_coef}users.csv", index=False)


