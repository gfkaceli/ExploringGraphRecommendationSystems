from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import cross_validate
import pandas as pd
from surprise.model_selection import train_test_split
from collections import defaultdict

reader = Reader(rating_scale=(1, 5))
ratings = pd.read_csv("datasets/encoded/ratings.csv")
ratings['uid'] = ratings['UID']
ratings['iid'] = ratings['MID']
data = Dataset.load_from_df(ratings[['uid', 'iid', 'rating']], reader)

train, test = train_test_split(data, test_size=0.2, random_state=42)

algo = SVD()

algo.fit(train)

predictions = algo.test(test)

accuracy.rmse(predictions)
accuracy.mae(predictions)
accuracy.fcp(predictions)


def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


top_n = get_top_n(predictions, 10)

for uid, ratings in top_n.items():
    print(uid, [iid for (iid, _) in ratings])
