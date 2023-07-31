import pandas as pd
from surprise import Dataset, Reader, SVD, model_selection

# movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')[['userId', 'movieId', 'rating']]
# credits_df = pd.read_csv('data/credits.csv')

# Create the dataset
reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(ratings_df, reader)

# Build the Training dataset
train_set = dataset.build_full_trainset()

# Train the Model
svd = SVD()
svd.fit(train_set)

# Predict Results
result = svd.predict(15, 1956, 4).est
print(result)

# Validate the Model
# print(model_selection.cross_validate(svd, dataset, measures=['RMSE', 'MAE']))
