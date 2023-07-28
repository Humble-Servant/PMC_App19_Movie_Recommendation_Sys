import pandas as pd

movies_df = pd.read_csv('data/movies.csv')
ratings_df = pd.read_csv('data/ratings.csv')
credits_df = pd.read_csv('data/credits.csv')

# Calculate a weighted rating
# ===========================
# WR = (v / (v + m)) x R + (m / (v+m)) x C
#
# v = number of votes for a movie (have already)
#
# m = minimum number of votes required
#
# R = average rating of the movie (have already)
#
# C = average rating across all movies

# 90% of movies in dataset have less than this number of votes
m = movies_df["vote_count"].quantile(0.9)
print("m = ", m)

# Average rating of all movies in dataset
C = movies_df["vote_average"].mean()
print("C = ", C)

movies_filtered = movies_df.copy().loc[movies_df["vote_count"] >= m]
print(movies_filtered)


def weighted_rating(df, m=m, C=C):
    v = df["vote_count"]
    R = df["vote_average"]
    return (v/(v+m) * R) + (m/(m+v) * C)


movies_filtered["weighted_rating"] = movies_filtered.apply(weighted_rating, axis=1)
print(movies_filtered.sort_values(by="weighted_rating", ascending=False))
