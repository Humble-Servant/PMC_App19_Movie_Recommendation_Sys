import pandas as pd

movies_df = pd.read_csv('data/movies.csv')

# Similarity Matrix
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
movies_df['overview'] = movies_df['overview'].fillna("")
tfidf_matrix = tfidf.fit_transform(movies_df['overview'])

from sklearn.metrics.pairwise import linear_kernel
similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)


def similar_movies(movie_title, num_movies):
    idx = movies_df.loc[movies_df['title'] == movie_title].index[0]
    scores = list(enumerate(similarity_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    movie_indices = [tpl[0] for tpl in scores[1:num_movies+1]]
    similar = list(movies_df["title"].iloc[movie_indices])
    return similar


print(similar_movies("The Avengers", 5))
