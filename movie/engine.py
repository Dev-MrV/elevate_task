import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import os
import re

class DataLoader:
    """Loads the IMDb Top 1000 dataset which natively contains Poster Links."""

    @classmethod
    def load_data(cls):
        movies_path = 'imdb_top_1000.csv'
        
        if not os.path.exists(movies_path):
            raise FileNotFoundError(f"Missing IMDb CSV file. Ensure '{movies_path}' is inside the mmovie folder.")
            
        print("Loading local IMDb Top 1000 dataset...")
        df = pd.read_csv(movies_path)
        
        # Standardize column names to match the backend expectation
        df = df.rename(columns={
            'Series_Title': 'title',
            'IMDB_Rating': 'vote_average',
            'No_of_Votes': 'vote_count',
            'Overview': 'overview',
            'Genre': 'genres_parsed'
        })
        
        # Handle high-res poster links by stripping Amazon's low-res suffix from the URL
        def get_high_res_poster(link):
            if pd.isna(link):
                return ""
            # E.g., ...@._V1_UX67_CR0,0,67,98_AL_.jpg -> ...@._V1_.jpg
            return re.sub(r'(\._V1_).*\.jpg$', r'\1.jpg', str(link))
            
        df['poster_path'] = df['Poster_Link'].apply(get_high_res_poster)
        
        # We don't have language data in IMDb Top 1000, so we label all as 'en' 
        # so they pass the Hollywood frontend filter check.
        df['original_language'] = 'en'
        
        # Safely fill NA for string concatenation
        df['genres_parsed'] = df['genres_parsed'].fillna('')
        df['overview'] = df['overview'].fillna('')
        df['Director'] = df['Director'].fillna('')
        df['Star1'] = df['Star1'].fillna('')
        df['Star2'] = df['Star2'].fillna('')
        df['Star3'] = df['Star3'].fillna('')
        df['Star4'] = df['Star4'].fillna('')
        
        # Construct the Metadata Soup
        df['soup'] = (df['genres_parsed'] + " " + 
                      df['Director'] + " " + 
                      df['Star1'] + " " + 
                      df['Star2'] + " " + 
                      df['Star3'] + " " + 
                      df['Star4'] + " " + 
                      df['overview'])
                      
        df['soup'] = df['soup'].str.lower()
        
        # Synthesize a movieId since the dataset doesn't have numerical IDs
        df['movieId'] = range(1, len(df) + 1)
        
        print("Simulating User Ratings for SVD Collaborative Filter...")
        np.random.seed(42)
        ratings = []
        popular_movies = df['movieId'].tolist()
        
        # Create 100 users rating 50 movies each
        for u in range(1, 101):
            if len(popular_movies) > 50:
                user_movies = np.random.choice(popular_movies, 50, replace=False)
            else:
                user_movies = popular_movies
                
            for m in user_movies:
                base_rating = df[df['movieId'] == m]['vote_average'].values[0] / 2.0  # Scale 10 down to 5
                rating = np.clip(np.random.normal(base_rating, 0.5), 1.0, 5.0)
                ratings.append({'userId': u, 'movieId': m, 'rating': round(rating, 1)})
                
        ratings_df = pd.DataFrame(ratings)
        
        return df, ratings_df


class RecommenderEngine:
    def __init__(self, movies_df, ratings_df):
        self.movies_df = movies_df.reset_index(drop=True)
        self.ratings_df = ratings_df
        
        print("Building Content Matrix (TF-IDF)...")
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.content_matrix = self.tfidf.fit_transform(self.movies_df['soup'])
        
        print("Building Collaborative Matrix (SVD)...")
        self.pivot = self.ratings_df.pivot(index='movieId', columns='userId', values='rating').fillna(0)
        
        valid_movie_ids = self.pivot.index.intersection(self.movies_df['movieId'])
        self.pivot = self.pivot.loc[valid_movie_ids]
        
        self.svd = TruncatedSVD(n_components=20, random_state=42)
        self.item_embeddings = self.svd.fit_transform(self.pivot)
        
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df['title'].str.lower()).drop_duplicates().to_dict()
        self.movie_id_to_pivot_idx = {m_id: i for i, m_id in enumerate(self.pivot.index)}

    def recommend(self, title, alpha=1.0, target_industries=None, top_n=5, api_key=None):
        title = title.lower()
        if title not in self.indices:
            return self._fallback_recommendations(target_industries, top_n)
            
        idx = self.indices[title]
        target_movie_id = self.movies_df.iloc[idx]['movieId']
        
        content_sim = cosine_similarity(self.content_matrix[idx], self.content_matrix).flatten()
        
        collab_sim = np.zeros(len(self.movies_df))
        if target_movie_id in self.movie_id_to_pivot_idx:
            pivot_idx = self.movie_id_to_pivot_idx[target_movie_id]
            target_embedding = self.item_embeddings[pivot_idx].reshape(1, -1)
            raw_collab_sim = cosine_similarity(target_embedding, self.item_embeddings).flatten()
            
            for i, m_id in enumerate(self.pivot.index):
                df_idx = self.movies_df.index[self.movies_df['movieId'] == m_id]
                if len(df_idx) > 0:
                    collab_sim[df_idx[0]] = raw_collab_sim[i]
        
        hybrid_scores = (alpha * content_sim) + ((1 - alpha) * collab_sim)
        
        vote_averages = self.movies_df['vote_average'].fillna(5.0).values
        sentiment_modifiers = 1.0 + ((vote_averages - 5.0) / 50.0) 
        hybrid_scores = hybrid_scores * sentiment_modifiers
        
        self.movies_df['hybrid_score'] = hybrid_scores
        candidates = self.movies_df[self.movies_df['movieId'] != target_movie_id].copy()
        
        if target_industries and len(target_industries) > 0:
            candidates = candidates[candidates['original_language'].isin(target_industries)]
            
        top_candidates = candidates.sort_values(by='hybrid_score', ascending=False).head(top_n)
        
        results = []
        for _, row in top_candidates.iterrows():
            poster = row['poster_path'] if row['poster_path'] else "https://placehold.co/500x750/1e293b/38bdf8.png?text=No+Poster+Found"
            results.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "original_language": row['original_language'],
                "genres": row['genres_parsed'].replace(" ", ", "),
                "poster_path": poster,
                "match_score": round(row['hybrid_score'] * 100, 1)
            })
            
        return results
        
    def _fallback_recommendations(self, target_industries, top_n):
        candidates = self.movies_df.copy()
        if target_industries and len(target_industries) > 0:
            candidates = candidates[candidates['original_language'].isin(target_industries)]
            
        if len(candidates) == 0:
            return []
            
        top_candidates = candidates.sort_values(by=['vote_count', 'vote_average'], ascending=False).head(top_n)
        results = []
        for _, row in top_candidates.iterrows():
            poster = row['poster_path'] if row['poster_path'] else "https://placehold.co/500x750/1e293b/38bdf8.png?text=No+Poster+Found"
            results.append({
                "movieId": int(row['movieId']),
                "title": row['title'],
                "original_language": row['original_language'],
                "genres": row['genres_parsed'].replace(" ", ", "),
                "poster_path": poster,
                "match_score": 0.0
            })
        return results
