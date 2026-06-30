from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from engine import DataLoader, RecommenderEngine

app = FastAPI(title="Hybrid Movie Recommender API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Initializing Data & Engine from Local TMDB CSVs...")
try:
    movies_df, ratings_df = DataLoader.load_data()
    engine = RecommenderEngine(movies_df, ratings_df)
    print("Engine ready!")
except FileNotFoundError as e:
    print(f"CRITICAL ERROR: {e}")
    # Initialize engine with empty dataframes so API doesn't crash on boot,
    # but endpoints will fail gracefully or return empty.
    import pandas as pd
    engine = RecommenderEngine(pd.DataFrame(columns=['movieId', 'title', 'soup', 'vote_average', 'original_language']), pd.DataFrame(columns=['movieId', 'userId', 'rating']))

class RecommendRequest(BaseModel):
    title: str
    industries: List[str] = []
    alpha: float = 1.0
    top_n: int = 5
    api_key: Optional[str] = None

@app.get("/api/movies")
async def get_movies():
    if engine.movies_df.empty:
        return {"movies": []}
    titles = engine.movies_df['title'].tolist()
    return {"movies": titles}

@app.post("/api/recommend")
async def recommend_movies(req: RecommendRequest):
    if engine.movies_df.empty:
        raise HTTPException(status_code=500, detail="Engine failed to start. Missing Kaggle TMDB CSV files in mmovie folder.")
        
    try:
        results = engine.recommend(
            title=req.title,
            alpha=req.alpha,
            target_industries=req.industries,
            top_n=req.top_n,
            api_key=req.api_key
        )
        return {"recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/random")
async def get_random_movies():
    if engine.movies_df.empty:
        return {"recommendations": []}
    
    # Sample 5 random movies
    sample = engine.movies_df.sample(5)
    results = []
    for _, row in sample.iterrows():
        poster = row['poster_path'] if pd.notna(row['poster_path']) and row['poster_path'] != "" else "https://placehold.co/500x750/1e293b/38bdf8.png?text=No+Poster+Found"
        results.append({
            "movieId": int(row['movieId']),
            "title": row['title'],
            "original_language": row['original_language'],
            "genres": row['genres_parsed'].replace(" ", ", ") if pd.notna(row['genres_parsed']) else "",
            "poster_path": poster,
            "match_score": 100.0 # Just a placeholder score for the landing page
        })
    return {"recommendations": results}
