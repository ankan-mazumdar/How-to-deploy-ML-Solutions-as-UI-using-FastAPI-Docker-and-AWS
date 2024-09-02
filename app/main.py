from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer
from app.functions import returnSearchResultIndexes

# define model info
model_name = 'all-MiniLM-L6-v2'
model_path = "app/data/" + model_name

# load model
model = SentenceTransformer(model_path)

# load video index
df = pl.scan_parquet('app/data/video-index.parquet')

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'yt-search', 'description': "Search API for Shaw Talebi's YouTube videos."}

@app.get("/search")
def search(query: str):
    idx_result = returnSearchResultIndexes(query, df, model)
    return df.select(['title', 'video_id']).collect()[idx_result].to_dict(as_series=False)
