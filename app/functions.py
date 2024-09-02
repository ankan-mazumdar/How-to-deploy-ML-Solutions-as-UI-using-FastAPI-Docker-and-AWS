import numpy as np
import polars
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances

# helper function
def returnSearchResultIndexes(query: str, 
                        df: polars.lazyframe.frame.LazyFrame, 
                        model, 
                        dist_metric: str = 'manhattan') -> np.ndarray:
    """
        Function to return indexes of top search results
    """
    
    # embed query
    query_embedding = model.encode(query).reshape(1, -1)
    
    # compute distances between query and titles/transcripts
    dist_arr = pairwise_distances(df.select(df.columns[4:388]).collect().to_numpy(), query_embedding, metric=dist_metric) + \
               pairwise_distances(df.select(df.columns[388:]).collect().to_numpy(), query_embedding, metric=dist_metric)

    # search parameters
    threshold = 40  # eyeballed threshold for Manhattan distance
    top_k = 5

    # evaluate videos close to query based on threshold
    idx_below_threshold = np.argwhere(dist_arr.flatten() < threshold).flatten()
    # keep top k closest videos
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()

    # return indexes of search results
    return idx_below_threshold[idx_sorted][:top_k]
