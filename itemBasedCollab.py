import numpy as np
import pandas as pd

def find_k_nearest_items(matrix, target_item, k, similarity_fn):
    
    similarities = []
    
    if target_item not in matrix.columns:
        return []
    
    vec1 = matrix[target_item]
    
    for item in matrix.columns:
        if item == target_item:
            continue
            
        vec2 = matrix[item]
        
        common_v1, common_v2 = get_co_rated(matrix, target_item, item, axis="columns")
        
        if len(common_v1) < 2:
            continue
            
        score = similarity_fn(common_v1, common_v2)
        if score is not None:
            similarities.append((item, score))
            
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]


def predict_rating_item_based(matrix, target_user, target_item, k, similarity_fn):

    nearest_items = find_k_nearest_items(matrix, target_item, k, similarity_fn)
    
    if not nearest_items:
        return None
        
    numerator = 0.0
    denominator = 0.0
    
    if target_user not in matrix.index:
        return None
        
    user_ratings = matrix.loc[target_user]
    
    for item, sim in nearest_items:
        if item in user_ratings.index and not pd.isna(user_ratings[item]):
            rating = user_ratings[item]
            numerator += sim * rating
            denominator += abs(sim)
            
    if denominator == 0:
        return None
        
    return numerator / denominator