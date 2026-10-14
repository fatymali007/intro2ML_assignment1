import numpy as np
import pandas as pd

def train_test_split_ratings(ratings_df, test_fraction=0.2, seed=42):

    np.random.seed(seed)
    test_rows = []
    train_rows = []
    
    for user_id, group in ratings_df.groupby('userId'):
        n_samples = len(group)
        if n_samples < 2:
            train_rows.append(group)
            continue
            
        n_test = max(1, int(n_samples * test_fraction))
        shuffled = group.sample(frac=1, random_state=seed)
        
        test_rows.append(shuffled.iloc[:n_test])
        train_rows.append(shuffled.iloc[n_test:])
        
    test_df = pd.concat(test_rows)
    train_df = pd.concat(train_rows)
    return train_df, test_df


def evaluate(train_matrix, test_df, k, similarity_fn, method="user-based"):

    predictions = []
    actuals = []
    none_count = 0
    total_count = len(test_df)
    
    for _, row in test_df.iterrows():
        user = row['userId']
        item = row['movieId']
        true_rating = row['rating']
        
        if user not in train_matrix.index or item not in train_matrix.columns:
            none_count += 1
            continue
            
        if method == "user-based":
            pred = predict_rating_user_based(train_matrix, user, item, k, similarity_fn)
        elif method == "item-based":
            pred = predict_rating_item_based(train_matrix, user, item, k, similarity_fn)
        else:
            pred = None
            
        if pred is None:
            none_count += 1
        else:
            predictions.append(pred)
            actuals.append(true_rating)
            
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    if len(predictions) == 0:
        rmse, mae = np.nan, np.nan
    else:
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))
        
    none_fraction = none_count / total_count if total_count > 0 else 0.0
    
    return {
        "RMSE": rmse,
        "MAE": mae,
        "None_Fraction": none_fraction,
        "Evaluated_Count": len(predictions)
    }