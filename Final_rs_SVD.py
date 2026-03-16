import pandas as pd
import numpy as np

def split_dataset_per_user(ratings, test_size=0.2, random_state=42):

    test = ratings.groupby('userId').sample(frac=test_size, random_state=random_state)
    train = ratings.drop(test.index)
    return train, test

def initialize_matrices(n_users, n_movies, n_factors=50):

    P = np.random.normal(scale=0.1, size=(n_users, n_factors))
    Q = np.random.normal(scale=0.1, size=(n_movies, n_factors))
    bu = np.zeros(n_users)
    bi = np.zeros(n_movies)
    return P, Q, bu, bi

def predict_rating(u, i, mu, P, Q, bu, bi):

    return mu + bu[u] + bi[i] + np.dot(P[u], Q[i])

def train_svd(train, all_users, all_movies, n_factors=50, lr=0.005, reg=0.02, epochs=40):
                                                                        
    user_map = {id: i for i, id in enumerate(all_users)}
    movie_map = {id: i for i, id in enumerate(all_movies)}
    
    n_users = len(all_users)
    n_movies = len(all_movies)
    
                           
    P, Q, bu, bi = initialize_matrices(n_users, n_movies, n_factors)
    mu = train["rating"].mean()
    
                                                
    train_users = np.array([user_map[u] for u in train["userId"]])
    train_movies = np.array([movie_map[m] for m in train["movieId"]])
    train_ratings = train["rating"].values
    n_samples = len(train_ratings)
    
    print(f"Training SVD model with lr={lr}, reg={reg}, factors={n_factors}, epochs={epochs}")
    
    for epoch in range(epochs):
                                                                 
        indices = np.random.permutation(n_samples)
        
        epoch_sq_err = 0.0
        
        for idx in indices:
            u = train_users[idx]
            i = train_movies[idx]
            r = train_ratings[idx]
            
                             
            pred = predict_rating(u, i, mu, P, Q, bu, bi)
            err = r - pred
            
            epoch_sq_err += err ** 2
            
                                                                
                                                                                                                 
            P_u_old = P[u].copy()
            
                                                                             
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            
            P[u] += lr * (err * Q[i] - reg * P[u])
            Q[i] += lr * (err * P_u_old - reg * Q[i])
            
        mse = epoch_sq_err / n_samples
        rmse = np.sqrt(mse)
        print(f"Epoch {epoch+1:02d}/{epochs} | Average MSE: {mse:.4f} | RMSE: {rmse:.4f}")
            
    model = {
        'P': P, 'Q': Q, 'bu': bu, 'bi': bi, 'mu': mu,
        'user_map': user_map, 'movie_map': movie_map,
        'n_factors': n_factors
    }
    return model

def generate_top_k_recommendations(user_id, candidate_movies, model, k=10):

    if user_id not in model['user_map']:
        return []
        
    u = model['user_map'][user_id]
    
                                                     
    candidate_idx = [model['movie_map'][m] for m in candidate_movies]
    
    mu = model['mu']
    bu = model['bu'][u]
    bi_candidates = model['bi'][candidate_idx]
    Q_candidates = model['Q'][candidate_idx]
    
    preds = mu + bu + bi_candidates + np.dot(Q_candidates, model['P'][u])
    
                                           
    top_k_indices = np.argsort(preds)[::-1][:k]
    return [candidate_movies[idx] for idx in top_k_indices]

def compute_recall_at_k(train, test, model, all_movies, k=10, n_negatives=100):

    recalls = []
    test_users = test['userId'].unique()
    
    print(f"Evaluating Recall@{k} for {len(test_users)} test users with {n_negatives} negatives per user...")
    
    train_grouped = train.groupby('userId')['movieId'].apply(set).to_dict()
    test_grouped = test.groupby('userId')['movieId'].apply(set).to_dict()
    all_movies_set = set(all_movies)
    
    for i, user in enumerate(test_users):
        relevant_items = test_grouped.get(user, set())
        if len(relevant_items) == 0:
            continue
            
        watched_train = train_grouped.get(user, set())
        
                                                               
        unrated_movies = list(all_movies_set - watched_train - relevant_items)
        
                           
        negatives = np.random.choice(unrated_movies, size=min(n_negatives, len(unrated_movies)), replace=False)
        
                                                                      
        candidate_movies = list(relevant_items) + list(negatives)
            
        recs = generate_top_k_recommendations(user, candidate_movies, model, k=k)
        hits = len(set(recs) & relevant_items)
        recall = hits / len(relevant_items)
        recalls.append(recall)
            
    mean_recall = np.mean(recalls)
    print(f"Final Average Recall@{k}: {mean_recall:.6f}")
    
    return mean_recall

if __name__ == "__main__":
    import os
    if os.path.exists("ml-1m/ratings.dat"):
        print("Loading ratings.dat...")
        ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", 
                              names=["userId", "movieId", "rating", "timestamp"])
        
        all_users = ratings["userId"].unique()
        all_movies = ratings["movieId"].unique()
        
        print("1. Splitting dataset per user...")
        train, test = split_dataset_per_user(ratings, test_size=0.2)
        print(f"Train size: {len(train)}, Test size: {len(test)}")
        
        print("\\n2. Training stable SVD model...")
        model = train_svd(train, all_users, all_movies, n_factors=50, lr=0.005, reg=0.02, epochs=40)
        
        print("\\n3. Generating recommendations and Computing Recall@10...")
        final_recall = compute_recall_at_k(train, test, model, all_movies, k=10, n_negatives=100)
    else:
        print("Put 'ml-1m/ratings.dat' in your current working directory to test the pipeline.")
