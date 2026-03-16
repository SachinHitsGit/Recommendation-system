import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
import os

def split_dataset_per_user(ratings, test_size=0.2, random_state=42):
                                                                                   
    test = ratings.groupby('userId').sample(frac=test_size, random_state=random_state)
    
                                                                                 
    train = ratings.drop(test.index)
    
    return train, test

def add_negative_samples_to_train(train_df, num_movies, global_mean, global_std, n_negatives=1):

    print(f"Adding {n_negatives} negative sample(s) per positive interaction in training...")
    neg_users = []
    neg_movies = []
    
    train_grouped = train_df.groupby('user')['movie'].apply(set).to_dict()
    all_movies = np.arange(num_movies)
    
    for u, u_movies in train_grouped.items():
        n_to_sample = len(u_movies) * n_negatives
        unrated = list(set(all_movies) - u_movies)
        if len(unrated) == 0: continue
        sampled = np.random.choice(unrated, size=min(n_to_sample, len(unrated)), replace=False)
        neg_users.extend([u] * len(sampled))
        neg_movies.extend(sampled)
            
    neg_df = pd.DataFrame({'user': neg_users, 'movie': neg_movies})
                             
    norm_0 = (0.0 - global_mean) / global_std
    neg_df['rating'] = norm_0
    
    return pd.concat([train_df[['user', 'movie', 'rating']], neg_df], ignore_index=True)

class RatingDataset(Dataset):

    def __init__(self, df):
        self.users = torch.tensor(df["user"].values, dtype=torch.long)
        self.movies = torch.tensor(df["movie"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

class NCF(nn.Module):

    def __init__(self, num_users, num_movies, emb_size=128, hidden_sizes=[256, 128, 64], dropout=0.3):
        super().__init__()
                    
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.movie_emb = nn.Embedding(num_movies, emb_size)
        
                    
        layers = []
        input_size = emb_size * 2
        for h in hidden_sizes:
            layers.append(nn.Linear(input_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))                             
            input_size = h
        self.mlp = nn.Sequential(*layers)
        
                      
        self.output = nn.Linear(input_size, 1)
        
    def forward(self, user, movie):
        u = self.user_emb(user)
        m = self.movie_emb(movie)
        x = torch.cat([u, m], dim=1)
        x = self.mlp(x)
        x = self.output(x).squeeze()
        return x

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001, device="cpu", patience=3):

    model = model.to(device)
                                                                                            
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    print(f"Training NCF model for up to {epochs} epochs on device: {device}")
    
    best_test_rmse = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, movie, rating in train_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)

            pred = model(user, movie)
            loss = loss_fn(pred, rating)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(user)
        
        train_rmse = math.sqrt(total_loss / len(train_loader.dataset))
        
                                             
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for user, movie, rating in test_loader:
                user = user.to(device)
                movie = movie.to(device)
                rating = rating.to(device)
                
                pred = model(user, movie)
                loss = loss_fn(pred, rating)
                test_loss += loss.item() * len(user)
                
        test_rmse = math.sqrt(test_loss / len(test_loader.dataset))
        
        print(f"Epoch {epoch+1:02d}/{epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
        
                              
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            epochs_no_improve = 0
                                 
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered! No improvement in Test RMSE for {patience} consecutive epochs.")
                break
                
                                                       
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
    return model

def generate_top_k_recommendations(user_idx, candidate_movie_indices, model, k=30, device="cpu"):

    model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * len(candidate_movie_indices), dtype=torch.long).to(device)
        movie_tensor = torch.tensor(candidate_movie_indices, dtype=torch.long).to(device)
        
                                                  
        scores = model(user_tensor, movie_tensor)
        
                                               
                                                                         
        top_k_idx = torch.topk(scores, min(k, len(candidate_movie_indices))).indices.cpu().numpy()
        
                                                          
        top_k_movies = [candidate_movie_indices[idx] for idx in top_k_idx]
        
    return top_k_movies

def compute_recall_at_k(train, test, model, num_movies, k_list=[10, 20, 30], n_negatives=200, device="cpu"):

    recalls = {k: [] for k in k_list}
    test_users = test['user'].unique()
    max_k = max(k_list)
    
    print(f"\\nEvaluating Recall@{k_list} for {len(test_users)} test users with {n_negatives} negatives per user...")
    
                                                          
    train_grouped = train.groupby('user')['movie'].apply(set).to_dict()
    test_grouped = test.groupby('user')['movie'].apply(set).to_dict()
    
    all_movies_set = set(range(num_movies))
    
    for i, user_idx in enumerate(test_users):
        relevant_items = test_grouped.get(user_idx, set())
        if len(relevant_items) == 0:
            continue
            
        watched_train = train_grouped.get(user_idx, set())
        
                                                               
        unrated_movies = list(all_movies_set - watched_train - relevant_items)
        
                           
        negatives = np.random.choice(unrated_movies, size=min(n_negatives, len(unrated_movies)), replace=False)
        
                                                                           
        candidate_movies = list(relevant_items) + list(negatives)
        
                                                                           
        top_recs_list = generate_top_k_recommendations(user_idx, candidate_movies, model, k=max_k, device=device)
        
        for k in k_list:
            recs_k = set(top_recs_list[:k])
            
                                                                            
            hits = len(recs_k & relevant_items)
            
                                                 
            recall = hits / len(relevant_items)
            recalls[k].append(recall)
        
        if (i + 1) % 1000 == 0:
            print(f"Evaluated {i+1} users...")
            
    mean_recalls = {k: np.mean(recalls[k]) for k in k_list}
    
    for k in k_list:
        print(f"Final Average Recall@{k}: {mean_recalls[k]:.6f}")
    
    return mean_recalls

if __name__ == "__main__":
    if os.path.exists("ml-1m/ratings.dat"):
        print("Loading ratings.dat...")
        ratings = pd.read_csv("ml-1m/ratings.dat", sep="::", engine="python", 
                              names=["userId", "movieId", "rating", "timestamp"])
        
        print("1. Splitting dataset per user (80% Train, 20% Test)...")
        train_df, test_df = split_dataset_per_user(ratings, test_size=0.2)
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

        print("\\n2. Mapping userId and movieId to 0-indexed integers across full dataset...")
                                                                                         
        user_ids = ratings["userId"].unique()
        movie_ids = ratings["movieId"].unique()
        
        user2idx = {u: i for i, u in enumerate(user_ids)}
        movie2idx = {m: i for i, m in enumerate(movie_ids)}
        
        num_users = len(user_ids)
        num_movies = len(movie_ids)
        
        train_df["user"] = train_df["userId"].map(user2idx)
        train_df["movie"] = train_df["movieId"].map(movie2idx)
        test_df["user"] = test_df["userId"].map(user2idx)
        test_df["movie"] = test_df["movieId"].map(movie2idx)
        
        print("\\n3. Normalizing ratings globally (mean=0, std=1)...")
        global_mean = train_df["rating"].mean()
        global_std = train_df["rating"].std()
        
        train_df["rating"] = (train_df["rating"].astype("float32") - global_mean) / global_std
        test_df["rating"] = (test_df["rating"].astype("float32") - global_mean) / global_std

        print("\\n3.5. Adding negative sampling to training...")
        train_df_with_neg = add_negative_samples_to_train(train_df, num_movies, global_mean, global_std, n_negatives=1)

        print("\\n4. Initializing DataLoaders...")
        train_dataset = RatingDataset(train_df_with_neg)
        test_dataset = RatingDataset(test_df)
        
        train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

        device = "cuda" if torch.cuda.is_available() else "cpu"
                                                                                        
        model = NCF(num_users, num_movies, emb_size=128, hidden_sizes=[256, 128, 64], dropout=0.3)
        
        print("\\n5. Training model with Early Stopping and L2 Regularization...")
                                                                                      
                                              
        model = train_model(model, train_loader, test_loader, epochs=30, lr=0.001, device=device, patience=3)
        
        print("\\n6. Computing Recall@K (10, 20, 30)...")
                                                                                                           
        compute_recall_at_k(train_df, test_df, model, num_movies, k_list=[10, 20, 30], n_negatives=200, device=device)
    else:
        print("Put 'ml-1m/ratings.dat' in your current working directory to test the pipeline.")
