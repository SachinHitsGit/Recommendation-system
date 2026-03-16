# Movie Recommendation System Development 🎬

This repository chronicles the development and optimization of a movie recommendation system utilizing the **MovieLens 1M dataset**. The project systematically explores different recommendation methodologies, starting from traditional content-based filtering leading up to deep-learning-based Neural Collaborative Filtering (NCF). 

It specifically addresses major challenges in recommender system evaluation, including strict data leakage prevention, test stratification, and scalable `Recall@K` measurements via negative sampling.

---

## 📂 Project Files & Approaches

### 1. `Final_rs_tf_idf.ipynb` (Content-Based Filtering)
This notebook implements a foundational **Content-Based Recommender** by analyzing the actual text-based metadata of the movies.
- **Techniques Used**: 
  - Text preprocessing and tokenization of movie titles and genres.
  - Vocabulary building and term weighting (e.g., TF-IDF logic).
  - Generating similarity scores to recommend movies strictly based on how closely their textual descriptions and genres align with movies the user has previously liked.

### 2. `Final_rs_SVD.py` (Matrix Factorization)
This script introduces Collaborative Filtering using **Singular Value Decomposition (SVD)** via Stochastic Gradient Descent (SGD). 
- **Techniques Used**:
  - Direct matrix factorization computing user matrix $P$ and item matrix $Q$ (latent factors = 50).
  - **Regularization**: L2 mathematical weight decay (`reg=0.02`) and low learning rates attached to the gradient steps to ensure stability across epochs.
  - **User-Based Split Strategy**: A strict 80% Train / 20% Test split is enforced *per user* ensuring every single user appears in the test distribution without data overlap.
  - **Evaluation**: Computes `Recall@10` by combining the true test-set items and randomly sampling 100 explicit "unrated" items.
- **Outcome**: A highly stable, classical matrix algorithm that gracefully mitigates exploding gradients, setting a baseline RMSE and `Recall@10`.

### 3. `Final_rs_DL.py` (Neural Collaborative Filtering)
The culmination of the project, taking the matrix interactions and scaling them into an expressive Deep Learning architecture using **PyTorch**.
- **Techniques Used**:
  - **Embeddings & MLP**: Massive embedding layers (`emb_size=128`) passed into a dense Multi-Layer Perceptron (MLP `[256, 128, 64]`) to capture intense non-linear relationships.
  - **Explicit Negative Training Samples**: Forces the model to understand non-interactions by specifically feeding it random unrated items coupled with pseudo-labels of `0`.
  - **Robust Regularization**: Combines L2 Weight Decay (`1e-5`), Heavy Dropout (`0.3`), and strict **Early Stopping** (Patience = 3 based on Test RMSE).
  - **Advanced Evaluation**: Excludes any movie seen during training from candidate pools. Generates `Recall@10`, `Recall@20`, and `Recall@30` against 200 randomly sampled negative items per user.
- **Outcome**: 
  - Massively boosted metrics. Test RMSE remains fully generalized due to early stopping. 
  - Significant Recall metrics (e.g., `Recall@30` pushing above ~0.36), blowing past the simpler linear capabilities of SVD.

---

## 🚀 How to Run

1. **Get the Data**: Obtain the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and place `ratings.dat` and `movies.dat` inside a directory named `ml-1m/`.
2. **Setup Environment**:
   Ensure `pandas`, `numpy`, `scikit-learn`, and `torch` are installed.
3. **Run Evaluations**: 
   ```bash
   # Run the SVD Pipeline
   python3 Final_rs_SVD.py
   
   # Run the Deep Learning NCF Pipeline
   python3 Final_rs_DL.py
   ```

## 🔍 The Evaluation Philosophy
A core focus of this project is proving that classical global accuracy metrics (like `MSE`) can be highly misleading in Recommendations. 
Because of this, both the SVD and DL pipelines rank arrays of truly unseen items using **Negative Sampling**, grading based strictly on `Recall@K`—the true industrial standard for information retrieval. 
