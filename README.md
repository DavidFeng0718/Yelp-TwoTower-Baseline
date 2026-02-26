
# Two-Tower Recommendation System 

A PyTorch implementation of an optimized Two-Tower model for large-scale retrieval and recommendation tasks.

This project focuses on scalable embedding-based retrieval using a dual-encoder architecture, suitable for industrial recommendation systems.

---

## Overview

This repository implements a Two-Tower architecture for user-item matching.

The model learns:

- A user embedding tower
- An item embedding tower

The similarity between user and item embeddings (dot product) determines recommendation scores.

---

##  Two-Tower Recommendation System 
    project/
    │
    ├── data/
    │ ├── cb_baseline.parquet
    │ └── geo_feature_matrix.csv
    │
    ├── artifacts/  # saved models and embeddings
    ├── src/        # model and training code
    │
    ├── runs/       # Tensorboard's training log directory
    │
    ├── TwoTowerMethod.ipynb 
    │
    ├── requirements.txt
    └── README.md

---

## Configuration

All hyperparameters are centralized in a `Config` dataclass:

```python
class Config:
    embed_dim: int = 128
    hidden_dim: int = 512
    lr: float = 1e-3                # learning rate -> Control step size for each parameter update
    weight_decay: float = 1e-5      # L2 regular term coefficient
    epochs: int = 30
    batch_size: int = 2048
    k: int = 10                     # K in "Top-k" recommendation
```

---

# Installation

Create virtual environment:

```Bash
conda create -n two-tower python=3.10 -y
conda activate two-tower
pip install -r requirements.txt
```
Adjust CUDA version if needed

---

# Training

Run the training notebook or script:

```Bash
jupyter notebook TwoTowerMethodOptimized.ipynb
```

---

# License
MIT License

---

# Technical Documentation

## Problem Definition

This project addresses large-scale user-item retrieval using embedding-based dual-encoder architecture.
Given a user matrix, the system retrieves top-K relevant items.

## Data Processing & feature Construction

### 1. Overviwe

This project integrates two large-scale datasets:
1.	Yelp Open Dataset – user–business interaction data.
2.  OpenStreetMap (OSM) GIS Dataset – spatial context features around businesses.

The goal is to construct a rich feature space for a Two-Tower retrieval model by combining:
1.  User-item interaction signals
2.  Business data
3.  Geographic environment features

### 2. Yelp Dataset Processing
I use `yelp_academic_dataset_business.json` and `yelp_academic_dataset_review.json` as the raw dataset.
Data extraction is performed using `DuckDB` + `SQL`.

#### 2.1 Business.json Filtering
From `business.json`, I
- Filter businesses belonging to restaurant related categories on DuckDB
- Retain businesses identifiers and geographic coordinates

This step ensures the recommendation task focuses exclusively on restaurant businesses.
The output fields:
- `business_id`
- `latitude`
- `longitude`
- `categories`

#### 2.2 User-Item Interaction Construction

From review.json, I extract:
- `user_id`
- `business_id`
- `rating`
- `centered_rating`

The centered rating is computed as:
$$
\text{centered}{\text{_rating}} = \text{rating} - 2
$$
The purpose of subtracting 2 is to shift the rating distribution such that:
- Ratings ≥ 3 become positive values
- Ratings ≤ 2 become zero or negative values

It increases separation between positive and negative feedback and allows the model to treat higher ratings as stronger positive signals, while low ratings become weak or negative signals.

### 3 OSM GIS Feature Engineering
#### 3.1 Motivation
The restaurant preference is influenced by ratings, but also by 
- Surrounding infrastructure
- Commercial density
- Accessibility
- Environmental context

To model these contextual signals, I use these method to extract spatial features:
- OpenStreetMap
- python `geopandas` model 

For each restaurant location, I create circular buffers at r = 100m, 200m and 500m.

These represent: Immediate street level context(100m), Neighborhood level context(200m) and District level context(500m)

#### 3.2 Extracted Feature Categories

##### 1. Point-based Density Features (Counts)
- `pois_*_count`
- `places_*_count`
- `traffic_*_count`
- `transport_*_count`

Measures commercial density and transportation accessibility.

##### 2. Linear Infrastructure Features (Count + Length)
- `roads_*_count`, `roads_*_lenroads_*_count`，`roads_*_len`
- `railways_*_count`, `railways_*_lenrailways_*_count`，`railways_*_len`
- `waterways_*_count`, `waterways_*_lenwaterways_*_count`，`waterways_*_len`

Reflects transportation network density and environmental layout.

##### 3. Area-Based Environmental Features (Count + Area)
- `buildings_*_count`, `buildings_*_areabuildings_*_count`，`buildings_*_area`
- `landuse_*_count`, `landuse_*_arealanduse_*_count`，`landuse_*_area`
- `natural_*_count`, `natural_*_areanatural_*_count`，`natural_*_area`
- `water_*_count`, `water_*_areawater_*_count`，`water_*_area`

Captures urban density, land usage patterns, natural coverage, and environmental context.


### 4. Train-Test Split
The dataset is split as 80% training and 20% testing.
80/20 split is standard for supervised recommendation tasks.
Fixed random_state ensures reproducibility
Random split ensures distributional consistency.

## Feature Engineering

### 1. User Tower
The User Tower encodes user-related information into a dense vector representation in latent space.
Feature:
- Only positive samples are used to build user profiles
- Profile vector is computed as the average of content features from positively interacted items

```
user_profile = mean(item_content_features | positive interactions)
```

#### Input to User Tower:
Each training sample contains:
- `X_user`: user profile features
- `X_item`: item content features

#### Network Architecture
The User Tower is implemented as a shallow MLP:
```text
User Profile Features
        ↓
      Linear
        ↓
       ReLU
        ↓
      Linear
        ↓
User Embedding (ℝ^embed_dim)
```

### 2. Item Tower
The Item Tower encodes item-related information into a dense latent representation that resides in the same embedding space as user embeddings.

This implementation uses content-based item features derived from `geo_feature_matrix.csv`

Before training, item features are standardized using `StandardScaler`:

```(x - mean) / std```

#### Network Architecture
The Item Tower is implemented as a shallow MLP
```text
Item Content Features
        ↓
      Linear
        ↓
      ReLU
        ↓
     Linear
        ↓
Item Embedding (ℝ^embed_dim -> same as user tower)
```

## Similarity Function & Scoring Mechanism

After generating user and item embeddings, the model computes a compatibility score to measure relevance.
```score(u, i) = similarity(user_embedding, item_embedding)```

The scoring function is defined as:
```score(u, i) = uᵀ i```

During the inference, it will compute the user embedding and compute dot product with all item embeddings. Then it will select Top-K items which final produces recommendations.

## Loss Function & Training Strategy

The objective of the Two-Tower model is to learn user embeddings u and item embeddings v such that positive interactions receive higher matching scores than negative ones.

```math
s(u, v) = uᵀ v
```

I consider two possible loss functions:

### 1 Loss function a: Binary Cross-Entropy Loss

The raw score is mapped to a probability via the sigmoid function:

$$
\hat{y} = \sigma(u^T v) = \frac{1}{1 + e^{-u^T v}}
$$

The Binary Cross-Entropy loss is defined as:


$$
L_{BCE} =
y \log(\sigma(u^T v))
(1 - y)\log(1 - \sigma(u^T v))
$$

### 2 Bayesian Personalized Ranking (BPR) Loss

BPR loss is defined as:

$$
s(u, v^+) > s(u, v^-)
$$

BPR encourages:

- The positive item score to be larger than the negative item score

- Maximization of the margin between positive and negative items

## Training Flow

### Step 1: Prepare Input Batch

For each mini-batch, we sample:

- User IDs

- Item features

- Interaction labels (for BCE)

- (User, positive item, negative item) triples (for BPR)

### Step2: Forward Pass

Compute User Embedding

$$
u = f_u(\text{user_id})
$$

The User Tower maps the user ID into a dense embedding vector.

Compute Item Embedding

$$
v = f_v(\text{item_features})
$$

The Item Tower maps item features into an embedding vector.

### Step3: Compute Matching Score 
For BCE:
$$
s = u^Tv
$$
For BPR:
$$
s^+ = u^Tv^+
$$
$$
s^-=u^Tv^-
$$

### Step4: Compute Loss

Apply sigmoid:

$$
\hat{y} = \sigma(s)
$$

Compute binary cross-entropy:

$$
L = - y \log(\hat{y}) - (1 - y)\log(1 - \hat{y})
$$

BPR Case

For a user $u$, a positive item $v^+$ and a negative item $v^-$:

Compute score difference:

$$
\Delta = s^+ - s^- = u^T v^+ - u^T v^-
$$

Apply sigmoid and log:

$$
L = - \log\left(\sigma(\Delta)\right)
$$

### Step5: Backpropagation

After computing the loss, gradients are calculated with respect to all learnable parameters:

$$
\frac{\partial L}{\partial \theta}
$$

Backpropagation flows through the following components:

1. Dot product similarity:
$$
s = u^T v
$$

2. Item Tower network

3. User Tower network

All parameters are updated using the Adam optimizer:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_{\theta} L
$$

where:
- $\eta$ is the learning rate
- $\nabla_{\theta} L$ is the gradient of the loss with respect to parameters

---

### Step6: Iterative Optimization

The above steps are repeated for:

- Every mini-batch
- Every epoch

Training continues until:

- Convergence of validation loss
- Or reaching the predefined number of epochs


### Summary

```text
Input Batch
   ↓
User Tower → u
Item Tower → v
   ↓
Dot Product
   ↓
Loss Computation
   ↓
Backpropagation
   ↓
Parameter Update
```
## Evaluation Metrics

To evaluate retrieval performance, we use ranking-based metrics:

### 1. Recall@K

$$
Recall@K = \frac{|Relevant \cap TopK|}{|Relevant|}
$$

Measures how many relevant items are retrieved within Top-K.

### 2. HitRate@K

$$
HitRate@K =
\begin{cases}
1 & \text{if at least one relevant item in TopK} \\
0 & \text{otherwise}
\end{cases}
$$

### 3. Precision@K

$$
Precision@K = \frac{|Relevant \cap TopK|}{K}
$$

## Experimental Results

| Metric | K=10   |
|--------|--------|
| Recall@10 | 0.0154 |
| NDCG@10 | 0.0230  |
