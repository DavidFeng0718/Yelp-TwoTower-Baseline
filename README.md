
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

The goal of training is to align user and item embeddings such that:
- Positive pairs have higher similarity scores

So: ```score(u, i⁺) > score(u, i⁻)```
- i⁺ = positively interacted item
- i⁻ = negative item

The model uses a binary objective
```L = BCEWithLogitsLoss(score, label)```

The score before sigmoid is:
```score(u, i) = uᵀ i```
