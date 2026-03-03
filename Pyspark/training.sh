spark-submit jobs/build_item_vectors.py \
  --items_path data/geo_feature_matrix.csv \
  --out_item_vectors artifacts/item_vectors.parquet \
  --out_scaler artifacts/item_scaler

spark-submit jobs/build_user_vectors.py \
  --interactions_path data/cb_baseline.parquet \
  --item_vectors_path artifacts/item_vectors.parquet \
  --out_user_vectors artifacts/user_vectors.parquet \
  --pos_threshold 0.0