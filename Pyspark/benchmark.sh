spark-submit jobs/auto_grade.py \
  --interactions_path data/cb_baseline.parquet \
  --user_vectors_path artifacts/user_vectors.parquet \
  --item_vectors_path artifacts/item_vectors.parquet \
  --k 10 \
  --max_users 2000 \
  --max_items 5000 \
  --metrics_out artifacts/metrics.json