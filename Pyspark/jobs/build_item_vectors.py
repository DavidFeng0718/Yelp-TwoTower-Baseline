# jobs/build_item_vectors.py
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, Normalizer
import argparse, json, os

def main(args):
    spark = SparkSession.builder.appName("build_item_vectors").getOrCreate()

    items = (spark.read.option("header", True).csv(args.items_path)
             .withColumn("business_id", F.col("business_id").cast("string")))

    # 强制所有非 business_id 列为 double
    feature_cols = [c for c in items.columns if c != "business_id"]
    for c in feature_cols:
        items = items.withColumn(c, F.col(c).cast("double"))



    assembler = VectorAssembler(inputCols=feature_cols, outputCol="item_raw")
    vec_df = assembler.transform(items).select("business_id", "item_raw")

    scaler = StandardScaler(inputCol="item_raw", outputCol="item_scaled", withStd=True, withMean=False)
    scaler_model = scaler.fit(vec_df)
    vec_df = scaler_model.transform(vec_df)

    # L2
    norm = Normalizer(inputCol="item_scaled", outputCol="item_vec", p=2.0)
    item_vec_df = norm.transform(vec_df).select("business_id", "item_vec")

    item_vec_df.write.mode("overwrite").parquet(args.out_item_vectors)
    scaler_model.write().overwrite().save(args.out_scaler)

    # 3) 保存 feature_cols
    os.makedirs(os.path.dirname(args.out_feature_cols) or ".", exist_ok=True)
    with open(args.out_feature_cols, "w") as f:
        json.dump(feature_cols, f)

    spark.stop()
    print("Saved item_vectors to:", args.out_item_vectors)
    print("Saved scaler_model to:", args.out_scaler)
    print("Saved feature_cols to:", args.out_feature_cols)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--items_path", default="data/geo_feature_matrix.csv")
    p.add_argument("--out_item_vectors", default="artifacts/item_vectors.parquet")
    p.add_argument("--out_scaler", default="artifacts/item_scaler")
    p.add_argument("--out_feature_cols", default="artifacts/feature_cols.json")
    args = p.parse_args()
    main(args)