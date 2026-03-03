# jobs/build_user_vectors.py
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.stat import Summarizer
from pyspark.ml.feature import Normalizer
import argparse

def main(args):
    spark = SparkSession.builder.appName("build_user_vectors").getOrCreate()

    inter = (spark.read.parquet(args.interactions_path)
             .select("user_id", "business_id", "centered_rating", "review_date")
             .withColumn("user_id", F.col("user_id").cast("string"))
             .withColumn("business_id", F.col("business_id").cast("string")))

    # label: centered_rating > 0
    inter = inter.withColumn("label", (F.col("centered_rating") > F.lit(args.pos_threshold)).cast("int"))

    # time split
    # parse time
    inter = inter.withColumn("review_ts", F.to_timestamp("review_date"))
    inter = inter.filter(F.col("review_ts").isNotNull())

    # convert timestamp -> long seconds for approxQuantile
    inter = inter.withColumn("review_sec", F.col("review_ts").cast("long"))

    cutoff_sec = inter.approxQuantile(
        "review_sec",
        [args.time_quantile],
        args.quantile_rel_error
    )[0]

    train = inter.filter(F.col("review_sec") <= F.lit(cutoff_sec))
    # test  = inter.filter(F.col("review_sec") >  F.lit(cutoff_sec))
    # test = inter.filter(F.col("review_date") > F.lit(cutoff))  # 评估可再用

    # positive interactions in train
    pos_train = train.filter(F.col("label") == 1).select("user_id", "business_id")

    item_vec_df = spark.read.parquet(args.item_vectors_path)

    pos_with_vec = pos_train.join(item_vec_df, on="business_id", how="inner")

    # user_vec = mean(item_vec)
    user_vec_df = (pos_with_vec
        .groupBy("user_id")
        .agg(Summarizer.mean(F.col("item_vec")).alias("user_vec_raw"))
        .select("user_id", F.col("user_vec_raw").alias("user_vec"))
    )

    # L2 normalize user vector
    norm = Normalizer(inputCol="user_vec", outputCol="user_vec_norm", p=2.0)
    user_vec_df = norm.transform(user_vec_df).select("user_id", F.col("user_vec_norm").alias("user_vec"))

    user_vec_df.write.mode("overwrite").parquet(args.out_user_vectors)

    spark.stop()

    print("Saved user_vectors to:", args.out_user_vectors)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interactions_path", default="data/cb_baseline.parquet")
    p.add_argument("--item_vectors_path", default="artifacts/item_vectors.parquet")
    p.add_argument("--out_user_vectors", default="artifacts/user_vectors.parquet")
    p.add_argument("--pos_threshold", type=float, default=0.0)
    p.add_argument("--time_quantile", type=float, default=0.8)
    p.add_argument("--quantile_rel_error", type=float, default=0.01)
    args = p.parse_args()
    main(args)