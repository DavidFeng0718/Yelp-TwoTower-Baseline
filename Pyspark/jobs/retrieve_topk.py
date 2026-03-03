# jobs/retrieve_topk.py
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import BucketedRandomProjectionLSH
import argparse

def dot_udf():
    from pyspark.sql.types import DoubleType
    from pyspark.ml.linalg import DenseVector, SparseVector

    def dot(a, b):
        # a,b are Spark vectors
        if a is None or b is None:
            return None
        return float(a.dot(b))
    return F.udf(dot, DoubleType())

def main(args):
    spark = SparkSession.builder.appName("retrieve_topk").getOrCreate()

    item_vec_df = spark.read.parquet(args.item_vectors_path)
    user_vec_df = spark.read.parquet(args.user_vectors_path)

    urow = user_vec_df.filter(F.col("user_id") == args.user_id).select("user_vec").head()
    if urow is None:
        raise ValueError(f"user_id not found: {args.user_id}")
    u_vec = urow["user_vec"]

    if args.mode == "exact":
        # exact dot-product ranking (item_vec and user_vec are L2-normalized => dot ~ cosine)
        u_df = spark.createDataFrame([(args.user_id, u_vec)], ["user_id", "user_vec"])
        cand = item_vec_df.crossJoin(u_df)

        score_fn = dot_udf()
        scored = cand.withColumn("score", score_fn(F.col("item_vec"), F.col("user_vec")))
        topk = scored.orderBy(F.col("score").desc()).select("business_id", "score").limit(args.k)
        topk.show(truncate=False)

    elif args.mode == "lsh":
        # build / load LSH model
        if args.lsh_model_path:
            from pyspark.ml.feature import BucketedRandomProjectionLSHModel
            lsh_model = BucketedRandomProjectionLSHModel.load(args.lsh_model_path)
        else:
            lsh = BucketedRandomProjectionLSH(
                inputCol="item_vec",
                outputCol="hashes",
                bucketLength=args.bucketLength,
                numHashTables=args.numHashTables
            )
            lsh_model = lsh.fit(item_vec_df)

        # returns nearest neighbors by Euclidean distance on normalized vectors
        nn = lsh_model.approxNearestNeighbors(item_vec_df, u_vec, args.k) \
                      .select("business_id", F.col("distCol").alias("distance"))
        nn.show(truncate=False)

    else:
        raise ValueError("mode must be exact or lsh")

    spark.stop()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--item_vectors_path", default="artifacts/item_vectors.parquet")
    p.add_argument("--user_vectors_path", default="artifacts/user_vectors.parquet")
    p.add_argument("--user_id", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--mode", choices=["exact", "lsh"], default="exact")

    # lsh params (only used when mode=lsh)
    p.add_argument("--lsh_model_path", default="")
    p.add_argument("--bucketLength", type=float, default=2.0)
    p.add_argument("--numHashTables", type=int, default=4)

    args = p.parse_args()
    main(args)