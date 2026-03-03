
import argparse
import json
from typing import Optional

from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.functions import vector_to_array


def pick_col(cols, prefer: Optional[str], fallback: str) -> str:
    s = set(cols)
    if prefer and prefer in s:
        return prefer
    if fallback in s:
        return fallback
    raise ValueError(f"Cannot find column. prefer={prefer!r}, fallback={fallback!r}, available={sorted(cols)}")


def pick_time_col(cols) -> str:
    # Common variants across projects
    candidates = [
        "review_time", "review_ts", "timestamp", "ts",
        "review_date", "date", "created_at", "time"
    ]
    s = set(cols)
    for c in candidates:
        if c in s:
            return c
    raise ValueError(f"Cannot find a time column among {candidates}. Available={sorted(cols)}")


def dot_udf_factory():
    from pyspark.sql.types import DoubleType

    def dot(a, b):
        if a is None or b is None:
            return None
        # a, b are python lists from vector_to_array
        # defensive: handle unequal lengths
        n = min(len(a), len(b))
        s = 0.0
        for i in range(n):
            ai = a[i]
            bi = b[i]
            if ai is None or bi is None:
                continue
            s += float(ai) * float(bi)
        return float(s)

    return F.udf(dot, DoubleType())


def ndcg_udf_factory(k: int):
    from pyspark.sql.types import DoubleType

    def ndcg(rec_list, gt_list):
        if not rec_list or not gt_list:
            return 0.0
        gt = set(gt_list)
        import math
        dcg = 0.0
        for i, bid in enumerate(rec_list[:k], start=1):
            if bid in gt:
                dcg += 1.0 / math.log2(i + 1)

        ideal_hits = min(len(gt), k)
        idcg = 0.0
        for i in range(ideal_hits):
            idcg += 1.0 / math.log2(i + 2)
        return float(dcg / idcg) if idcg > 0 else 0.0

    return F.udf(ndcg, DoubleType())


def recall_udf_factory(k: int):
    from pyspark.sql.types import DoubleType

    def recall(rec_list, gt_list):
        if not rec_list or not gt_list:
            return 0.0
        rec = rec_list[:k]
        gt = set(gt_list)
        hit = 0
        for bid in rec:
            if bid in gt:
                hit += 1
        return float(hit / len(gt)) if len(gt) > 0 else 0.0

    return F.udf(recall, DoubleType())


def hitrate_udf_factory(k: int):
    from pyspark.sql.types import DoubleType

    def hitrate(rec_list, gt_list):
        if not rec_list or not gt_list:
            return 0.0
        gt = set(gt_list)
        for bid in rec_list[:k]:
            if bid in gt:
                return 1.0
        return 0.0

    return F.udf(hitrate, DoubleType())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactions_path", default="data/cb_baseline.parquet")
    ap.add_argument("--user_vectors_path", default="artifacts/user_vectors.parquet")
    ap.add_argument("--item_vectors_path", default="artifacts/item_vectors.parquet")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--split_ratio", type=float, default=0.8, help="Quantile for time split; <= this is history, > this is future")
    ap.add_argument("--pos_threshold", type=float, default=0.0, help="centered_rating > pos_threshold is positive")
    ap.add_argument("--user_vec_col", default=None, help="Preferred user vector column name (optional)")
    ap.add_argument("--item_vec_col", default=None, help="Preferred item vector column name (optional)")
    ap.add_argument("--max_users", type=int, default=0, help="If >0, cap number of users evaluated (random sample)")
    ap.add_argument("--max_items", type=int, default=0, help="If >0, cap number of candidate items (head after shuffle)")
    ap.add_argument("--item_sample_frac", type=float, default=1.0, help="If <1, sample candidate items for grading")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--broadcast_items", action="store_true", help="Broadcast item table (useful when items are small)")
    ap.add_argument("--metrics_out", default="", help="Optional path to write metrics JSON (e.g., artifacts/metrics.json)")
    args = ap.parse_args()

    spark = SparkSession.builder.appName("auto_grade").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    k = args.k

    # -------- Load interactions & compute split time --------
    inter = spark.read.parquet(args.interactions_path)
    time_col = pick_time_col(inter.columns)
    if "centered_rating" not in inter.columns:
        raise ValueError("interactions must contain centered_rating")
    if "user_id" not in inter.columns or "business_id" not in inter.columns:
        raise ValueError("interactions must contain user_id and business_id")

    # Normalize time to a numeric seconds column for quantile split
    # Normalize time to epoch seconds safely (works for TIMESTAMP_NTZ / TIMESTAMP / STRING)
    t = inter.withColumn(
        "__ts",
        F.unix_timestamp(F.col(time_col).cast("string"))
    ).filter(F.col("__ts").isNotNull())

    split_ts = t.approxQuantile("__ts", [args.split_ratio], 0.0)[0]

    # History & future
    hist = t.filter(F.col("__ts") <= F.lit(split_ts))
    fut = t.filter(F.col("__ts") > F.lit(split_ts))

    # Ground truth: future positives
    gt = (
        fut.filter(F.col("centered_rating") > F.lit(args.pos_threshold))
        .select("user_id", "business_id")
        .groupBy("user_id")
        .agg(F.collect_set("business_id").alias("gt_list"))
        .filter(F.size("gt_list") > 0)
    )

    # -------- Load vectors --------
    users = spark.read.parquet(args.user_vectors_path)
    items = spark.read.parquet(args.item_vectors_path)

    uvec = pick_col(users.columns, args.user_vec_col, "user_vec")
    ivec = pick_col(items.columns, args.item_vec_col, "item_vec")

    if "user_id" not in users.columns:
        raise ValueError("user_vectors must contain user_id")
    if "business_id" not in items.columns:
        raise ValueError("item_vectors must contain business_id")

    users = users.select("user_id", F.col(uvec).alias("user_vec"))
    items = items.select("business_id", F.col(ivec).alias("item_vec"))

    # Candidate pool controls (VERY IMPORTANT for exact evaluation)
    if args.item_sample_frac < 1.0:
        items = items.sample(withReplacement=False, fraction=args.item_sample_frac, seed=args.seed)
    if args.max_items and args.max_items > 0:
        items = items.orderBy(F.rand(args.seed)).limit(int(args.max_items))

    if args.broadcast_items:
        items = F.broadcast(items)

    # Only evaluate users that have ground truth AND user vectors
    eval_users = users.join(gt.select("user_id"), on="user_id", how="inner")

    if args.max_users and args.max_users > 0:
        eval_users = eval_users.orderBy(F.rand(args.seed)).limit(int(args.max_users))

    # -------- Exact retrieval (cosine ~= dot due to L2 normalization) --------
    dot_udf = dot_udf_factory()

    # Convert Spark ML vectors to arrays
    eval_users_arr = eval_users.withColumn("uarr", vector_to_array("user_vec")).select("user_id", "uarr")
    items_arr = items.withColumn("iarr", vector_to_array("item_vec")).select("business_id", "iarr")

    scored = (
        eval_users_arr.crossJoin(items_arr)
        .withColumn("score", dot_udf(F.col("uarr"), F.col("iarr")))
    )

    w = Window.partitionBy("user_id").orderBy(F.desc("score"))
    topk = (
        scored.withColumn("rank", F.row_number().over(w))
        .filter(F.col("rank") <= F.lit(k))
        .select("user_id", "business_id", "rank")
    )

    # Aggregate to ordered recommendation list
    rec = (
        topk.groupBy("user_id")
        .agg(F.sort_array(F.collect_list(F.struct("rank", "business_id"))).alias("tmp"))
        .select("user_id", F.expr("transform(tmp, x -> x.business_id) as rec_list"))
    )

    # -------- Metrics --------
    ndcg_udf = ndcg_udf_factory(k)
    recall_udf = recall_udf_factory(k)
    hit_udf = hitrate_udf_factory(k)

    eval_df = (
        rec.join(gt, on="user_id", how="inner")
        .withColumn("ndcg", ndcg_udf("rec_list", "gt_list"))
        .withColumn("recall", recall_udf("rec_list", "gt_list"))
        .withColumn("hitrate", hit_udf("rec_list", "gt_list"))
    )

    agg = eval_df.agg(
        F.count("*").alias("users_evaluated"),
        F.avg("ndcg").alias(f"ndcg@{k}"),
        F.avg("recall").alias(f"recall@{k}"),
        F.avg("hitrate").alias(f"hitrate@{k}"),
    ).collect()[0].asDict()

    # Add split info
    agg["split_ratio"] = float(args.split_ratio)
    agg["split_ts"] = int(split_ts)
    agg["pos_threshold"] = float(args.pos_threshold)
    agg["item_sample_frac"] = float(args.item_sample_frac)
    agg["max_items"] = int(args.max_items) if args.max_items else 0
    agg["max_users"] = int(args.max_users) if args.max_users else 0

    # Pretty print
    print("\n=== AUTO GRADE RESULTS ===")
    print(f"Users evaluated : {agg['users_evaluated']}")
    print(f"NDCG@{k}         : {agg[f'ndcg@{k}']:.6f}")
    print(f"Recall@{k}       : {agg[f'recall@{k}']:.6f}")
    print(f"HitRate@{k}      : {agg[f'hitrate@{k}']:.6f}")
    print(f"Split (ts)       : <= {agg['split_ts']} is history, > is future")
    print("==========================\n")

    # Optional save
    if args.metrics_out:
        out = args.metrics_out
        payload = {
            "k": k,
            "metrics": agg,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Wrote metrics JSON to: {out}")

    spark.stop()


if __name__ == "__main__":
    main()
