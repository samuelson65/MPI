# ============================================
# PYSPARK SCRIPT: CPT DENIAL PROBABILITY MODEL
# ============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, explode, map_keys, map_values, map_entries,
    count, sum as spark_sum, when, lit, coalesce,
    expr, collect_list, struct, map_from_entries
)
from pyspark.sql.types import DoubleType

# --------------------------------------------
# INPUTS (ASSUMED)
# --------------------------------------------
# training_df schema:
# claim_id (optional)
# cpt_findingline_dict (map<string,int>)
# findings_status (string)

# scoring_df schema:
# claim_id
# cpt_findingline_dict (map<string,int>)
# findings_status (optional)

# --------------------------------------------
# STEP 1: FLATTEN TRAINING DATA
# --------------------------------------------

train_flat = (
    training_df
    .select(
        explode(map_entries(col("cpt_findingline_dict"))).alias("kv")
    )
    .select(
        col("kv.key").alias("CPT_Code"),
        col("kv.value").alias("line_denied")
    )
)

# --------------------------------------------
# STEP 2: AGGREGATE CPT METRICS
# --------------------------------------------

agg_df = (
    train_flat
    .groupBy("CPT_Code")
    .agg(
        count("*").alias("Total_Occurrences"),
        spark_sum("line_denied").alias("Total_Drops")
    )
    .withColumn(
        "Drop_Probability",
        col("Total_Drops") / col("Total_Occurrences")
    )
)

# --------------------------------------------
# STEP 3: FILTER LOW VOLUME CPTs
# --------------------------------------------

MIN_VOLUME = 30

agg_filtered = agg_df.filter(col("Total_Occurrences") >= MIN_VOLUME)

# --------------------------------------------
# STEP 4: GLOBAL FALLBACK
# --------------------------------------------

global_avg = agg_filtered.agg(
    {"Drop_Probability": "avg"}
).collect()[0][0]

# Broadcast CPT probability map
cpt_prob_df = agg_filtered.select("CPT_Code", "Drop_Probability")

# --------------------------------------------
# STEP 5: FLATTEN SCORING DATA
# --------------------------------------------

score_flat = (
    scoring_df
    .select(
        col("claim_id"),
        explode(map_entries(col("cpt_findingline_dict"))).alias("kv")
    )
    .select(
        col("claim_id"),
        col("kv.key").alias("CPT_Code")
    )
)

# --------------------------------------------
# STEP 6: JOIN WITH CPT PROBABILITIES
# --------------------------------------------

score_with_prob = (
    score_flat
    .join(cpt_prob_df, on="CPT_Code", how="left")
    .withColumn(
        "Drop_Probability",
        coalesce(col("Drop_Probability"), lit(global_avg))
    )
)

# --------------------------------------------
# STEP 7: REBUILD CPT → PROBABILITY MAP
# --------------------------------------------

score_map = (
    score_with_prob
    .groupBy("claim_id")
    .agg(
        map_from_entries(
            collect_list(
                struct(
                    col("CPT_Code"),
                    col("Drop_Probability").cast(DoubleType())
                )
            )
        ).alias("cpt_drop_probability")
    )
)

# --------------------------------------------
# STEP 8: FINAL CLAIM-LEVEL OUTPUT
# --------------------------------------------

final_df = (
    scoring_df
    .join(score_map, on="claim_id", how="left")
    .select(
        "claim_id",
        "findings_status",
        "cpt_findingline_dict",
        "cpt_drop_probability"
    )
)

# --------------------------------------------
# OPTIONAL: ADD CLAIM-LEVEL RISK METRICS
# --------------------------------------------

final_df = final_df.withColumn(
    "claim_max_risk",
    expr("aggregate(map_values(cpt_drop_probability), 0D, (acc, x) -> greatest(acc, x))")
).withColumn(
    "claim_avg_risk",
    expr("aggregate(map_values(cpt_drop_probability), 0D, (acc, x) -> acc + x) / size(cpt_drop_probability)")
)

# --------------------------------------------
# OUTPUT
# --------------------------------------------

final_df.show(truncate=False)
