# ============================================
# CPT AUDIT RECOMMENDATION ENGINE (PYSPARK)
# ============================================

from pyspark.sql import functions as F
from pyspark.sql.types import *

# --------------------------------------------
# PARAMETERS (TUNE THESE)
# --------------------------------------------

MIN_VOLUME = 20         # minimum CPT occurrences
ALPHA = 20              # Bayesian smoothing strength
HIGH_RISK = 0.6
MEDIUM_RISK = 0.4

# --------------------------------------------
# STEP 1: CLEAN INPUT
# --------------------------------------------

training_df = training_df.filter(F.col("cpt_findingline_dict").isNotNull())
scoring_df = scoring_df.filter(F.col("cpt_findingline_dict").isNotNull())

# --------------------------------------------
# STEP 2: FLATTEN TRAINING DATA
# --------------------------------------------

train_flat = (
    training_df
    .select(
        F.explode(F.map_entries("cpt_findingline_dict")).alias("kv")
    )
    .select(
        F.col("kv.key").alias("CPT_Code"),
        F.col("kv.value").cast("int").alias("line_denied")
    )
)

# --------------------------------------------
# STEP 3: GLOBAL DROP RATE
# --------------------------------------------

global_stats = train_flat.agg(
    F.sum("line_denied").alias("total_drops"),
    F.count("*").alias("total_count")
).collect()[0]

global_rate = (
    global_stats["total_drops"] / global_stats["total_count"]
    if global_stats["total_count"] > 0 else 0.0
)

# --------------------------------------------
# STEP 4: CPT-LEVEL BAYESIAN PROBABILITY
# --------------------------------------------

agg_df = (
    train_flat
    .groupBy("CPT_Code")
    .agg(
        F.count("*").alias("N"),
        F.sum("line_denied").alias("D")
    )
    .withColumn(
        "Drop_Probability",
        (F.col("D") + F.lit(ALPHA * global_rate)) /
        (F.col("N") + F.lit(ALPHA))
    )
)

# --------------------------------------------
# STEP 5: FILTER LOW VOLUME CPTs
# --------------------------------------------

agg_filtered = agg_df.filter(F.col("N") >= MIN_VOLUME)

# --------------------------------------------
# STEP 6: FLATTEN SCORING DATA
# --------------------------------------------

score_flat = (
    scoring_df
    .select(
        "claim_id",
        F.explode(F.map_entries("cpt_findingline_dict")).alias("kv")
    )
    .select(
        "claim_id",
        F.col("kv.key").alias("CPT_Code")
    )
)

# --------------------------------------------
# STEP 7: JOIN WITH CPT PROBABILITY
# --------------------------------------------

score_with_prob = (
    score_flat
    .join(agg_filtered.select("CPT_Code", "Drop_Probability"),
          on="CPT_Code", how="left")
    .withColumn(
        "Drop_Probability",
        F.coalesce(F.col("Drop_Probability"), F.lit(global_rate))
    )
)

# --------------------------------------------
# STEP 8: ADD RISK BUCKETS
# --------------------------------------------

score_with_prob = score_with_prob.withColumn(
    "risk_bucket",
    F.when(F.col("Drop_Probability") >= HIGH_RISK, "HIGH")
     .when(F.col("Drop_Probability") >= MEDIUM_RISK, "MEDIUM")
     .otherwise("LOW")
)

# --------------------------------------------
# STEP 9: BUILD CPT → {prob, risk} MAP
# --------------------------------------------

cpt_map = (
    score_with_prob
    .groupBy("claim_id")
    .agg(
        F.map_from_entries(
            F.collect_list(
                F.struct(
                    F.col("CPT_Code"),
                    F.struct(
                        F.col("Drop_Probability").alias("prob"),
                        F.col("risk_bucket").alias("risk")
                    )
                )
            )
        ).alias("cpt_audit_recommendation")
    )
)

# --------------------------------------------
# STEP 10: CLAIM-LEVEL METRICS
# --------------------------------------------

claim_metrics = (
    score_with_prob
    .groupBy("claim_id")
    .agg(
        F.max("Drop_Probability").alias("claim_max_risk"),
        F.avg("Drop_Probability").alias("claim_avg_risk")
    )
)

# --------------------------------------------
# STEP 11: FINAL OUTPUT
# --------------------------------------------

final_df = (
    scoring_df
    .join(cpt_map, on="claim_id", how="left")
    .join(claim_metrics, on="claim_id", how="left")
    .select(
        "claim_id",
        "findings_status",
        "cpt_findingline_dict",
        "cpt_audit_recommendation",
        "claim_max_risk",
        "claim_avg_risk"
    )
)

# --------------------------------------------
# OUTPUT
# --------------------------------------------

final_df.show(truncate=False)
