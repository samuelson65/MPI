# demo_run_rules.py
# Minimal demonstration on synthetic data.
# Replace the synthetic generator with your real claims dataframe (df).

import numpy as np
import pandas as pd
from drg_ruleminer import RuleMiner

rng = np.random.default_rng(7)
n = 5000

df = pd.DataFrame({
    "mcc_count": rng.poisson(1.2, size=n),
    "cc_count": rng.poisson(1.8, size=n),
    "procedure_count": rng.poisson(2.0, size=n),
    "length_of_stay": rng.integers(1, 12, size=n),
    "discharge_status": rng.choice(["Home","SNF","Expired","AMA", "Other", None], size=n),
})

# Create a synthetic target
overpay_signal = (
    (df["mcc_count"] >= 3).astype(int) +
    ((df["length_of_stay"] <= 4) & (df["cc_count"] >= 1)).astype(int) +
    ((df["discharge_status"].fillna("Missing").isin(["AMA","Expired"])) & (df["procedure_count"] >= 2)).astype(int)
)
y = (overpay_signal >= 2).astype(int)
df["label"] = np.where(y==1, "overpayment", "nofindings")

rm = RuleMiner(target_col="label", pos_label="overpayment", min_precision=0.7, verbose=True)
result = rm.fit(df)

print("=== Policy metrics (TEST set) ===")
for k,v in result.metrics.items():
    print(f"{k}: {v}")

print("\n=== Combined Policy SQL ===")
print(result.policy_sql)

print("\n=== Top 10 Candidate Rules ===")
print(result.rules_df.head(10))
