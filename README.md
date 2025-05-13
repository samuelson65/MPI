-- Step 1: Explode all combinations into claim pairs
WITH base_data AS (
  SELECT
    readmission_id,
    initial_claim_id,
    readmission_claim_id,
    initial_dos,
    readmission_dos
  FROM claims_table
),

-- Step 2: Join base with itself to find intermediate claims
intermediate_flags AS (
  SELECT
    a.readmission_id,
    a.initial_claim_id AS claim_A,
    a.readmission_claim_id AS claim_C,
    a.initial_dos AS dos_A,
    a.readmission_dos AS dos_C,
    b.readmission_claim_id AS claim_B
  FROM base_data a
  JOIN base_data b
    ON a.readmission_id = b.readmission_id
    AND a.initial_claim_id = b.initial_claim_id
    AND a.readmission_claim_id != b.readmission_claim_id
    AND b.readmission_dos > a.initial_dos
    AND b.readmission_dos < a.readmission_dos
),

-- Step 3: Final flag
flagged_data AS (
  SELECT DISTINCT
    concat(claim_A, '_', claim_C) AS pk,
    1 AS has_intermediate_flag
  FROM intermediate_flags
)

-- Step 4: Merge with original
SELECT
  c.*,
  COALESCE(f.has_intermediate_flag, 0) AS has_intermediate_flag
FROM claims_table c
LEFT JOIN flagged_data f
  ON c.pk = f.pk;
