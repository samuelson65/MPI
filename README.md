SELECT
  curr.pk,
  curr.provider_id,
  curr.member_id,
  curr.ref_claim,
  curr.read_claim,
  CASE
    WHEN interim_flag = 1 AND matching_flag = 1 THEN 'Both'
    WHEN interim_flag = 1 THEN 'Interim'
    WHEN matching_flag = 1 THEN 'Matching'
    ELSE 'None'
  END AS claim_flag
FROM (
  SELECT
    curr.pk,
    curr.provider_id,
    curr.member_id,
    curr.ref_claim,
    curr.read_claim,

    -- Flag for interim claims
    CASE
      WHEN EXISTS (
        SELECT 1
        FROM pairs_table other
        WHERE
          other.pk <> curr.pk
          AND other.provider_id = curr.provider_id
          AND other.member_id = curr.member_id
          AND (
            (other.ref_fdos BETWEEN curr.ref_ldos AND curr.readm_fdos) OR
            (other.ref_ldos BETWEEN curr.ref_ldos AND curr.readm_fdos) OR
            (other.readm_fdos BETWEEN curr.ref_ldos AND curr.readm_fdos) OR
            (other.readm_ldos BETWEEN curr.ref_ldos AND curr.readm_fdos)
          )
      ) THEN 1 ELSE 0
    END AS interim_flag,

    -- Flag for ref/readm claims with exact same fdos/ldos
    CASE
      WHEN EXISTS (
        SELECT 1
        FROM pairs_table other
        WHERE
          other.pk <> curr.pk
          AND other.provider_id = curr.provider_id
          AND other.member_id = curr.member_id
          AND (
            (other.ref_fdos = curr.ref_fdos AND other.ref_ldos = curr.ref_ldos) OR
            (other.readm_fdos = curr.readm_fdos AND other.readm_ldos = curr.readm_ldos)
          )
      ) THEN 1 ELSE 0
    END AS matching_flag

  FROM pairs_table curr
) final;
