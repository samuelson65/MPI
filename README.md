SELECT
  curr.pk,
  curr.provider_id,
  curr.member_id,
  curr.ref_claim,
  curr.read_claim,
  CASE
    WHEN int.has_interim = 1 AND mat.has_matching = 1 THEN 'Both'
    WHEN int.has_interim = 1 THEN 'Interim'
    WHEN mat.has_matching = 1 THEN 'Matching'
    ELSE 'None'
  END AS claim_flag
FROM pairs_table curr
-- Check for interim claims
LEFT JOIN (
  SELECT DISTINCT p1.pk, 1 AS has_interim
  FROM pairs_table p1
  JOIN pairs_table p2 ON p1.provider_id = p2.provider_id
                      AND p1.member_id = p2.member_id
                      AND p1.pk <> p2.pk
  WHERE
    (
      -- p2's ref or readm dates are BETWEEN p1's ref_ldos and readm_fdos
      (p2.ref_fdos BETWEEN p1.ref_ldos AND p1.readm_fdos)
      OR (p2.ref_ldos BETWEEN p1.ref_ldos AND p1.readm_fdos)
      OR (p2.readm_fdos BETWEEN p1.ref_ldos AND p1.readm_fdos)
      OR (p2.readm_ldos BETWEEN p1.ref_ldos AND p1.readm_fdos)
    )
) int ON curr.pk = int.pk

-- Check for exact date match with other pk's
LEFT JOIN (
  SELECT DISTINCT p1.pk, 1 AS has_matching
  FROM pairs_table p1
  JOIN pairs_table p2 ON p1.provider_id = p2.provider_id
                      AND p1.member_id = p2.member_id
                      AND p1.pk <> p2.pk
  WHERE
    (
      (p1.ref_fdos = p2.ref_fdos AND p1.ref_ldos = p2.ref_ldos)
      OR (p1.readm_fdos = p2.readm_fdos AND p1.readm_ldos = p2.readm_ldos)
    )
) mat ON curr.pk = mat.pk
;
