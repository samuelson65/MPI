SELECT
  p.pk,
  p.provider_id,
  p.member_id,
  p.ref_claim,
  p.read_claim,
  CASE
    WHEN ic.has_interim = 1 AND mp.is_matching = 1 THEN 'Both'
    WHEN ic.has_interim = 1 THEN 'Interim'
    WHEN mp.is_matching = 1 THEN 'Matching'
    ELSE 'None'
  END AS claim_flag
FROM
  pairs_table p
LEFT JOIN (
  -- Subquery for interim claims
  SELECT
    pk,
    1 AS has_interim
  FROM
    pairs_table pt
  WHERE EXISTS (
    SELECT 1
    FROM claims c
    WHERE
      c.provider_id = pt.provider_id
      AND c.member_id = pt.member_id
      AND (
        (c.fdos BETWEEN pt.ref_ldos AND pt.readm_fdos)
        OR (c.ldos BETWEEN pt.ref_ldos AND pt.readm_fdos)
      )
      AND c.claim_id NOT IN (pt.ref_claim, pt.read_claim)
  )
) ic ON p.pk = ic.pk
LEFT JOIN (
  -- Subquery for matching fdos/ldos
  SELECT
    pk,
    1 AS is_matching
  FROM
    pairs_table pt
  WHERE EXISTS (
    SELECT 1
    FROM pairs_table p2
    WHERE
      p2.provider_id = pt.provider_id
      AND p2.member_id = pt.member_id
      AND p2.pk <> pt.pk
      AND (
        (p2.ref_fdos = pt.ref_fdos AND p2.ref_ldos = pt.ref_ldos)
        OR (p2.readm_fdos = pt.readm_fdos AND p2.readm_ldos = pt.readm_ldos)
      )
  )
) mp ON p.pk = mp.pk
;
