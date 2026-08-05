#!/usr/bin/env python3
"""
export_for_dashboard.py
------------------------
Dump the three tables analytics.html needs into Parquet.

Parquet is the recommended hand-off because its format is stable across DuckDB
versions, whereas a raw .duckdb file must match the storage version compiled
into DuckDB WASM — a mismatch is the single most common reason the dashboard
refuses to attach a database directly.

    python export_for_dashboard.py --db transparency.duckdb --out ./web

Then serve the folder and pick the three .parquet files in the dashboard's
"Connect a data source" dialog:

    python -m http.server 8000
"""

import argparse
from pathlib import Path

import duckdb

TABLES = ["negotiated_rates", "payers", "billing_codes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="transparency.duckdb")
    ap.add_argument("--out", default=".", help="Directory to write .parquet files into")
    ap.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "uncompressed"])
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.db, read_only=True)

    present = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    missing = [t for t in TABLES if t not in present]
    if missing:
        raise SystemExit(
            f"{args.db} is missing table(s): {', '.join(missing)}.\n"
            f"Found: {', '.join(sorted(present)) or '(none)'}\n"
            "Run stream_parser.py first to populate the database."
        )

    total = 0
    for t in TABLES:
        dest = out / f"{t}.parquet"
        con.execute(
            f"COPY (SELECT * FROM {t}) TO '{dest.as_posix()}' "
            f"(FORMAT PARQUET, COMPRESSION '{args.compression}')"
        )
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        size = dest.stat().st_size / 1024
        total += n
        print(f"  {t:20s} {n:>10,} rows  →  {dest.name}  ({size:,.0f} KB)")

    print(f"\n{total:,} rows exported to {out.resolve()}")
    print("Load all three files together in the dashboard's data-source dialog.")


if __name__ == "__main__":
    main()
