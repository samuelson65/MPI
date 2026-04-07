#!/usr/bin/env python3
"""
Medical Code Weighted Similarity Tool
======================================
Uses the FWA embedding CSV to:
  1. Compare two codes and get a weighted similarity score
  2. Input one code and get top-N nearest matches

Axis blocks (each 10 dims):
  D001-D010  Clinical Domain
  D011-D020  Severity / CC-MCC
  D021-D030  Service Intensity
  D031-D040  Anatomical Site
  D041-D050  Episode Type
  D051-D060  Billing Channel
  D061-D070  Bundling Cohesion
  D071-D080  DX-Proc Link
  D081-D090  FWA Risk Signal
  D091-D100  DRG / RVU Weight

Usage:
  python medical_similarity.py --csv embeddings.csv --compare I21.0 99232
  python medical_similarity.py --csv embeddings.csv --find I21.0 --topn 10
  python medical_similarity.py --csv embeddings.csv --find I21.0 --topn 10 --weights severity:2.0 fwa:3.0
  python medical_similarity.py --csv embeddings.csv  # interactive menu
"""

import numpy as np
import csv
import sys
import os
import argparse
import time
from typing import Optional

# ── Default axis weights (1.0 = equal importance) ───────────────────────────
DEFAULT_WEIGHTS = {
    "clinical":    1.0,   # D001-D010
    "severity":    1.0,   # D011-D020
    "intensity":   1.0,   # D021-D030
    "anatomical":  1.0,   # D031-D040
    "temporal":    1.0,   # D041-D050
    "billing":     1.0,   # D051-D060
    "bundling":    1.0,   # D061-D070
    "dxproc":      1.0,   # D071-D080
    "fwa":         1.0,   # D081-D090
    "drgrvu":      1.0,   # D091-D100
}

AXIS_RANGES = {
    "clinical":   (0,  10),
    "severity":   (10, 20),
    "intensity":  (20, 30),
    "anatomical": (30, 40),
    "temporal":   (40, 50),
    "billing":    (50, 60),
    "bundling":   (60, 70),
    "dxproc":     (70, 80),
    "fwa":        (80, 90),
    "drgrvu":     (90, 100),
}

AXIS_DESCRIPTIONS = {
    "clinical":   "Clinical Domain (Cardio/Neuro/Ortho/GI/Pulm/Metabolic/Renal/MH/Infect/Prev)",
    "severity":   "Severity / CC-MCC tier",
    "intensity":  "Service Intensity / E&M level / Procedure complexity",
    "anatomical": "Anatomical Site (Head/Chest/Spine/Abdomen/Extremity/Skin)",
    "temporal":   "Episode Type (Acute/Chronic/Post-op/Preventive)",
    "billing":    "Billing Channel (Inpatient/DME/Outpatient)",
    "bundling":   "Bundling Cohesion (component vs comprehensive)",
    "dxproc":     "DX-Procedure Medical Necessity Link",
    "fwa":        "FWA Risk Signal (Upcoding/Unbundling/Over-order/Phantom)",
    "drgrvu":     "DRG / RVU Financial Weight",
}

DIM = 100
EMBEDDINGS_CSV = os.environ.get(
    "EMBEDDINGS_CSV",
    "/mnt/user-data/outputs/Medical_Code_FWA_Embeddings_100k.csv"
)


# ── Embedding store ──────────────────────────────────────────────────────────

class EmbeddingStore:
    def __init__(self):
        self.codes  = []          # list of code strings
        self.types  = []          # list of type strings
        self.descs  = []          # list of descriptions
        self.matrix = None        # np.ndarray (N, 100)
        self.index  = {}          # code -> row index

    def load(self, path: str):
        if not os.path.exists(path):
            print(f"[ERROR] CSV not found: {path}")
            print("  Set EMBEDDINGS_CSV env var or use --csv flag.")
            sys.exit(1)

        print(f"Loading embeddings from {path} ...", flush=True)
        t0 = time.time()

        rows_codes, rows_types, rows_descs, rows_vecs = [], [], [], []
        with open(path, "r", newline="", buffering=1024*1024*16) as f:
            reader = csv.reader(f)
            header = next(reader)
            # Find where D001 starts
            try:
                dim_start = header.index("D001")
            except ValueError:
                # fallback: assume cols 0=Code,1=Type,2=Desc, dims start at 4 or 7
                dim_start = 7 if len(header) > 107 else 4

            for row in reader:
                if len(row) < dim_start + DIM:
                    continue
                code = row[0].strip().upper()
                rows_codes.append(code)
                rows_types.append(row[1])
                rows_descs.append(row[2])
                rows_vecs.append(row[dim_start:dim_start + DIM])

        self.codes  = rows_codes
        self.types  = rows_types
        self.descs  = rows_descs
        self.matrix = np.array(rows_vecs, dtype=np.float32)

        # L2-normalise rows (should already be, but ensure)
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.matrix /= norms

        # Build lookup index
        self.index = {c: i for i, c in enumerate(self.codes)}

        elapsed = time.time() - t0
        print(f"  Loaded {len(self.codes):,} codes in {elapsed:.1f}s\n")

    def get_vector(self, code: str) -> Optional[tuple]:
        """Return (index, vector) or None."""
        code = code.strip().upper()
        # Exact match
        if code in self.index:
            i = self.index[code]
            return i, self.matrix[i]
        # Try without decimal
        nd = code.replace(".", "")
        if nd in self.index:
            i = self.index[nd]
            return i, self.matrix[i]
        # Prefix match (specificity fallback)
        for length in [7, 6, 5, 4, 3]:
            prefix = code[:length]
            if prefix in self.index:
                i = self.index[prefix]
                return i, self.matrix[i]
        return None

    def build_weight_vector(self, weights: dict) -> np.ndarray:
        """Build a 100-dim per-dimension weight vector from axis weights."""
        w = np.ones(DIM, dtype=np.float32)
        for axis, (start, end) in AXIS_RANGES.items():
            w[start:end] = weights.get(axis, 1.0)
        return w

    def weighted_cosine(self, v1: np.ndarray, v2: np.ndarray, w: np.ndarray) -> float:
        """Weighted cosine similarity."""
        wv1 = v1 * w
        wv2 = v2 * w
        n1  = np.linalg.norm(wv1)
        n2  = np.linalg.norm(wv2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(wv1, wv2) / (n1 * n2))

    def compare(self, code_a: str, code_b: str, weights: dict) -> dict:
        """Compare two codes and return detailed similarity breakdown."""
        ra = self.get_vector(code_a)
        rb = self.get_vector(code_b)

        if ra is None:
            return {"error": f"Code '{code_a}' not found in embeddings"}
        if rb is None:
            return {"error": f"Code '{code_b}' not found in embeddings"}

        ia, va = ra
        ib, vb = rb
        w = self.build_weight_vector(weights)

        overall = self.weighted_cosine(va, vb, w)

        # Per-axis breakdown
        axis_scores = {}
        for axis, (start, end) in AXIS_RANGES.items():
            seg_a = va[start:end]
            seg_b = vb[start:end]
            n1 = np.linalg.norm(seg_a)
            n2 = np.linalg.norm(seg_b)
            if n1 > 0 and n2 > 0:
                s = float(np.dot(seg_a, seg_b) / (n1 * n2))
            else:
                s = 0.0
            axis_scores[axis] = s

        return {
            "code_a": code_a.upper(),
            "desc_a": self.descs[ia],
            "type_a": self.types[ia],
            "code_b": code_b.upper(),
            "desc_b": self.descs[ib],
            "type_b": self.types[ib],
            "overall_similarity": overall,
            "axis_scores": axis_scores,
            "weights_used": weights,
        }

    def find_similar(self, code: str, weights: dict, topn: int = 10,
                     filter_type: str = None) -> dict:
        """Find top-N most similar codes to the query code."""
        r = self.get_vector(code)
        if r is None:
            return {"error": f"Code '{code}' not found in embeddings"}

        iq, vq = r
        w = self.build_weight_vector(weights)

        # Apply weights to query vector
        wvq = vq * w

        # Apply weights to entire matrix and compute all cosines at once
        wmat = self.matrix * w[np.newaxis, :]          # (N, 100)
        nq   = np.linalg.norm(wvq)
        nmat = np.linalg.norm(wmat, axis=1)            # (N,)
        mask = nmat > 0
        sims = np.zeros(len(self.codes), dtype=np.float32)
        sims[mask] = wmat[mask] @ wvq / (nmat[mask] * nq)

        # Exclude the query code itself
        sims[iq] = -2.0

        # Filter by type if requested
        if filter_type:
            ft = filter_type.strip()
            for i, t in enumerate(self.types):
                if ft.lower() not in t.lower():
                    sims[i] = -2.0

        # Get top-N
        top_idx = np.argpartition(sims, -topn)[-topn:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        matches = []
        for i in top_idx:
            if sims[i] < -1:
                continue
            matches.append({
                "rank":       len(matches) + 1,
                "code":       self.codes[i],
                "type":       self.types[i],
                "description":self.descs[i],
                "similarity": round(float(sims[i]), 6),
            })

        return {
            "query_code": code.upper(),
            "query_desc": self.descs[iq],
            "query_type": self.types[iq],
            "weights_used": weights,
            "topn": topn,
            "matches": matches,
        }


# ── Display helpers ──────────────────────────────────────────────────────────

def bar(value, width=20, fill="█", empty="░") -> str:
    """Text progress bar for similarity value [-1, 1]."""
    clamped = max(-1.0, min(1.0, value))
    filled  = int((clamped + 1) / 2 * width)
    return fill * filled + empty * (width - filled)

def sim_label(s: float) -> str:
    if s >= 0.85: return "VERY HIGH"
    if s >= 0.65: return "HIGH     "
    if s >= 0.40: return "MODERATE "
    if s >= 0.15: return "LOW      "
    if s >= 0.00: return "MINIMAL  "
    return          "NEGATIVE "

def print_compare_result(r: dict, weights: dict):
    if "error" in r:
        print(f"\n  [ERROR] {r['error']}\n")
        return

    SEP = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  CODE COMPARISON")
    print(f"{'═'*72}")
    print(f"  Code A : {r['code_a']:12s} [{r['type_a']}]")
    print(f"           {r['desc_a'][:60]}")
    print(f"  Code B : {r['code_b']:12s} [{r['type_b']}]")
    print(f"           {r['desc_b'][:60]}")
    print(SEP)

    s = r["overall_similarity"]
    print(f"  WEIGHTED SIMILARITY  {bar(s, 30)}  {s:+.4f}  {sim_label(s)}")
    print(SEP)
    print(f"  {'AXIS':<22} {'WEIGHT':>6}  {'SCORE':>7}  {'BAR':<22}  LABEL")
    print(f"  {'─'*22} {'─'*6}  {'─'*7}  {'─'*22}  {'─'*9}")

    for axis, score in r["axis_scores"].items():
        wt = weights.get(axis, 1.0)
        wt_str = f"{wt:.1f}x"
        b = bar(score, 22)
        lbl = sim_label(score)
        flag = " ◄" if wt >= 2.0 else ("  " if wt > 0 else " ✕")
        print(f"  {AXIS_DESCRIPTIONS[axis][:22]:<22} {wt_str:>6}  {score:+7.4f}  {b}  {lbl}{flag}")

    print(SEP)
    non_default = {k: v for k, v in weights.items() if v != 1.0}
    if non_default:
        print(f"  Active weight overrides: " +
              ", ".join(f"{k}={v}" for k, v in non_default.items()))
    print()

def print_find_result(r: dict, weights: dict):
    if "error" in r:
        print(f"\n  [ERROR] {r['error']}\n")
        return

    SEP = "─" * 72
    print(f"\n{'═'*72}")
    print(f"  TOP-{r['topn']} SIMILAR CODES")
    print(f"{'═'*72}")
    print(f"  Query : {r['query_code']:12s} [{r['query_type']}]")
    print(f"          {r['query_desc'][:60]}")

    non_default = {k: v for k, v in weights.items() if v != 1.0}
    if non_default:
        print(f"  Weights: " + ", ".join(f"{k}={v}" for k, v in non_default.items()))

    print(SEP)
    print(f"  {'#':>2}  {'CODE':<14} {'TYPE':<14} {'SIM':>7}  {'BAR':<22}  DESCRIPTION")
    print(f"  {'─'*2}  {'─'*14} {'─'*14} {'─'*7}  {'─'*22}  {'─'*30}")

    for m in r["matches"]:
        b   = bar(m["similarity"], 22)
        lbl = sim_label(m["similarity"])
        desc_short = m["description"][:35]
        print(f"  {m['rank']:>2}  {m['code']:<14} {m['type']:<14} "
              f"{m['similarity']:+7.4f}  {b}  {desc_short}")

    print(SEP)
    print()

def print_weights(weights: dict):
    print("\n  CURRENT AXIS WEIGHTS")
    print("  " + "─" * 60)
    for axis, desc in AXIS_DESCRIPTIONS.items():
        w = weights.get(axis, 1.0)
        bar_str = "█" * int(w * 5) + "░" * max(0, 20 - int(w * 5))
        flag = " ◄ BOOSTED" if w > 1.0 else (" ✕ DISABLED" if w == 0 else "")
        print(f"  {axis:<12}  {w:>5.1f}x  {bar_str[:10]}  {desc[:35]}{flag}")
    print()


# ── Interactive menu ─────────────────────────────────────────────────────────

def interactive(store: EmbeddingStore):
    weights = dict(DEFAULT_WEIGHTS)

    print("\n" + "═"*72)
    print("  MEDICAL CODE WEIGHTED SIMILARITY TOOL")
    print("  FWA Embedding Space Navigator")
    print("="*72)
    print("  Commands:")
    print("    w               — view / edit axis weights")
    print("    c CODE_A CODE_B — compare two codes")
    print("    f CODE [N]      — find top-N similar codes")
    print("    f CODE [N] -t TYPE  — filter by type (ICD-10-CM / CPT / HCPCS / ICD-10-PCS)")
    print("    r               — reset weights to 1.0")
    print("    q               — quit")
    print("─"*72 + "\n")

    while True:
        try:
            raw = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.\n")
            break

        if not raw:
            continue

        parts = raw.split()
        cmd   = parts[0].lower()

        # ── Quit ──────────────────────────────────────────────────────────
        if cmd in ("q", "quit", "exit"):
            print("\n  Goodbye.\n")
            break

        # ── Reset weights ──────────────────────────────────────────────────
        elif cmd == "r":
            weights = dict(DEFAULT_WEIGHTS)
            print("  Weights reset to 1.0\n")

        # ── View / edit weights ────────────────────────────────────────────
        elif cmd == "w":
            print_weights(weights)
            print("  Enter  axis=value  to update (e.g. fwa=3.0 severity=2.0),")
            print("  or press Enter to keep current weights.\n")
            try:
                line = input("  update > ").strip()
            except EOFError:
                continue
            if line:
                for token in line.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        k = k.strip().lower()
                        if k in weights:
                            try:
                                weights[k] = max(0.0, float(v))
                                print(f"  Set {k} = {weights[k]:.1f}x")
                            except ValueError:
                                print(f"  Invalid value: {v}")
                        else:
                            print(f"  Unknown axis: {k}  "
                                  f"(valid: {', '.join(AXIS_RANGES)})")
            print_weights(weights)

        # ── Compare two codes ──────────────────────────────────────────────
        elif cmd == "c":
            if len(parts) < 3:
                print("  Usage: c CODE_A CODE_B\n")
                continue
            result = store.compare(parts[1], parts[2], weights)
            print_compare_result(result, weights)

        # ── Find similar ───────────────────────────────────────────────────
        elif cmd == "f":
            if len(parts) < 2:
                print("  Usage: f CODE [N] [-t TYPE]\n")
                continue
            code  = parts[1]
            topn  = 10
            ftype = None
            i = 2
            while i < len(parts):
                if parts[i] == "-t" and i + 1 < len(parts):
                    ftype = parts[i + 1]; i += 2
                else:
                    try:
                        topn = int(parts[i])
                    except ValueError:
                        pass
                    i += 1
            result = store.find_similar(code, weights, topn=topn,
                                        filter_type=ftype)
            print_find_result(result, weights)

        else:
            print(f"  Unknown command: {cmd}  (type w/c/f/r/q)\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_weights_arg(args_weights) -> dict:
    """Parse list of 'axis:value' strings into weights dict."""
    weights = dict(DEFAULT_WEIGHTS)
    if not args_weights:
        return weights
    for token in args_weights:
        if ":" in token:
            k, v = token.split(":", 1)
        elif "=" in token:
            k, v = token.split("=", 1)
        else:
            continue
        k = k.strip().lower()
        if k in weights:
            try:
                weights[k] = max(0.0, float(v))
            except ValueError:
                print(f"[WARN] Invalid weight value: {token}")
        else:
            print(f"[WARN] Unknown axis: {k}")
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Medical Code Weighted Similarity Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python medical_similarity.py

  # Compare two codes (equal weights)
  python medical_similarity.py --compare I21.0 99232

  # Compare with boosted FWA and severity axes
  python medical_similarity.py --compare I21.0 I21.9 --weights fwa:3.0 severity:2.0

  # Find top 10 similar to I21.0
  python medical_similarity.py --find I21.0

  # Find top 20 similar, only ICD-10-CM, boost clinical and severity
  python medical_similarity.py --find I21.0 --topn 20 --filter-type ICD-10-CM \\
      --weights clinical:2.0 severity:3.0 fwa:0.0

  # Use custom embeddings CSV
  python medical_similarity.py --csv /path/to/embeddings.csv --find E11.9

Axis names for --weights:
  clinical  severity  intensity  anatomical  temporal
  billing   bundling  dxproc     fwa         drgrvu

Setting a weight to 0 disables that axis entirely.
Setting it to 3.0 makes it 3x more important than default.
        """
    )
    parser.add_argument("--csv",     default=EMBEDDINGS_CSV,
                        help="Path to FWA embeddings CSV")
    parser.add_argument("--compare", nargs=2, metavar=("CODE_A","CODE_B"),
                        help="Compare two codes")
    parser.add_argument("--find",    metavar="CODE",
                        help="Find top-N similar codes")
    parser.add_argument("--topn",    type=int, default=10,
                        help="Number of results for --find (default 10)")
    parser.add_argument("--filter-type", metavar="TYPE",
                        help="Filter --find results by code type")
    parser.add_argument("--weights", nargs="+", metavar="AXIS:VALUE",
                        help="Weight overrides e.g. fwa:3.0 severity:2.0")

    args = parser.parse_args()

    store   = EmbeddingStore()
    store.load(args.csv)
    weights = parse_weights_arg(args.weights)

    if args.compare:
        result = store.compare(args.compare[0], args.compare[1], weights)
        print_compare_result(result, weights)

    elif args.find:
        result = store.find_similar(
            args.find, weights,
            topn=args.topn,
            filter_type=args.filter_type
        )
        print_find_result(result, weights)

    else:
        interactive(store)


if __name__ == "__main__":
    main()
