"""
evaluate_retrieval.py

Full, reproducible pipeline for the GDPR compliance-retrieval evaluation:
  1. Build the gold-eligible eval set from the OPP-115 parquet splits +
     the JURIX (Poplavska et al., 2020) expert category->article mapping.
  2. Rank all 99 GDPR articles per query with BM25, dense (embeddings you
     supply), and RRF hybrid fusion over a top-20 candidate pool from each.
  3. Compute top-1 accuracy and MRR for each method, restricted to the
     top-20 pool (matching the paper draft's stated methodology).
  4. Run McNemar's test on every pairwise top-1 comparison.
  5. Run a paired bootstrap (10,000 resamples) on the hybrid-vs-BM25 top-1
     difference, reporting the 95% CI and the fraction of resamples where
     BM25 matches or beats hybrid.

INPUTS (paths are CLI args, see bottom of file / --help):
  --gdpr_articles      JSON: [{"id": "Art.N", "title": ..., "text": ...}, ...]
                        (99 entries; see build_gdpr_corpus.py to regenerate
                        from GDPRtEXT if you don't already have this file)
  --opp115_train / --opp115_test / --opp115_validation
                        The three OPP-115 parquet files (text, label columns;
                        label is a list of ints per the 12-class scheme used
                        by the alzoubi36/opp_115 / PrivacyGLUE label set)
  --corpus_embeddings   JSON: {"ids": [...], "embeddings": [...], "model": ...}
                        Dense embeddings for the 99 GDPR articles (must be
                        the SAME model/space as --segment_embeddings)
  --segment_embeddings  JSON: {"texts": [...], "embeddings": [...], "model": ...}
                        Dense embeddings for the OPP-115 segments

OUTPUT: prints the results table + significance tests, and writes a JSON
with all per-example and aggregate results to --output.

Usage:
    python evaluate_retrieval.py \
        --gdpr_articles gdpr_99_articles.json \
        --opp115_train train.parquet --opp115_test test.parquet --opp115_validation validation.parquet \
        --corpus_embeddings corpus_embeddings_bge.json \
        --segment_embeddings segment_embeddings_bge.json \
        --output results.json
"""

import argparse
import json
import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from statsmodels.stats.contingency_tables import mcnemar


# ---------------------------------------------------------------------------
# Gold mapping: Poplavska et al. (2020) / JURIX 2020 expert-validated
# category -> GDPR article connections (from connections_overview.csv,
# categories_articles_matrix.csv in the published JURIX dataset release).
# Categories not listed here (Policy Change, Do Not Track, Other) have no
# associated GDPR articles under this mapping and are excluded from eval.
# ---------------------------------------------------------------------------
GOLD_MAP = {
    "First Party Collection/Use": [4, 5, 6, 7, 8, 9, 10, 11, 24, 25, 30, 33, 34, 35, 36, 37, 38, 39, 89, 91, 95],
    "Third Party Sharing/Collection": [4, 6, 9, 19, 28, 29, 30, 37, 38, 39, 44, 45, 46, 47, 48, 49, 96],
    "User Choice/Control": [4, 6, 7, 8, 9, 13, 14, 17, 18, 20, 21, 26, 49, 77, 78, 79, 80, 82],
    "User Access, Edit and Deletion": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 25],
    "Data Retention": [5, 25, 30],
    "Data Security": [4, 5, 6, 12, 24, 25, 28, 30, 32, 33, 34, 35, 36, 45, 89],
    "International and Specific Audiences": [8],
}

# 12-class label order used by the OPP-115 parquet files' integer labels
# (PrivacyGLUE / alzoubi36/opp_115 convention: ClassLabel(names=LABELS)).
LABELS = [
    "Data Retention", "Data Security", "Do Not Track", "First Party Collection/Use",
    "International and Specific Audiences", "Introductory/Generic", "Policy Change",
    "Practice not covered", "Privacy contact information", "Third Party Sharing/Collection",
    "User Access, Edit and Deletion", "User Choice/Control",
]

POOL = 20      # candidate pool size per retriever, per Section 3.3/3.4 of the draft
K_RRF = 60     # RRF constant, standard default (Cormack et al., 2009)
N_BOOTSTRAP = 10_000


def tokenize(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


def build_eval_examples(train_path, test_path, val_path):
    """Load OPP-115 splits, dedupe, attach gold GDPR article sets, keep only
    segments with >=1 category that has a JURIX gold mapping."""
    dfs = [pd.read_parquet(p) for p in [train_path, test_path, val_path]]
    full = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)

    examples = []
    for _, row in full.iterrows():
        cats = [LABELS[l] for l in row["label"]]
        gold_articles = set()
        has_gold_cat = False
        for c in cats:
            if c in GOLD_MAP:
                has_gold_cat = True
                gold_articles.update(GOLD_MAP[c])
        if has_gold_cat:
            examples.append({"text": row["text"], "categories": cats, "gold": sorted(gold_articles)})
    return examples


def build_bm25(gdpr_articles_path):
    articles = json.load(open(gdpr_articles_path))
    art_nums = [int(a["id"].split(".")[1]) for a in articles]
    art_texts = [a["text"] for a in articles]
    bm25 = BM25Okapi([tokenize(t) for t in art_texts])
    return bm25, art_nums


def load_dense(corpus_embeddings_path, segment_embeddings_path):
    art_data = json.load(open(corpus_embeddings_path))
    art_nums = [int(s.split(".")[1]) for s in art_data["ids"]]
    art_vecs = np.array(art_data["embeddings"], dtype=np.float32)
    art_vecs /= np.linalg.norm(art_vecs, axis=1, keepdims=True)

    seg_data = json.load(open(segment_embeddings_path))
    text_to_vec = {}
    for t, v in zip(seg_data["texts"], seg_data["embeddings"]):
        v = np.array(v, dtype=np.float32)
        text_to_vec[t] = v / (np.linalg.norm(v) + 1e-9)

    return art_vecs, art_nums, text_to_vec


def topk_metrics(ranked, gold_set, pool):
    cand = ranked[:pool]
    top1 = int(cand[0] in gold_set) if cand else 0
    rr = 0.0
    for rank, a in enumerate(cand, start=1):
        if a in gold_set:
            rr = 1.0 / rank
            break
    return top1, rr


def rrf_fuse(bm25_top20, dense_top20, k_rrf=K_RRF):
    bm25_rank_of = {a: r + 1 for r, a in enumerate(bm25_top20)}
    dense_rank_of = {a: r + 1 for r, a in enumerate(dense_top20)}
    candidate_pool = set(bm25_top20) | set(dense_top20)
    scores = {}
    for a in candidate_pool:
        s = 0.0
        if a in bm25_rank_of:
            s += 1.0 / (k_rrf + bm25_rank_of[a])
        if a in dense_rank_of:
            s += 1.0 / (k_rrf + dense_rank_of[a])
        scores[a] = s
    return sorted(scores.keys(), key=lambda a: -scores[a])


def mcnemar_test(a, b):
    a, b = np.array(a), np.array(b)
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    res = mcnemar([[0, n10], [n01, 0]], exact=False, correction=True)
    return n10, n01, res.pvalue


def bootstrap_ci(diffs, n_boot=N_BOOTSTRAP, seed=42):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    idx = np.arange(n)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        samp = rng.choice(idx, size=n, replace=True)
        boot_means[i] = diffs[samp].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    frac_le_zero = float(np.mean(boot_means <= 0))
    return float(lo), float(hi), frac_le_zero


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gdpr_articles", required=True)
    p.add_argument("--opp115_train", required=True)
    p.add_argument("--opp115_test", required=True)
    p.add_argument("--opp115_validation", required=True)
    p.add_argument("--corpus_embeddings", required=True)
    p.add_argument("--segment_embeddings", required=True)
    p.add_argument("--output", default="results.json")
    args = p.parse_args()

    examples = build_eval_examples(args.opp115_train, args.opp115_test, args.opp115_validation)
    print(f"Gold-eligible eval examples: {len(examples)}")

    bm25, art_nums_bm25 = build_bm25(args.gdpr_articles)
    art_vecs, art_nums_dense, text_to_vec = load_dense(args.corpus_embeddings, args.segment_embeddings)

    missing = sum(1 for ex in examples if ex["text"] not in text_to_vec)
    if missing:
        print(f"WARNING: {missing} examples have no matching dense embedding and will error out below.")

    bm25_hits, bm25_rr = [], []
    dense_hits, dense_rr = [], []
    hybrid_hits, hybrid_rr = [], []
    per_cat = {}

    for ex in examples:
        text = ex["text"]
        gold_set = set(ex["gold"])

        bscores = bm25.get_scores(tokenize(text))
        bm25_ranked = [art_nums_bm25[i] for i in np.argsort(-bscores)]

        qvec = text_to_vec[text]
        dscores = art_vecs @ qvec
        dense_ranked = [art_nums_dense[i] for i in np.argsort(-dscores)]

        b_t1, b_rr = topk_metrics(bm25_ranked, gold_set, POOL)
        d_t1, d_rr = topk_metrics(dense_ranked, gold_set, POOL)

        hybrid_ranked = rrf_fuse(bm25_ranked[:POOL], dense_ranked[:POOL])
        h_t1, h_rr = topk_metrics(hybrid_ranked, gold_set, pool=len(hybrid_ranked))

        bm25_hits.append(b_t1); bm25_rr.append(b_rr)
        dense_hits.append(d_t1); dense_rr.append(d_rr)
        hybrid_hits.append(h_t1); hybrid_rr.append(h_rr)

        for method, t1, rr in [("bm25", b_t1, b_rr), ("dense", d_t1, d_rr), ("hybrid", h_t1, h_rr)]:
            for c in ex["categories"]:
                d = per_cat.setdefault(c, {}).setdefault(method, {"n": 0, "top1": 0, "rr": 0.0})
                d["n"] += 1
                d["top1"] += t1
                d["rr"] += rr

    n = len(examples)
    print(f"\nn = {n}")
    print(f"{'Method':10s} {'Top-1':>8s} {'MRR':>8s}")
    print(f"{'BM25':10s} {np.mean(bm25_hits):8.4f} {np.mean(bm25_rr):8.4f}")
    print(f"{'Dense':10s} {np.mean(dense_hits):8.4f} {np.mean(dense_rr):8.4f}")
    print(f"{'Hybrid':10s} {np.mean(hybrid_hits):8.4f} {np.mean(hybrid_rr):8.4f}")

    print("\nMcNemar's test (paired top-1 hit/miss):")
    mcnemar_results = {}
    for name, x, y in [
        ("bm25_vs_dense", bm25_hits, dense_hits),
        ("bm25_vs_hybrid", bm25_hits, hybrid_hits),
        ("dense_vs_hybrid", dense_hits, hybrid_hits),
    ]:
        n10, n01, pval = mcnemar_test(x, y)
        mcnemar_results[name] = {"n10": n10, "n01": n01, "p_value": pval}
        print(f"  {name}: n10={n10}, n01={n01}, p={pval:.4g}")

    bm25_arr = np.array(bm25_hits)
    hyb_arr = np.array(hybrid_hits)
    diffs = hyb_arr - bm25_arr
    obs_diff = float(diffs.mean())
    ci_lo, ci_hi, frac_le_zero = bootstrap_ci(diffs)
    print(f"\nHybrid - BM25 top-1 diff: {obs_diff:.4f}")
    print(f"95% bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]  (n_boot={N_BOOTSTRAP})")
    print(f"Fraction of resamples where diff <= 0: {frac_le_zero:.4f}")

    out = {
        "n": n,
        "aggregate": {
            "bm25": {"top1": float(np.mean(bm25_hits)), "mrr": float(np.mean(bm25_rr))},
            "dense": {"top1": float(np.mean(dense_hits)), "mrr": float(np.mean(dense_rr))},
            "hybrid": {"top1": float(np.mean(hybrid_hits)), "mrr": float(np.mean(hybrid_rr))},
        },
        "mcnemar": mcnemar_results,
        "bootstrap_hybrid_minus_bm25": {
            "observed_diff": obs_diff, "ci_95": [ci_lo, ci_hi], "frac_diff_le_zero": frac_le_zero,
        },
        "per_category": per_cat,
    }
    json.dump(out, open(args.output, "w"), indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
