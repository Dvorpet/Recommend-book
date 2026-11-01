"""Offline comparison of recommender approaches on the cleaned 5-core data.

No timestamps exist, so we use a random per-user hold-out (20 % of each user's
works, min. 1) as test. Hyper-parameters are chosen on a second hold-out carved
from the training part (validation), then the best config is scored on test.

Models: most-popular (count), quality-weighted popularity (Bayesian mean),
item-kNN (cosine), ALS (implicit matrix factorisation).

Usage: python experiment.py <artifacts_dir>
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from threadpoolctl import threadpool_limits
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import CosineRecommender, BM25Recommender

A = Path(sys.argv[1]); K = 10; SEED = 42
core = pd.read_parquet(A / "interactions_core.parquet")
cat = pd.read_parquet(A / "catalogue.parquet").set_index("work_key")

uids = {u: i for i, u in enumerate(core["User-ID"].unique())}
iids = {w: i for i, w in enumerate(core["work_key"].unique())}
core["u"] = core["User-ID"].map(uids); core["i"] = core["work_key"].map(iids)
n_u, n_i = len(uids), len(iids)
inv_i = np.array(list(iids.keys()))

def holdout(df, frac=0.2, seed=SEED):
    rng = np.random.default_rng(seed)
    df = df.assign(_r=rng.random(len(df)))
    df["_rank"] = df.groupby("u")["_r"].rank(method="first")
    n = df.groupby("u")["u"].transform("size")
    k = np.maximum(1, np.floor(n * frac)).astype(int)
    test = df["_rank"] <= k
    return df[~test].drop(columns=["_r", "_rank"]), df[test].drop(columns=["_r", "_rank"])

train_full, test = holdout(core)
train, valid = holdout(train_full, seed=SEED + 1)

def csr(df):
    return sp.csr_matrix((df["conf"].astype(np.float32), (df["u"], df["i"])), shape=(n_u, n_i))

# ---------------------------------------------------------------- models -----
class PopCount:
    name = "Most popular (count)"
    def fit(self, df):
        self.order = df.groupby("i").size().sort_values(ascending=False).index.values
        return self
    def recommend(self, users, X):
        return np.vstack([self._top(X[u]) for u in users])
    def _top(self, row, N=K):
        seen = set(row.indices); out = []
        for i in self.order:
            if i not in seen:
                out.append(i)
                if len(out) == N: break
        return out

class PopQuality(PopCount):
    """Bayesian-average rating among works with enough interactions."""
    name = "Popularity, quality-weighted"
    def __init__(self, m=20, min_n=30): self.m, self.min_n = m, min_n
    def fit(self, df):
        e = df[df["rating"] > 0]; C = e["rating"].mean()
        g = e.groupby("i")["rating"].agg(["count", "mean"])
        n_all = df.groupby("i").size()
        g = g.join(n_all.rename("n")); g = g[g["n"] >= self.min_n]
        g["bayes"] = (g["count"] * g["mean"] + self.m * C) / (g["count"] + self.m)
        self.order = g.sort_values("bayes", ascending=False).index.values
        return self

class Implicit:
    def __init__(self, model, name): self.model, self.name = model, name
    def fit(self, df):
        self.X = csr(df); self.model.fit(self.X, show_progress=False); return self
    def recommend(self, users, X):
        ids, _ = self.model.recommend(np.asarray(users), X[users], N=K, filter_already_liked_items=True)
        return ids

# --------------------------------------------------------------- metrics -----
pop_counts = train_full.groupby("i").size().reindex(range(n_i), fill_value=0).values
pop_pct = pd.Series(pop_counts).rank(pct=True).values      # 1.0 = most popular

def evaluate(model, train_df, eval_df, by_segment=False):
    X = csr(train_df)
    truth = eval_df.groupby("u")["i"].apply(set)
    users = truth.index.values
    recs = model.recommend(users, X)
    disc = 1 / np.log2(np.arange(2, K + 2))
    rows = []
    for u, rec in zip(users, recs):
        t = truth[u]; hits = np.array([i in t for i in rec], dtype=float)
        idcg = disc[: min(len(t), K)].sum()
        rows.append((u, hits.sum() / K, hits.sum() / len(t), (hits * disc).sum() / idcg, float(hits.any())))
    res = pd.DataFrame(rows, columns=["u", "precision", "recall", "ndcg", "hit"])
    allrec = np.concatenate([np.asarray(r) for r in recs])
    out = {"precision@10": res.precision.mean(), "recall@10": res.recall.mean(),
           "ndcg@10": res.ndcg.mean(), "hit_rate@10": res.hit.mean(),
           "coverage@10": len(np.unique(allrec)) / n_i,
           "avg_pop_percentile": float(pop_pct[allrec].mean()), "users": len(users)}
    if by_segment:
        n_train = train_df.groupby("u").size()
        res["seg"] = pd.cut(res["u"].map(n_train), [0, 9, 49, 10**9], labels=["4-9", "10-49", "50+"])
        out["ndcg_by_segment"] = res.groupby("seg", observed=True)["ndcg"].mean().round(4).to_dict()
        out["users_by_segment"] = res.groupby("seg", observed=True).size().to_dict()
    return out

def build(kind, **p):
    if kind == "pop": return PopCount()
    if kind == "popq": return PopQuality(**p)
    if kind == "cos": return Implicit(CosineRecommender(K=p["k"]), "Item-kNN (cosine)")
    if kind == "bm25": return Implicit(BM25Recommender(K=p["k"]), "Item-kNN (BM25)")
    if kind == "als":
        class ALSw(Implicit):
            def fit(self, df):
                self.X = csr(df) * p["alpha"]; self.model.fit(self.X, show_progress=False); return self
        return ALSw(AlternatingLeastSquares(factors=p["f"], regularization=p["reg"], iterations=20,
                                            random_state=SEED), "ALS (matrix factorisation)")

grid = {"pop": [{}], "popq": [{"m": 20, "min_n": 30}, {"m": 50, "min_n": 50}],
        "cos": [{"k": 50}, {"k": 200}], "bm25": [{"k": 50}, {"k": 200}],
        "als": [{"f": f, "reg": reg, "alpha": a} for f in (64, 128) for reg in (0.05,) for a in (5, 20)]}

results, tuning = {}, {}
with threadpool_limits(1, "blas"):
    for kind, cfgs in grid.items():
        best = None
        for p in cfgs:
            t0 = time.time(); m = build(kind, **p).fit(train)
            v = evaluate(m, train, valid)["ndcg@10"]
            tuning.setdefault(kind, []).append({"params": p, "valid_ndcg@10": round(v, 4), "sec": round(time.time() - t0, 1)})
            if best is None or v > best[0]: best = (v, p)
        m = build(kind, **best[1]).fit(train_full)
        r = evaluate(m, train_full, test, by_segment=True)
        r["params"] = best[1]; r["name"] = m.name
        results[kind] = r
        print(kind, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        if kind == "als": als_model = m
        if kind in ("cos", "bm25"):
            if kind == "bm25" or "knn_model" not in globals(): knn_model = m

# ------------------------------------------------- qualitative sanity check ---
def find(title_part):
    c = cat[cat["title"].str.contains(title_part, case=False, regex=False)]
    c = c[c.index.isin(iids)]
    return None if c.empty else c.index[np.argmax([pop_counts[iids[w]] for w in c.index])]

examples = {}
for q in ["Harry Potter and the Sorcerer", "The Da Vinci Code", "The Lovely Bones", "The Hobbit",
          "Bridget Jones's Diary", "The Firm"]:
    w = find(q)
    if w is None: continue
    out = {}
    for label, mdl in [("ALS", als_model), ("item-kNN", knn_model)]:
        ids, _ = mdl.model.similar_items(iids[w], N=6)
        out[label] = [f'{cat.loc[inv_i[i], "title"]} — {cat.loc[inv_i[i], "author"]}' for i in ids if i != iids[w]][:5]
    examples[f'{cat.loc[w, "title"]} — {cat.loc[w, "author"]}'] = out

summary = {"setup": {"users": n_u, "items": n_i, "train": len(train_full), "test": len(test),
                     "split": "random 20 % per-user hold-out (seed 42); params tuned on a 2nd hold-out"},
           "results": results, "tuning": tuning, "similar_items_examples": examples}
json.dump(summary, open(A / "experiment_results.json", "w"), indent=2, default=lambda o: float(o) if isinstance(o, np.floating) else int(o) if isinstance(o, np.integer) else str(o))
print(json.dumps(examples, indent=1, ensure_ascii=False))
