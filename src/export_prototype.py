"""Export data for the web prototype.

Trains the PoC model (item-kNN with BM25 weighting, the winner of experiment.py)
on all cleaned 5-core interactions and writes a compact JSON the static page reads:

- works:      metadata of every work the page can show
- neighbours: for each selectable work, its top-N similar works with
              similarity score and number of shared readers
- lists:      naive bestsellers (count) and quality-weighted bestsellers
- shelves:    example reading histories used as presets

Usage: python export_prototype.py <artifacts_dir> <out_json>
"""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import ftfy
from threadpoolctl import threadpool_limits
from implicit.nearest_neighbours import BM25Recommender

A, OUT = Path(sys.argv[1]), Path(sys.argv[2])
N_SELECTABLE, N_NEIGH, SEED = 2500, 25, 7

core = pd.read_parquet(A / "interactions_core.parquet")
cat = pd.read_parquet(A / "catalogue.parquet").set_index("work_key")
params = json.load(open(A / "experiment_results.json"))["results"]["bm25"]["params"]

items = core["work_key"].unique(); iid = {w: i for i, w in enumerate(items)}
users = core["User-ID"].unique(); uid = {u: i for i, u in enumerate(users)}
core["i"] = core["work_key"].map(iid); core["u"] = core["User-ID"].map(uid)
X = sp.csr_matrix((core["conf"].astype(np.float32), (core["u"], core["i"])), shape=(len(users), len(items)))
B = (X > 0).astype(np.float32).tocsc()

with threadpool_limits(1, "blas"):
    model = BM25Recommender(K=params["k"]); model.fit(X, show_progress=False)
S = model.similarity.tocsr()

pop = np.asarray((X > 0).sum(axis=0)).ravel()
selectable = np.argsort(-pop)[:N_SELECTABLE]

def clean(t):
    t = ftfy.fix_text(str(t)).replace("\\", "")
    return re.sub(r"\s+", " ", t).strip()

def author(a):
    a = clean(a)
    return a.title() if a.isupper() else a

neigh, needed = {}, set(selectable.tolist())
Bt = B.T.tocsr()
for i in selectable:
    row = S.getrow(i); idx, val = row.indices, row.data
    keep = idx != i; idx, val = idx[keep], val[keep]
    order = np.argsort(-val)[:N_NEIGH]; idx, val = idx[order], val[order]
    shared = np.asarray(Bt[i].multiply(Bt[idx]).sum(axis=1)).ravel() if len(idx) else []
    neigh[int(i)] = [[int(j), round(float(v), 4), int(c)] for j, v, c in zip(idx, val, shared)]
    needed.update(idx.tolist())

# bestseller lists ---------------------------------------------------------------
e = core[core["rating"] > 0]; C = e["rating"].mean()
g = e.groupby("i")["rating"].agg(["count", "mean"]).join(pd.Series(pop, name="n"))
g = g[g["n"] >= 50]
g["bayes"] = (g["count"] * g["mean"] + 50 * C) / (g["count"] + 50)
naive = np.argsort(-pop)[:20].tolist()
quality = g.sort_values("bayes", ascending=False).index[:20].tolist()
needed.update(naive); needed.update(quality)
# displayed average rating: all explicit ratings 1-10 of the work's editions (raw data, dislikes included)
def _nt(t):
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", str(t).lower()).split(":")[0]
    return re.sub(r"[^a-z0-9]+", " ", t).strip()
import html as _html
_b = pd.read_csv(A.parent / "data" / "Books.csv", dtype=str, keep_default_na=False, na_values=[""])
_b["work_key"] = _b["Book-Title"].map(lambda t: _nt(_html.unescape(str(t)))) + "|" + _b["Book-Author"].fillna("").map(lambda a: re.sub(r"[^a-z]+", "", _html.unescape(a).lower()))
_r = pd.read_csv(A.parent / "data" / "Ratings.csv", dtype={"ISBN": str})
_r = _r[_r["Book-Rating"] > 0].merge(_b[["ISBN", "work_key"]], on="ISBN")
_m = _r.groupby("work_key")["Book-Rating"].mean()
mean_expl = pd.Series({i: _m[w] for w, i in iid.items() if w in _m.index})

# example shelves ----------------------------------------------------------------
def find(q):
    hits = [i for i in selectable if q.lower() in cat.loc[items[i], "title"].lower()]
    return max(hits, key=lambda i: pop[i]) if hits else None
presets = {
    "fantasy": ["The Hobbit", "The Fellowship of the Ring", "Harry Potter and the Sorcerer's Stone"],
    "thriller": ["The Firm", "The Pelican Brief", "Along Came a Spider"],
    "womens": ["Bridget Jones's Diary", "The Secret Life of Bees", "Divine Secrets of the Ya-Ya Sisterhood"],
}
shelves = {k: [int(i) for i in (find(q) for q in v) if i is not None] for k, v in presets.items()}
# one real reader from the data: 8-15 interactions, all within the selectable set
rng = np.random.default_rng(SEED); sel = set(selectable.tolist())
cand = [u for u in range(len(users)) if 8 <= X[u].nnz <= 15 and set(X[u].indices) <= sel]
real_u = int(rng.choice(cand))
shelves["real"] = [int(i) for i in X[real_u].indices]
shelves["real_id"] = int(users[real_u])

# works metadata ---------------------------------------------------------------------
works = {}
for i in sorted(needed):
    r = cat.loc[items[i]]
    works[int(i)] = {"t": clean(r["title"]), "a": author(r["author"]),
                     "y": None if pd.isna(r["year"]) else int(r["year"]),
                     "isbn": r["isbn"], "n": int(pop[i]),
                     "r": None if i not in mean_expl.index else round(float(mean_expl[i]), 1)}

out = {"meta": {"model": f"item-kNN (BM25, K={params['k']})", "users": len(users), "works_in_model": len(items),
                "interactions": int(X.nnz), "selectable": N_SELECTABLE, "neighbours": N_NEIGH},
       "selectable": [int(i) for i in selectable], "works": works, "neighbours": neigh,
       "lists": {"naive": naive, "quality": [int(i) for i in quality]}, "shelves": shelves}
OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
print("works", len(works), "selectable", len(selectable), "size KB", OUT.stat().st_size // 1024)
print("shelves", {k: [works[i]["t"] for i in v] if isinstance(v, list) else v for k, v in shelves.items()})
print("naive top5", [works[i]["t"] for i in naive[:5]])
print("quality top5", [works[i]["t"] for i in quality[:5]])
