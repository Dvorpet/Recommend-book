"""Data preparation for the recommender PoC.

Steps (each one motivated by the EDA, see docs/eda/eda_findings.md):
1. keep only ratings whose ISBN exists in Books.csv (metadata needed for display)
2. clean titles/authors (HTML entities, whitespace) and merge editions to a "work"
   (normalised title without parentheticals + normalised author)
3. treat interactions as implicit feedback; drop explicit dislikes (rating 1-4)
4. remove bulk "catalogue upload" users (>1000 interactions and >90 % implicit)
5. iterative 5-core filter on user x work

Usage: python prep.py <data_dir> <out_dir>
"""
import sys, re, html, json
from pathlib import Path
import numpy as np
import pandas as pd

D, O = Path(sys.argv[1]), Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
log = {}

books = pd.read_csv(D / "Books.csv", dtype=str, keep_default_na=False, na_values=[""])
r = pd.read_csv(D / "Ratings.csv", dtype={"ISBN": str})
log["raw_interactions"] = len(r)

# 1 join ----------------------------------------------------------------------
r = r[r["ISBN"].isin(set(books["ISBN"]))]
log["after_join"] = len(r)

# 2 clean + work id -----------------------------------------------------------
def clean(s):
    s = html.unescape(str(s)) if isinstance(s, str) else ""
    return re.sub(r"\s+", " ", s).strip()

def norm_title(s):
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s.lower())       # drop "(Book 1)", "(Paperback)"...
    s = s.split(":")[0]                                   # drop subtitles
    return re.sub(r"[^a-z0-9]+", " ", s).strip()

def norm_author(s):
    return re.sub(r"[^a-z]+", "", s.lower())

books["title"] = books["Book-Title"].map(clean)
books["author"] = books["Book-Author"].map(clean)
books["year"] = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
books.loc[(books["year"] < 1800) | (books["year"] > 2006), "year"] = np.nan
books["publisher"] = books["Publisher"].map(clean)
books["work_key"] = books["title"].map(norm_title) + "|" + books["author"].map(norm_author)

r = r.merge(books[["ISBN", "work_key"]], on="ISBN")
# representative edition = the most-interacted ISBN of a work (best metadata/cover odds)
isbn_pop = r.groupby("ISBN").size().rename("isbn_n")
rep = (books[["ISBN", "work_key"]].join(isbn_pop, on="ISBN").fillna({"isbn_n": 0})
       .sort_values("isbn_n", ascending=False).drop_duplicates("work_key"))
log["isbns_in_ratings"] = int(r["ISBN"].nunique())
log["works_in_ratings"] = int(r["work_key"].nunique())

# 3 implicit signal ----------------------------------------------------------
log["explicit_dislikes_dropped"] = int(r["Book-Rating"].between(1, 4).sum())
r = r[~r["Book-Rating"].between(1, 4)]
# confidence: explicit positive rating is stronger evidence than a bare interaction
r["conf"] = np.where(r["Book-Rating"] > 0, 1.0 + r["Book-Rating"] / 5.0, 1.0)
ui = (r.groupby(["User-ID", "work_key"])
        .agg(conf=("conf", "max"), rating=("Book-Rating", "max")).reset_index())
log["user_work_pairs"] = len(ui)

# 4 bulk users -----------------------------------------------------------------
u = ui.groupby("User-ID").agg(n=("work_key", "size"), impl=("rating", lambda s: (s == 0).mean()))
bulk = u[(u.n > 1000) & (u.impl > 0.9)].index
log["bulk_users_removed"] = int(len(bulk))
log["bulk_users_interactions"] = int(u.loc[bulk, "n"].sum())
ui = ui[~ui["User-ID"].isin(bulk)]

# 5 k-core -----------------------------------------------------------------------
def kcore(df, k=5):
    while True:
        n0 = len(df)
        df = df[df.groupby("work_key")["User-ID"].transform("size") >= k]
        df = df[df.groupby("User-ID")["work_key"].transform("size") >= k]
        if len(df) == n0:
            return df
core = kcore(ui, 5)
log.update(core_interactions=len(core), core_users=int(core["User-ID"].nunique()),
           core_items=int(core["work_key"].nunique()))
log["core_density"] = log["core_interactions"] / (log["core_users"] * log["core_items"])

# catalogue of works (all works, for popularity/cold-start display) -------------
cat = rep.merge(books[["ISBN", "title", "author", "year", "publisher", "Image-URL-M", "Image-URL-L"]], on="ISBN")
cat = cat.rename(columns={"ISBN": "isbn", "Image-URL-M": "img_m", "Image-URL-L": "img_l"})
cat = cat[cat["work_key"].isin(ui["work_key"])]

core.to_parquet(O / "interactions_core.parquet", index=False)
ui.to_parquet(O / "interactions_all.parquet", index=False)
cat.drop(columns=["isbn_n"]).to_parquet(O / "catalogue.parquet", index=False)
json.dump(log, open(O / "prep_log.json", "w"), indent=2)
print(json.dumps(log, indent=1))
