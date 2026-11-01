"""Book-Crossing dataset - exploratory data analysis for the recommender PoC.
Usage: python eda.py <data_dir> <out_dir>
"""
import sys, json, re
from pathlib import Path
import numpy as np
import pandas as pd

D, O = Path(sys.argv[1]), Path(sys.argv[2])
O.mkdir(parents=True, exist_ok=True)
R = {}

books = pd.read_csv(D / "Books.csv", dtype=str, keep_default_na=False, na_values=[""])
users = pd.read_csv(D / "Users.csv", dtype={"User-ID": int, "Location": str})
ratings = pd.read_csv(D / "Ratings.csv", dtype={"User-ID": int, "ISBN": str, "Book-Rating": int})

# ---------- basic shape / types / missing / dupes ----------
for name, df in [("books", books), ("users", users), ("ratings", ratings)]:
    R[name] = {
        "rows": len(df), "cols": df.shape[1],
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(v) for c, v in df.isna().sum().items()},
        "full_dup_rows": int(df.duplicated().sum()),
    }
R["books"]["dup_isbn"] = int(books["ISBN"].duplicated().sum())
R["ratings"]["dup_user_isbn"] = int(ratings.duplicated(["User-ID", "ISBN"]).sum())
R["users"]["dup_user_id"] = int(users["User-ID"].duplicated().sum())

# ---------- ISBN quality ----------
valid_isbn = re.compile(r"^(\d{9}[\dX]|\d{13})$")
def norm_isbn(s):
    return re.sub(r"[^0-9X]", "", str(s).upper())
r_isbn = ratings["ISBN"]
R["ratings"]["isbn_valid_format_share"] = float(r_isbn.str.match(valid_isbn).mean())
R["books"]["isbn_valid_format_share"] = float(books["ISBN"].str.match(valid_isbn).mean())
book_set = set(books["ISBN"])
in_books = r_isbn.isin(book_set)
R["join"] = {
    "ratings_rows_matching_books": int(in_books.sum()),
    "ratings_rows_matching_books_share": float(in_books.mean()),
    "unique_rated_isbns": int(r_isbn.nunique()),
    "unique_rated_isbns_in_books": int(r_isbn[in_books].nunique()),
}
norm_books = set(books["ISBN"].map(norm_isbn))
in_books_norm = r_isbn.map(norm_isbn).isin(norm_books)
R["join"]["share_after_normalisation"] = float(in_books_norm.mean())
R["join"]["users_in_ratings_not_in_users"] = int((~ratings["User-ID"].isin(set(users["User-ID"]))).sum())

# ---------- rating distribution ----------
dist = ratings["Book-Rating"].value_counts().sort_index()
R["ratings"]["distribution"] = {int(k): int(v) for k, v in dist.items()}
R["ratings"]["implicit_share"] = float((ratings["Book-Rating"] == 0).mean())
expl = ratings[ratings["Book-Rating"] > 0]
R["ratings"]["explicit_mean"] = float(expl["Book-Rating"].mean())
R["ratings"]["explicit_median"] = float(expl["Book-Rating"].median())

# ---------- activity ----------
def desc(s):
    q = s.quantile([.25, .5, .75, .9, .99]).to_dict()
    return {"n": int(len(s)), "mean": float(s.mean()), "max": int(s.max()),
            **{f"p{int(k*100)}": float(v) for k, v in q.items()},
            "eq1_share": float((s == 1).mean()),
            "ge5_share": float((s >= 5).mean()), "ge10_share": float((s >= 10).mean()),
            "ge20_share": float((s >= 20).mean())}
ru = ratings.groupby("User-ID").size()
rb = ratings.groupby("ISBN").size()
R["activity"] = {"per_user_all": desc(ru), "per_book_all": desc(rb),
                 "per_user_explicit": desc(expl.groupby("User-ID").size()),
                 "per_book_explicit": desc(expl.groupby("ISBN").size())}
R["activity"]["users_with_no_ratings"] = int((~users["User-ID"].isin(ru.index)).sum())
# concentration
for k, s in [("users", ru), ("books", rb)]:
    s2 = s.sort_values(ascending=False)
    cum = s2.cumsum() / s2.sum()
    R["activity"][f"top1pct_{k}_share_of_ratings"] = float(cum.iloc[max(1, len(s2)//100) - 1])
    R["activity"][f"top10pct_{k}_share_of_ratings"] = float(cum.iloc[max(1, len(s2)//10) - 1])
top_users = ru.sort_values(ascending=False).head(10)
R["activity"]["top10_users"] = {int(k): int(v) for k, v in top_users.items()}

# ---------- sparsity ----------
nu, nb = ratings["User-ID"].nunique(), ratings["ISBN"].nunique()
R["sparsity"] = {"users": nu, "items": nb, "interactions": len(ratings),
                 "density": len(ratings) / (nu * nb)}
# filtered core: iterative k-core
def kcore(df, ku, ki, max_iter=20):
    for _ in range(max_iter):
        n0 = len(df)
        ic = df.groupby("ISBN")["User-ID"].transform("size")
        df = df[ic >= ki]
        uc = df.groupby("User-ID")["ISBN"].transform("size")
        df = df[uc >= ku]
        if len(df) == n0:
            break
    return df
cores = {}
base = ratings[in_books]
for ku, ki in [(5, 5), (10, 10), (20, 20)]:
    c = kcore(base, ku, ki)
    cores[f"{ku}u_{ki}i"] = {"interactions": len(c), "users": c["User-ID"].nunique(),
                             "items": c["ISBN"].nunique(),
                             "density": len(c) / max(1, c["User-ID"].nunique() * c["ISBN"].nunique()),
                             "share_of_interactions_kept": len(c) / len(ratings),
                             "implicit_share": float((c["Book-Rating"] == 0).mean())}
R["kcore"] = cores

# ---------- users ----------
age = users["Age"]
R["users"]["age_missing_share"] = float(age.isna().mean())
R["users"]["age_implausible"] = int(((age < 5) | (age > 100)).sum())
R["users"]["age_plausible_median"] = float(age[(age >= 5) & (age <= 100)].median())
country = users["Location"].str.split(",").str[-1].str.strip().replace("", np.nan)
R["users"]["top_countries"] = {k: int(v) for k, v in country.value_counts().head(10).items()}
R["users"]["country_missing_or_blank"] = int(country.isna().sum())
active = set(ru.index)
R["users"]["top_countries_active"] = {k: int(v) for k, v in country[users["User-ID"].isin(active)].value_counts().head(8).items()}

# ---------- book metadata ----------
yr = pd.to_numeric(books["Year-Of-Publication"], errors="coerce")
R["books"]["year_non_numeric"] = int(yr.isna().sum())
R["books"]["year_zero"] = int((yr == 0).sum())
R["books"]["year_future"] = int((yr > 2006).sum())
R["books"]["year_valid_range"] = [int(yr[(yr > 1800) & (yr <= 2006)].min()), int(yr[(yr > 1800) & (yr <= 2006)].max())]
R["books"]["year_p50"] = float(yr[(yr > 1800) & (yr <= 2006)].median())
R["books"]["non_numeric_examples"] = books.loc[yr.isna(), ["ISBN", "Book-Title", "Book-Author", "Year-Of-Publication", "Publisher"]].head(5).to_dict("records")
R["books"]["unique_authors"] = int(books["Book-Author"].nunique())
R["books"]["unique_publishers"] = int(books["Publisher"].nunique())
R["books"]["unique_titles"] = int(books["Book-Title"].nunique())
# same title+author under multiple ISBNs (editions)
ta = books["Book-Title"].str.lower().str.strip() + "||" + books["Book-Author"].str.lower().str.strip()
R["books"]["isbns_sharing_title_author"] = int(ta.duplicated(keep=False).sum())
R["books"]["title_author_groups_with_multiple_isbns"] = int((ta.value_counts() > 1).sum())
# image url sanity
R["books"]["img_url_missing"] = int(books["Image-URL-M"].isna().sum())
R["books"]["genre_or_description_column"] = False

# ---------- most-rated books ----------
rb_df = ratings[in_books].groupby("ISBN").agg(n=("Book-Rating", "size"),
            n_expl=("Book-Rating", lambda s: (s > 0).sum()),
            mean_expl=("Book-Rating", lambda s: s[s > 0].mean() if (s > 0).any() else np.nan))
rb_df = rb_df.join(books.set_index("ISBN")[["Book-Title", "Book-Author"]])
R["top_books"] = rb_df.sort_values("n", ascending=False).head(10).reset_index().to_dict("records")
# edition effect: work-level aggregation
rk = ratings[in_books].merge(books[["ISBN"]].assign(work=ta.values), on="ISBN")
R["work_level"] = {"unique_works_rated": int(rk["work"].nunique()),
                   "unique_isbns_rated": int(rk["ISBN"].nunique())}

json.dump(R, open(O / "eda_results.json", "w"), indent=2, default=str, ensure_ascii=False)

# save arrays for plotting
np.save(O / "per_user.npy", ru.values); np.save(O / "per_book.npy", rb.values)
print(json.dumps(R, indent=1, default=str, ensure_ascii=False))
