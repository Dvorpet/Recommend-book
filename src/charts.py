"""EDA charts for the Book-Crossing recommender PoC.
Usage: python charts.py <data_dir> <out_dir>
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D, O = Path(sys.argv[1]), Path(sys.argv[2]); O.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, INK2, GRID, BG = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "figure.facecolor": BG, "axes.facecolor": BG, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13, "axes.titlelocation": "left"})

books = pd.read_csv(D/"Books.csv", dtype=str, keep_default_na=False, na_values=[""])
r = pd.read_csv(D/"Ratings.csv", dtype={"ISBN": str})

def fmt_k(v):
    return f"{v/1e6:.2f} M".replace(".", ",") if v >= 1e6 else (f"{v/1e3:.0f} k" if v >= 1e3 else f"{v:.0f}")

# 1 ---- rating distribution
dist = r["Book-Rating"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4.2))
cols = [ORANGE] + [BLUE]*10
ax.bar(dist.index, dist.values, color=cols, width=0.72)
ax.set_xticks(range(11)); ax.set_xlabel("Book-Rating"); ax.set_ylabel("Počet hodnocení")
ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_k(v)))
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("62 % interakcí je implicitních (rating 0)\nExplicitní hodnocení jsou silně pozitivní")
ax.annotate("0 = implicitní interakce\n716 k (62 %)", (0, dist[0]), xytext=(1.2, dist[0]*0.85),
            color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
ax.text(10.4, dist[8]*1.02, "explicitní 1–10: průměr 7,6\nmedián 8", ha="right", va="bottom", color=INK2, fontsize=10)
fig.tight_layout(); fig.savefig(O/"01_rating_distribution.png", dpi=160); plt.close(fig)

# 2 ---- long tail: users & books by activity bucket
bins = [0, 1, 4, 9, 19, 99, 10**9]; labels = ["1", "2–4", "5–9", "10–19", "20–99", "100+"]
ru = r.groupby("User-ID").size(); rb = r.groupby("ISBN").size()
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, s, name in [(axes[0], ru, "uživatelů"), (axes[1], rb, "knih")]:
    c = pd.cut(s, bins=bins, labels=labels).value_counts().reindex(labels)
    share = c / c.sum() * 100
    bars = ax.bar(labels, share.values, color=[ORANGE] + [BLUE]*5, width=0.7)
    for b, v in zip(bars, share.values):
        ax.text(b.get_x()+b.get_width()/2, v+1, f"{v:.0f} %", ha="center", color=INK, fontsize=10)
    ax.set_xlabel("Počet interakcí"); ax.set_title(f"Podíl {name} ({len(s):,} celkem)".replace(",", " "))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel("% entit"); axes[0].set_ylim(0, 68)
fig.suptitle("Dlouhý chvost: 56 % uživatelů a 58 % knih má jedinou interakci", x=0.01, ha="left",
             fontweight="bold", fontsize=14, color=INK)
fig.tight_layout(); fig.savefig(O/"02_long_tail.png", dpi=160); plt.close(fig)

# 3 ---- usable data funnel
steps = [("Všechny interakce (Ratings.csv)", 1149780, "105 k uživatelů · 341 k ISBN"),
         ("ISBN existuje v Books.csv", 1031136, "90 % · 270 k knih s metadaty"),
         ("5-core (≥5 interakcí uživatel i kniha)", 564718, "49 % · 15,2 k uživatelů · 36 k knih"),
         ("10-core", 365894, "32 % · 6,6 k uživatelů · 13,2 k knih"),
         ("20-core", 195445, "17 % · 2,6 k uživatelů · 4,3 k knih")]
fig, ax = plt.subplots(figsize=(10, 4.2))
y = np.arange(len(steps))[::-1]
ax.barh(y, [s[1] for s in steps], color=[BLUE, BLUE, ORANGE, BLUE, BLUE], height=0.62)
for yi, (lab, v, note) in zip(y, steps):
    ax.text(v + 15000, yi, f"{fmt_k(v)}  ·  {note}", va="center", color=INK, fontsize=10)
ax.set_yticks(y); ax.set_yticklabels([s[0] for s in steps]); ax.set_xlim(0, 1.95e6)
ax.xaxis.set_visible(False); ax.spines["bottom"].set_visible(False)
ax.set_title("Kolik dat zbude pro collaborative filtering\nHustota matice: 0,003 % (vše) → 0,1 % (5-core)")
fig.tight_layout(); fig.savefig(O/"03_usable_data_funnel.png", dpi=160); plt.close(fig)

# 4 ---- popularity trap: count vs explicit mean
m = r[r.ISBN.isin(set(books.ISBN))]
g = m.groupby("ISBN").agg(n=("Book-Rating", "size"),
                          n_e=("Book-Rating", lambda s: (s > 0).sum()),
                          mean_e=("Book-Rating", lambda s: s[s > 0].mean() if (s > 0).any() else np.nan))
g = g[(g.n >= 100) & (g.n_e >= 20)].join(books.set_index("ISBN")["Book-Title"])
fig, ax = plt.subplots(figsize=(9, 4.8))
wa = g.index == "0971880107"
ax.scatter(g.n[~wa], g.mean_e[~wa], s=22, color=BLUE, alpha=0.55, edgecolor=BG, lw=0.6)
ax.scatter(g.n[wa], g.mean_e[wa], s=70, color=ORANGE, edgecolor=BG, lw=1.5, zorder=3)
ax.annotate("Wild Animus — nejčastější kniha v datech\n2 502 interakcí, ale průměr jen 4,4/10\n(77 % interakcí je implicitních)",
            (g.n[wa].iloc[0], g.mean_e[wa].iloc[0]), xytext=(700, 5.0), color=INK, fontsize=10,
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
for isbn, lab in [("0316666343", "The Lovely Bones"), ("0385504209", "The Da Vinci Code")]:
    if isbn in g.index:
        ax.annotate(lab, (g.loc[isbn, "n"], g.loc[isbn, "mean_e"]), xytext=(8, 6), textcoords="offset points",
                    color=INK2, fontsize=9)
ax.set_xscale("log"); ax.set_xlabel("Počet interakcí (log)"); ax.set_ylabel("Průměr explicitních hodnocení")
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True); ax.set_ylim(3.5, 10)
ax.set_title("Popularity trap: nejvíc interakcí ≠ nejlepší doporučení\nKnihy se ≥100 interakcemi (n = 727)")
fig.tight_layout(); fig.savefig(O/"04_popularity_trap.png", dpi=160); plt.close(fig)
print("ok", len(g))
