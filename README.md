# Book recommendations: PoC kick-off

Hiring task for the AI Consultant / Architect role at DataSentics. The client is an online book shop with a 30 MD Proof of Concept to find out whether recommendation engines can drive sales.

The deliverable is a small website with two parts:

- **Kick-off presentation** (`site/index.html`), 15 slides in English with a Czech toggle. It covers scoping, team, delivery management, validation, PoC focus vs. later improvements, open questions, the PoC solution architecture and the production design.
- **Interactive prototype**: next step.

## Repository layout

| Path | What it is |
|---|---|
| `src/eda.py`, `src/eda2.py` | Exploratory data analysis of the three CSVs |
| `src/charts.py` | EDA charts (`docs/eda/*.png`) |
| `src/prep.py` | Cleaning: join to catalogue, merge editions into works, implicit signal, bulk-account removal, 5-core |
| `src/experiment.py` | Offline comparison: popularity, quality-weighted popularity, item-kNN (cosine, BM25), ALS |
| `docs/eda/eda_findings.md` | Written data findings and what they mean for the solution |
| `docs/experiment_results.json` | Offline metrics, tuning grid, similar-item examples |
| `site/index.html` | The presentation (single self-contained page) |

## Reproduce

1. Download the [Book-Crossing dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset) and put `Books.csv`, `Users.csv` and `Ratings.csv` in `data/`. The raw data is not committed.
2. Install the dependencies and run the steps:

```bash
pip install -r requirements.txt
python src/eda.py data docs/eda
python src/charts.py data docs/eda
python src/prep.py data artifacts
python src/experiment.py artifacts
```

## Headline results (offline, 15.6 k users × 29.4 k works)

| Model | NDCG@10 | Hit rate@10 | Catalogue coverage |
|---|---:|---:|---:|
| Most popular (count) | 0.019 | 8.1 % | 0.1 % |
| Popularity, quality-weighted | 0.008 | 3.8 % | 0.1 % |
| Item-kNN (cosine) | 0.035 | 9.7 % | 58.6 % |
| ALS | 0.054 | 18.2 % | 8.5 % |
| **Item-kNN (BM25)** | **0.066** | **19.2 %** | **76.5 %** |

Split: random 20 % per-user hold-out (no timestamps in the data), hyper-parameters tuned on a second hold-out. Because the split is random rather than time-based, these numbers are optimistic. On client data we would use a time-based split.
