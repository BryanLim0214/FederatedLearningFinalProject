# RankBridge 🔐🌐

**Mitigating Heterogeneity in Federated Phishing Detection via Privacy-Preserving Rank-Based Explanation Clustering**

> Author: Panhapiseth Lim — University of Texas Permian Basin, Spring 2026

---

## What Is This?

RankBridge is a federated learning system for phishing detection that handles the core challenge of **client heterogeneity**: different organizations face completely different kinds of phishing attacks (banks face URL manipulation; hospitals face urgency-language social engineering). Naively averaging all clients' models together (standard FedAvg) blends these differences away.

**RankBridge solves this by:**
1. Having each client compute SHAP feature importances locally (private, never shared)
2. Converting those importances to a **top-K ordinal rank list** (only indices, not magnitudes)
3. The server clusters clients by rank-list similarity (Spearman distance)
4. Aggregating models **within each cluster only** → cluster-specific models per threat profile

The rank list is the privacy contribution: it discards cardinal SHAP magnitudes, exposing only ordinal structure needed for grouping.

---

## Quick Start

### 1. Install Dependencies (CPU-only, no GPU required)

```bash
pip install -r requirements.txt
```

> On Windows, use PowerShell or Command Prompt. Python 3.10+ recommended.

### 2. Run the Full RankBridge Experiment (uses synthetic data, ~10-20 min)

```bash
# Step 1: Generate client splits (16 Non-IID clients, synthetic data)
python -m src.data_prep

# Step 2: Run RankBridge (30 rounds, 3 seeds)
python -m src.experiments.run_fedrankx

# Step 3: Run all baselines
python -m src.experiments.run_baselines

# Step 4: Generate paper tables and figures
python -m src.evaluate
```

### 3. Run Ablation Study

```bash
python -m src.experiments.ablation

# Or a single ablation:
python -m src.experiments.ablation --ablation topk
```

---

## Using Real Datasets (Recommended for Paper)

### Dataset A: ISCX-URL2016
1. Download from: https://www.unb.ca/cic/datasets/url-2016.html
2. Place `All.csv` at `data/raw/ISCX-URL2016/All.csv`

### Dataset B: Phishing Email (Kaggle)
1. Download from: https://www.kaggle.com/datasets/subhajournal/phishingemails
2. Place `Phishing_Email.csv` at `data/raw/phishing_email/Phishing_Email.csv`

Then run with `--use_real_data`:
```bash
python -m src.data_prep --use_real_data
python -m src.experiments.run_fedrankx --use_real_data
python -m src.experiments.run_baselines --use_real_data
```

---

## Project Structure

```
RankBridge/
├── src/
│   ├── data_prep.py           # Dataset loading + Non-IID client splitting
│   ├── rank_utils.py          # SHAP extraction + rank conversion + privacy analysis
│   ├── clustering.py          # Rank-distance matrix + hierarchical/k-medoids clustering
│   ├── client.py              # Flower FlowerClient (LightGBM + SHAP rank)
│   ├── fedrank_strategy.py    # RankBridge custom Flower strategy (key contribution)
│   ├── server.py              # Flower simulation server entrypoint
│   ├── evaluate.py            # Result loading + paper tables + figures
│   ├── baselines/
│   │   ├── local_only.py      # LocalOnly (lower bound)
│   │   ├── fedavg_standard.py # Standard FedAvg (no clustering)
│   │   ├── fedclust.py        # FedClust (weight-cosine clustering)
│   │   └── ifca.py            # IFCA (Ghosh et al. 2020)
│   └── experiments/
│       ├── run_fedrankx.py    # Main RankBridge experiment (multi-seed)
│       ├── run_baselines.py   # All baselines (multi-seed)
│       └── ablation.py        # Ablation study sweeps
├── data/
│   ├── raw/                   # Place real datasets here
│   └── processed/
│       └── client_splits/     # Auto-generated Non-IID splits
├── results/                   # JSON outputs + CSV tables + figures
├── plots/                     # High-res PNG paper figures
├── RankBridge_Paper_Summary.md # Complete manuscript methodology and findings
├── requirements.txt
└── README.md
```

---

## Key CLI Options

| Script | Key Options |
|--------|------------|
| `data_prep.py` | `--n_clients 16`, `--samples_per_client 1500`, `--use_real_data` |
| `server.py` | `--n_rounds 30`, `--top_k 15`, `--rank_metric spearman`, `--n_clusters 5` |
| `run_fedrankx.py` | `--n_seeds 3`, `--n_rounds 50`, `--top_k 10` |
| `run_baselines.py` | `--methods fedavg fedclust`, `--n_rounds 30` |
| `ablation.py` | `--ablation topk\|metric\|linkage\|rounds\|all` |

---

## Methodology Summary

| Component | Choice | Justification |
|-----------|--------|--------------|
| Local model | LightGBM | Fastest SHAP (TreeExplainer), best tabular performance |
| XAI method | SHAP TreeExplainer | 200x faster than KernelExplainer on trees |
| Privacy signal | Top-K rank list | Discards magnitudes; O(K) not O(F) communication |
| Distance metric | Spearman ρ | Handles ties, robust to rank-set overlap |
| Clustering | Ward linkage | Minimises intra-cluster variance |
| FL framework | Flower (flwr) | CPU simulation, no ray server needed |

---

## Paper Claims & Verification

| Claim | How Verified |
|-------|-------------|
| RankBridge outperforms FedAvg on heterogeneous clients | F1 comparison across N=32 hard non-IID clients |
| Rank lists leak less information than raw SHAP | Reconstruction RMSE analysis (evaluate.py) |
| Clusters match true organizational types | Near-perfect NMI/ARI vs. ground-truth domain labels |
| Communication overhead from ranking is negligible | K×2 bytes vs. model size comparison |

---

## Hardware Requirements

- **CPU:** Any modern multi-core (4+ cores recommended for parallel client execution)
- **RAM:** 8–16 GB
- **GPU:** Not required
- **OS:** Windows 10/11, Linux, or macOS with Python 3.10+
- **Estimated runtime:** ~10–30 min per experiment (30 rounds, 32 clients, real data)
