import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats.qmc import Sobol, LatinHypercube

# =========================
# Config
# =========================
N_ROUNDS = 100
N_PER_ROUND = 10
DIM = 6
SEED = 42

OUT_DIR = Path("sampling_compare_hartmann6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# Hartmann6 (minimize)
# =========================
def hartmann6(x):
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([
        [10,   3, 17, 3.50,  1.7,  8],
        [0.05,10, 17, 0.1,   8,   14],
        [3,   3.5,1.7,10,   17,    8],
        [17,   8,0.05,10,   0.1,  14]
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569,  124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091,  381]
    ])
    s = 0.0
    for i in range(4):
        s += alpha[i] * np.exp(-np.sum(A[i] * (x - P[i]) ** 2))
    return -s

def evaluate_batch(X):
    return np.array([hartmann6(x) for x in X], dtype=float)

# =========================
# Sampling methods
# =========================
def sample_random(n, dim, rng):
    return rng.uniform(0.0, 1.0, size=(n, dim))

def sample_lhs(n, dim, seed):
    sampler = LatinHypercube(d=dim, seed=seed)
    return sampler.random(n=n)

def sample_sobol_all(n_total, dim, seed):
    sampler = Sobol(d=dim, scramble=True, seed=seed)
    return sampler.random(n=n_total)

# =========================
# Run one method
# =========================
def run_method_random(n_rounds=100, n_per_round=10, dim=6, seed=42):
    rng = np.random.default_rng(seed)
    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        X = sample_random(n_per_round, dim, rng)
        y = evaluate_batch(X)
        round_min = float(np.min(y))

        round_rows.append({
            "method": "random",
            "round": r + 1,
            "round_min_yield": round_min,
        })

        for i in range(n_per_round):
            row = {
                "method": "random",
                "round": r + 1,
                **{f"x{j+1}": float(X[i, j]) for j in range(dim)},
                "yield": float(y[i]),
            }
            point_rows.append(row)

    return pd.DataFrame(round_rows), pd.DataFrame(point_rows)

def run_method_lhs(n_rounds=100, n_per_round=10, dim=6, seed=42):
    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        X = sample_lhs(n_per_round, dim, seed=seed + r)
        y = evaluate_batch(X)
        round_min = float(np.min(y))

        round_rows.append({
            "method": "lhs",
            "round": r + 1,
            "round_min_yield": round_min,
        })

        for i in range(n_per_round):
            row = {
                "method": "lhs",
                "round": r + 1,
                **{f"x{j+1}": float(X[i, j]) for j in range(dim)},
                "yield": float(y[i]),
            }
            point_rows.append(row)

    return pd.DataFrame(round_rows), pd.DataFrame(point_rows)

def run_method_sobol(n_rounds=100, n_per_round=10, dim=6, seed=42):
    n_total = n_rounds * n_per_round
    X_all = sample_sobol_all(n_total=n_total, dim=dim, seed=seed)

    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        start = r * n_per_round
        end = (r + 1) * n_per_round
        X = X_all[start:end]
        y = evaluate_batch(X)
        round_min = float(np.min(y))

        round_rows.append({
            "method": "sobol",
            "round": r + 1,
            "round_min_yield": round_min,
        })

        for i in range(n_per_round):
            row = {
                "method": "sobol",
                "round": r + 1,
                **{f"x{j+1}": float(X[i, j]) for j in range(dim)},
                "yield": float(y[i]),
            }
            point_rows.append(row)

    return pd.DataFrame(round_rows), pd.DataFrame(point_rows)

# =========================
# Summary
# =========================
def summarize_round_min(df_round):
    rows = []
    for method, g in df_round.groupby("method"):
        vals = g["round_min_yield"].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "n_rounds": int(len(vals)),
            "mean": float(np.mean(vals)),
            "var": float(np.var(vals, ddof=1)),  # sample variance
            "std": float(np.std(vals, ddof=1)),  # sample std
        })
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

# =========================
# Main
# =========================
def main():
    sobol_round, sobol_points = run_method_sobol(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, dim=DIM, seed=SEED
    )
    lhs_round, lhs_points = run_method_lhs(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, dim=DIM, seed=SEED
    )
    random_round, random_points = run_method_random(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, dim=DIM, seed=SEED
    )

    round_all = pd.concat([sobol_round, lhs_round, random_round], ignore_index=True)
    points_all = pd.concat([sobol_points, lhs_points, random_points], ignore_index=True)
    summary = summarize_round_min(round_all)

    round_all.to_csv(OUT_DIR / "round_min_yield.csv", index=False)
    points_all.to_csv(OUT_DIR / "all_sampled_points.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_stats.csv", index=False)

    print("===== Summary stats =====")
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(OUT_DIR / "round_min_yield.csv")
    print(OUT_DIR / "all_sampled_points.csv")
    print(OUT_DIR / "summary_stats.csv")

if __name__ == "__main__":
    main()