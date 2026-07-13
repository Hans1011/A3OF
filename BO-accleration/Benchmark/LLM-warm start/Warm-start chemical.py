import numpy as np
import pandas as pd
from scipy.stats.qmc import Sobol, LatinHypercube
from pathlib import Path

# =========================
# 1. Problem definition
# =========================
N_ROUNDS = 25
N_PER_ROUND = 3
SEED = 42

CATS_SOLVENT = ["MeOH", "THF", "Dioxane"]

# Optional output files
OUT_DIR = Path("sampling_compare_chemical")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Yield function
# Same logic as your previous code
# =========================
T0 = 25
T1 = 100
e0 = np.exp((T1 + 0) / T0)
e60 = np.exp((T1 + 60) / T0)
de = e60 - e0

boiling_points = {"MeOH": 64.7, "THF": 66.0, "Dioxane": 101.0}
density = {"MeOH": 0.792, "THF": 0.886, "Dioxane": 1.03}
solvent_descriptors = pd.DataFrame(
    {"boiling_points": boiling_points, "density": density}
)

def calc_volume_fact(V):
    x = (V - 20) / 70
    x = 0.5 + (x - 0.75) * 0.1 + (x - 0.4) ** 2
    return x

def calc_rhofact(solvent_type, Tfact):
    x = solvent_descriptors["density"][solvent_type]
    x = (1.5 - x) * (Tfact + 0.5) / 2
    return x.values

def calc_Tfact(T):
    x = np.exp((T1 + T) / T0)
    return (x - e0) / de

def evaluate_yield(df, A=25, B=90):
    T = df["Temperature"].values.astype(float)
    V = df["Solvent Volume"].values.astype(float)
    S = df["Solvent Type"].values.astype(str)

    Tfact = calc_Tfact(T)
    rhofact = calc_rhofact(S, Tfact)
    Vfact = calc_volume_fact(V)

    y_core = A * Tfact + B * rhofact
    y_core = 0.5 * y_core + 0.5 * y_core * Vfact
    return y_core

# =========================
# 3. Sampling helpers
# =========================
def scale_continuous(x01):
    """
    x01: array of shape [n, 2] in [0,1]
    col0 -> Temperature in [0,60]
    col1 -> Solvent Volume in [20,90]
    """
    x01 = np.asarray(x01, dtype=float)
    T = 0.0 + x01[:, 0] * (60.0 - 0.0)
    V = 20.0 + x01[:, 1] * (90.0 - 20.0)
    return T, V

def make_dataframe(T, V, S):
    return pd.DataFrame({
        "Temperature": T,
        "Solvent Volume": V,
        "Solvent Type": S,
    })

def sample_random(n, rng):
    x01 = rng.uniform(0.0, 1.0, size=(n, 2))
    T, V = scale_continuous(x01)
    S = rng.choice(CATS_SOLVENT, size=n, replace=True)
    return make_dataframe(T, V, S)

def sample_lhs(n, seed):
    sampler = LatinHypercube(d=2, seed=seed)
    x01 = sampler.random(n=n)
    T, V = scale_continuous(x01)

    rng = np.random.default_rng(seed + 10000)
    S = rng.choice(CATS_SOLVENT, size=n, replace=True)
    return make_dataframe(T, V, S)

def sample_sobol_all(n_total, seed):
    """
    Sobol is best generated as one continuous sequence, then chunked by round.
    """
    sampler = Sobol(d=2, scramble=True, seed=seed)
    x01 = sampler.random(n=n_total)
    T, V = scale_continuous(x01)

    rng = np.random.default_rng(seed + 20000)
    S = rng.choice(CATS_SOLVENT, size=n_total, replace=True)
    return make_dataframe(T, V, S)

# =========================
# 4. Run one method
# =========================
def run_method_random(n_rounds=25, n_per_round=3, seed=42):
    rng = np.random.default_rng(seed)
    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        df = sample_random(n_per_round, rng)
        df["Yield"] = evaluate_yield(df)

        round_max = float(df["Yield"].max())
        round_rows.append({"method": "random", "round": r + 1, "round_max_yield": round_max})

        df = df.copy()
        df["method"] = "random"
        df["round"] = r + 1
        point_rows.append(df)

    return pd.DataFrame(round_rows), pd.concat(point_rows, ignore_index=True)

def run_method_lhs(n_rounds=25, n_per_round=3, seed=42):
    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        df = sample_lhs(n_per_round, seed=seed + r)
        df["Yield"] = evaluate_yield(df)

        round_max = float(df["Yield"].max())
        round_rows.append({"method": "lhs", "round": r + 1, "round_max_yield": round_max})

        df = df.copy()
        df["method"] = "lhs"
        df["round"] = r + 1
        point_rows.append(df)

    return pd.DataFrame(round_rows), pd.concat(point_rows, ignore_index=True)

def run_method_sobol(n_rounds=25, n_per_round=3, seed=42):
    n_total = n_rounds * n_per_round
    df_all = sample_sobol_all(n_total=n_total, seed=seed)
    df_all["Yield"] = evaluate_yield(df_all)

    round_rows = []
    point_rows = []

    for r in range(n_rounds):
        start = r * n_per_round
        end = (r + 1) * n_per_round
        df = df_all.iloc[start:end].copy()

        round_max = float(df["Yield"].max())
        round_rows.append({"method": "sobol", "round": r + 1, "round_max_yield": round_max})

        df["method"] = "sobol"
        df["round"] = r + 1
        point_rows.append(df)

    return pd.DataFrame(round_rows), pd.concat(point_rows, ignore_index=True)

# =========================
# 5. Summary stats
# =========================
def summarize_round_max(df_round):
    rows = []
    for method, g in df_round.groupby("method"):
        vals = g["round_max_yield"].to_numpy(dtype=float)
        rows.append({
            "method": method,
            "n_rounds": len(vals),
            "mean": float(np.mean(vals)),
            "var": float(np.var(vals, ddof=1)),   # sample variance
            "std": float(np.std(vals, ddof=1)),   # sample std
        })
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)

# =========================
# 6. Main
# =========================
def main():
    sobol_round, sobol_points = run_method_sobol(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, seed=SEED
    )
    lhs_round, lhs_points = run_method_lhs(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, seed=SEED
    )
    random_round, random_points = run_method_random(
        n_rounds=N_ROUNDS, n_per_round=N_PER_ROUND, seed=SEED
    )

    round_all = pd.concat([sobol_round, lhs_round, random_round], ignore_index=True)
    points_all = pd.concat([sobol_points, lhs_points, random_points], ignore_index=True)

    summary = summarize_round_max(round_all)

    # save
    round_all.to_csv(OUT_DIR / "round_max_yield.csv", index=False)
    points_all.to_csv(OUT_DIR / "all_sampled_points.csv", index=False)
    summary.to_csv(OUT_DIR / "summary_stats.csv", index=False)

    print("===== Per-method summary =====")
    print(summary.to_string(index=False))

    print("\nSaved files:")
    print(OUT_DIR / "round_max_yield.csv")
    print(OUT_DIR / "all_sampled_points.csv")
    print(OUT_DIR / "summary_stats.csv")

if __name__ == "__main__":
    main()