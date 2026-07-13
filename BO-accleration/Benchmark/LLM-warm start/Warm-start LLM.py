import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR_1 = Path(r"Chemical")
ROOT_DIR_2 = Path(r"Testfunction")

OUT_CSV = Path(r"init_mean_compare-chemical.csv")
X_COLS = ["Yield"]
N_INIT = 3

from typing import Optional

def load_trace(run_dir: Path) -> Optional[pd.DataFrame]:

    path = run_dir / "bo_trace.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def get_init_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly extract the initial 10 samples.
    Priority:
      1) use is_initial column if exists
      2) fallback: iter==0
      3) fallback: first N rows
    """
    df = df.copy()

    # force numeric for x columns
    for c in X_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 1) if has is_initial, use it robustly
    if "is_initial" in df.columns:
        # convert to string lower for safe compare
        flag = df["is_initial"].astype(str).str.strip().str.lower()
        init_df = df[flag.isin(["true", "1", "yes"])].copy()
        if len(init_df) >= 1:
            return init_df.iloc[:N_INIT]

    # 2) fallback: iter==0
    if "iter" in df.columns:
        init_df = df[df["iter"] == 0].copy()
        if len(init_df) >= 1:
            return init_df.iloc[:N_INIT]

    # 3) fallback: first N rows
    return df.iloc[:N_INIT].copy()

def compute_init_mean(init_df: pd.DataFrame) -> np.ndarray:
    return init_df[X_COLS].mean(axis=0).to_numpy()

def process_root(root_dir: Path, tag: str):
    rows = []
    for i in range(100):
        run_name = f"run_{i:03d}"
        run_dir = root_dir / run_name
        if not run_dir.exists():
            continue

        df = load_trace(run_dir)
        if df is None or len(df) == 0:
            continue

        init_df = get_init_df(df)

        # Debug: how many init points extracted
        n_found = len(init_df)

        mean_vec = compute_init_mean(init_df)

        rows.append({
            "folder": tag,
            "run": run_name,
            "n_init_found": n_found,
            **{f"mean_{c}": float(mean_vec[j]) for j, c in enumerate(X_COLS)},
        })

    return pd.DataFrame(rows)

def main():
    df1 = process_root(ROOT_DIR_1, "out_pibo_100runs-1.2")
    df2 = process_root(ROOT_DIR_2, "out_pibo_100runs-1.18")

    out = pd.concat([df1, df2], ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    print(" Saved:", OUT_CSV)
    print("Preview:")
    print(out.head(10))

    # quick sanity check: show runs with missing init points
    bad = out[out["n_init_found"] < N_INIT]
    if len(bad) > 0:
        print(bad[["folder", "run", "n_init_found"]].head(20))

if __name__ == "__main__":
    main()
