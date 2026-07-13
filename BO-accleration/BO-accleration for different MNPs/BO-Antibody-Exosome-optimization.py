

import importlib
import inspect
import dataclasses as _dc
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from pydantic import BaseModel, Field

# ============================================================
# 0) Runtime config
# ============================================================

DEBUG = True

# Keep your original API configuration structure.
# For safety, the recommended practice is to set AGITCO_API_KEY in environment.
# If you insist on a hardcoded key, paste it into HARDCODED_API_KEY locally.
API_BASE = "https://api.agicto.cn/v1"
HARDCODED_API_KEY = "YOUR_API_KEY_HERE"
WARMSTART_MODEL = "deepseek-chat"

# Historical experiment file.
# It should contain at least:
#   SW,Rwater,ROil,Oil,SO,Ion,reward
EXPERIMENTS_CSV = Path("BO-test.csv")

# Optional SHAP file. If not found, the code falls back to uniform discrete prior.
SHAP_JSON_PATH = Path("shap_results_water_oil.json")

# Optional impact summary for LLM warmstart.
IMPACT_SUMMARY_PATH = Path("Prior_knowledge_from_used_MNP.txt")

# Output directory.
OUT_DIR = Path("water_oil_pibo_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of LLM warmstart suggestions when no valid reward history exists.
BO_INIT_N = 3

# Number of new BO recommendations when history exists.
N_RECOMMEND = 1

# PiBO hyperparameters.
SHAP_SIGN_EPS = 1e-6
MIN_POS_DIMS_TOTAL = 6
SOFT_TAU = 0.05
BETA = 2.0
REDUCE = "mean"  # "mean" or "max"

# If True, the output CSV appends the new recommendation row with reward = NaN.
APPEND_RECOMMENDATION_TO_HISTORY = True

# If True, automatically rename long CSV columns to short BoFire keys.
AUTO_RENAME_COLUMNS = True




def dbg(msg: str):
    if DEBUG:
        print(msg)


def dbg_block(title: str, text: str, chunk: int = 1200):
    if not DEBUG:
        return
    text = "" if text is None else str(text)
    print(f"\n========== {title} (len={len(text)}) ==========")
    for i in range(0, len(text), chunk):
        print(text[i:i + chunk])
    print("========== END ==========\n")


def extract_json(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r + 1]
    return s


def get_api_key() -> str:
    api_key = os.getenv("AGITCO_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key and HARDCODED_API_KEY and HARDCODED_API_KEY != "YOUR_API_KEY_HERE":
        api_key = HARDCODED_API_KEY
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set AGITCO_API_KEY / OPENAI_API_KEY, "
            "or paste your existing key into HARDCODED_API_KEY locally."
        )
    return api_key


# ============================================================
# 2) Domain definition: 6 discrete inputs -> maximize reward
# ============================================================

import bofire.strategies.api as strategies
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDM
from bofire.data_models.features.api import ContinuousOutput, CategoricalInput, DiscreteInput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.domain.api import Domain, Inputs, Outputs

SW = CategoricalInput(
    key="SW",
    categories=["ODEA", "LDEA", "CDEA"],
)

Rwater = DiscreteInput(
    key="Rwater",
    values=[0, 0.0002, 0.01, 0.005],
)

ROil = DiscreteInput(
    key="ROil",
    values=[0, 0.0002, 0.01, 0.005],
)

Oil = CategoricalInput(
    key="Oil",
    categories=[
        "PMX-10", "PMX-20", "PMX-50", "PMX-100",
        "PMX-200", "PMX-500", "PMX-1000",
        "PMX-30000", "PMX-60000", "7500",
        "FC-40", "PFPE", "mineral", "PMX-350",
    ],
)

SO = CategoricalInput(
    key="SO",
    categories=["AEO-5", "Pico-surf", "Perfluoro", "TEGO-410"],
)

Ion = DiscreteInput(
    key="Ion",
    values=[0, 0.5, 1, 1.5, 2, 2.5, 3],
)

# Keep this feature order stable. The PiBO prior transform below is written for this order.
inputs = Inputs(features=[SW, Rwater, ROil, Oil, SO, Ion])

objective1 = MaximizeObjective(w=1.0)
reward = ContinuousOutput(key="reward", objective=objective1)
outputs = Outputs(features=[reward])

domain = Domain(inputs=inputs, outputs=outputs)

X_COLS_RAW = ["SW", "Rwater", "ROil", "Oil", "SO", "Ion"]
Y_COL = "reward"

CATS_SW = list(SW.categories)
VALS_RWATER = [float(x) for x in Rwater.values]
VALS_ROIL = [float(x) for x in ROil.values]
CATS_OIL = list(Oil.categories)
CATS_SO = list(SO.categories)
VALS_ION = [float(x) for x in Ion.values]

COLUMN_RENAME_MAP = {
    "surfactant_in_water": "SW",
    "ratio_of_surfactant_in_water": "Rwater",
    "ratio_of_surfactant_in_oil": "ROil",
    "oil_type": "Oil",
    "surfactant_in_oil": "SO",
    "ion_concentration_in_water": "Ion",
}




def _nearest_allowed_numeric(x, allowed: List[float]) -> float:
    try:
        xv = float(x)
    except Exception:
        return float(allowed[0])
    arr = np.asarray(allowed, dtype=float)
    return float(arr[np.argmin(np.abs(arr - xv))])


def clean_candidate_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and snap candidate columns to the allowed discrete sets."""
    out = df.copy()

    if AUTO_RENAME_COLUMNS:
        out = out.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in out.columns})

    missing = [c for c in X_COLS_RAW if c not in out.columns]
    if missing:
        raise ValueError(f"Missing input columns: {missing}. Existing columns: {list(out.columns)}")

    out["SW"] = out["SW"].astype(str).str.strip()
    out["Oil"] = out["Oil"].astype(str).str.strip()
    out["SO"] = out["SO"].astype(str).str.strip()

    out["Rwater"] = out["Rwater"].apply(lambda z: _nearest_allowed_numeric(z, VALS_RWATER))
    out["ROil"] = out["ROil"].apply(lambda z: _nearest_allowed_numeric(z, VALS_ROIL))
    out["Ion"] = out["Ion"].apply(lambda z: _nearest_allowed_numeric(z, VALS_ION))

    for col, allowed in [("SW", CATS_SW), ("Oil", CATS_OIL), ("SO", CATS_SO)]:
        bad = sorted(set(out[col].dropna().astype(str)) - set(allowed))
        if bad:
            raise ValueError(f"Invalid values in {col}: {bad}. Allowed: {allowed}")

    return out


def load_experiments_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        dbg(f"[DATA] Historical CSV not found: {path}. Treating as no-history mode.")
        return pd.DataFrame(columns=X_COLS_RAW + [Y_COL])

    df = pd.read_csv(path)
    if AUTO_RENAME_COLUMNS:
        df = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})

    if Y_COL not in df.columns:
        df[Y_COL] = np.nan

    df = clean_candidate_df(df)
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

    return df[X_COLS_RAW + [Y_COL]].copy()


def get_valid_history(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    hist = df.dropna(subset=[Y_COL]).reset_index(drop=True)
    return hist[X_COLS_RAW + [Y_COL]].copy()


def generate_init_samples(n: int, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "SW": rng.choice(CATS_SW, size=n),
        "Rwater": rng.choice(VALS_RWATER, size=n),
        "ROil": rng.choice(VALS_ROIL, size=n),
        "Oil": rng.choice(CATS_OIL, size=n),
        "SO": rng.choice(CATS_SO, size=n),
        "Ion": rng.choice(VALS_ION, size=n),
    })


def row_key(row: pd.Series) -> Tuple:
    return (
        str(row["SW"]),
        float(row["Rwater"]),
        float(row["ROil"]),
        str(row["Oil"]),
        str(row["SO"]),
        float(row["Ion"]),
    )


def remove_already_tested(candidates: pd.DataFrame, history_all: pd.DataFrame) -> pd.DataFrame:
    if history_all.empty:
        return candidates.reset_index(drop=True)
    tested = {row_key(r) for _, r in history_all.iterrows()}
    keep_rows = []
    for _, r in candidates.iterrows():
        if row_key(r) not in tested:
            keep_rows.append(r)
    if not keep_rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.DataFrame(keep_rows).reset_index(drop=True)



from openai import OpenAI


class _InitDesignOutWaterOil(BaseModel):
    points: List[Dict[str, object]]
    notes: Optional[str] = None


def load_impact_summary() -> str:
    if not IMPACT_SUMMARY_PATH.exists():
        return ""
    try:
        return IMPACT_SUMMARY_PATH.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return IMPACT_SUMMARY_PATH.read_text(errors="ignore").strip()


def generate_init_points_water_oil_with_agitco(
    *,
    n_points: int,
    model: str = WARMSTART_MODEL,
    temperature: float = 0.7,
    max_retries: int = 2,
    problem_spec: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Return exactly n_points initial experiment candidates.
    columns = [SW, Rwater, ROil, Oil, SO, Ion]
    """
    client = OpenAI(api_key=get_api_key(), base_url=API_BASE)

    sys_msg = (
        "You are a top-tier Bayesian optimization assistant. "
        "Your task is to propose warmstart initial experiment conditions for a water-oil interface optimization problem. "
        "The objective is to maximize reward. "
        "Reward may represent MNP crossing completeness, uniformity, success rate, or experimental score. "
        "Return ONLY a valid JSON object, no markdown, no extra text. "
        "The output MUST start with '{' and end with '}'. "
        "Keep diversity and avoid repeated combinations."
    )

    accepted_rows: List[Dict[str, object]] = []

    def _dedupe_key(r: Dict[str, object]) -> Tuple:
        return (
            str(r["SW"]),
            float(r["Rwater"]),
            float(r["ROil"]),
            str(r["Oil"]),
            str(r["SO"]),
            float(r["Ion"]),
        )

    last_err = None

    for attempt in range(max_retries + 1):
        try:
            accepted_rows = []

            for round_idx in range(12):
                need = int(n_points - len(accepted_rows))
                if need <= 0:
                    break

                user_payload = {
                    "task": "warmstart_initial_design_water_oil_interface",
                    "n_points": int(need),
                    "variables": {
                        "SW": {"type": "categorical", "categories": CATS_SW},
                        "Rwater": {"type": "discrete", "values": VALS_RWATER},
                        "ROil": {"type": "discrete", "values": VALS_ROIL},
                        "Oil": {"type": "categorical", "categories": CATS_OIL},
                        "SO": {"type": "categorical", "categories": CATS_SO},
                        "Ion": {"type": "discrete", "values": VALS_ION},
                    },
                    "objective": {"type": "maximize", "name": "reward"},
                    "already_selected_points": accepted_rows,
                    "problem_spec": problem_spec or {},
                    "output_schema": {
                        "points": (
                            f"List of exactly {need} points. "
                            "Each point must have keys: SW, Rwater, ROil, Oil, SO, Ion."
                        ),
                        "notes": "Optional string.",
                    },
                    "hard_rules": [
                        "Return ONLY valid JSON.",
                        f"Return exactly {need} points in `points`.",
                        f"SW must be one of {CATS_SW}.",
                        f"Rwater must be one of {VALS_RWATER}.",
                        f"ROil must be one of {VALS_ROIL}.",
                        f"Oil must be one of {CATS_OIL}.",
                        f"SO must be one of {CATS_SO}.",
                        f"Ion must be one of {VALS_ION}.",
                        "Do not repeat already_selected_points.",
                    ],
                    "soft_rules": [
                        "Prefer combinations likely to improve MNP water-oil interface crossing reward.",
                        "Maintain diversity across SW, Oil, SO, and concentration values.",
                        "Use impact_summary as additional hints if present.",
                    ],
                }

                dbg(f"[AGITCO] warmstart attempt={attempt} round={round_idx} need={need} model={model}")
                dbg_block("[AGITCO] user_payload(json)", json.dumps(user_payload, ensure_ascii=False, indent=2))

                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    temperature=temperature,
                    top_p=0.9,
                )

                if resp is None or not getattr(resp, "choices", None):
                    raise RuntimeError("Agitco returned empty response.")

                msg = getattr(resp.choices[0], "message", None)
                content = getattr(msg, "content", None)
                if not content or not str(content).strip():
                    raise RuntimeError("Agitco message.content empty.")

                raw = str(content).strip()
                dbg_block("[AGITCO] raw_response", raw)

                data = json.loads(extract_json(raw))
                out = _InitDesignOutWaterOil.model_validate(data)

                seen = set(_dedupe_key(r) for r in accepted_rows)

                for p in out.points:
                    if len(accepted_rows) >= n_points:
                        break

                    sw = str(p.get("SW", "")).strip()
                    oil = str(p.get("Oil", "")).strip()
                    so = str(p.get("SO", "")).strip()

                    if sw not in CATS_SW or oil not in CATS_OIL or so not in CATS_SO:
                        continue

                    candidate = {
                        "SW": sw,
                        "Rwater": _nearest_allowed_numeric(p.get("Rwater", None), VALS_RWATER),
                        "ROil": _nearest_allowed_numeric(p.get("ROil", None), VALS_ROIL),
                        "Oil": oil,
                        "SO": so,
                        "Ion": _nearest_allowed_numeric(p.get("Ion", None), VALS_ION),
                    }

                    k = _dedupe_key(candidate)
                    if k in seen:
                        continue

                    accepted_rows.append(candidate)
                    seen.add(k)

                dbg_block("[AGITCO] accepted_rows_so_far", json.dumps(accepted_rows, ensure_ascii=False, indent=2))

            if len(accepted_rows) != n_points:
                raise ValueError(f"LLM failed to provide {n_points} valid points. got={len(accepted_rows)}")

            df = pd.DataFrame(accepted_rows, columns=X_COLS_RAW)
            dbg_block("[AGITCO] final_init_df", df.to_csv(index=False))
            return df

        except Exception as e:
            last_err = e
            dbg(f"[AGITCO] attempt failed: {e}")
            accepted_rows = []

    raise RuntimeError(f"Agitco warmstart failed after retries. Last error: {last_err}")



def _val_token(v) -> str:
    try:
        fv = float(v)
        if abs(fv - int(fv)) < 1e-12:
            s = str(int(fv))
        else:
            s = str(fv).rstrip("0").rstrip(".")
    except Exception:
        s = str(v)
    return s.replace(".", "p").replace("-", "m")


def _val_key(v) -> str:
    try:
        fv = float(v)
        return str(fv).rstrip("0").rstrip(".")
    except Exception:
        return str(v)


def one_hot_df_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df_raw = clean_candidate_df(df_raw)
    parts = []

    sw = pd.get_dummies(df_raw["SW"].astype(str), prefix="SW")
    for c in CATS_SW:
        col = f"SW_{c}"
        if col not in sw.columns:
            sw[col] = 0.0
    parts.append(sw[[f"SW_{c}" for c in CATS_SW]])

    rw = pd.DataFrame(index=df_raw.index)
    for v in VALS_RWATER:
        col = f"Rwater_{_val_token(v)}"
        rw[col] = (df_raw["Rwater"].astype(float) == float(v)).astype(float)
    parts.append(rw)

    ro = pd.DataFrame(index=df_raw.index)
    for v in VALS_ROIL:
        col = f"ROil_{_val_token(v)}"
        ro[col] = (df_raw["ROil"].astype(float) == float(v)).astype(float)
    parts.append(ro)

    oil = pd.get_dummies(df_raw["Oil"].astype(str), prefix="Oil")
    for c in CATS_OIL:
        col = f"Oil_{c}"
        if col not in oil.columns:
            oil[col] = 0.0
    parts.append(oil[[f"Oil_{c}" for c in CATS_OIL]])

    so = pd.get_dummies(df_raw["SO"].astype(str), prefix="SO")
    for c in CATS_SO:
        col = f"SO_{c}"
        if col not in so.columns:
            so[col] = 0.0
    parts.append(so[[f"SO_{c}" for c in CATS_SO]])

    ion = pd.DataFrame(index=df_raw.index)
    for v in VALS_ION:
        col = f"Ion_{_val_token(v)}"
        ion[col] = (df_raw["Ion"].astype(float) == float(v)).astype(float)
    parts.append(ion)

    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    return out.astype(float)


def encode_df_like_shap_json(df_raw: pd.DataFrame, shap_feature_names: List[str]) -> pd.DataFrame:

    df_raw = clean_candidate_df(df_raw)
    out = pd.DataFrame(index=df_raw.index)

    for name in shap_feature_names:
        if name in ["Rwater", "ROil", "Ion"]:
            out[name] = pd.to_numeric(df_raw[name], errors="coerce").astype(float)

        elif name.startswith("SW_"):
            val = name.replace("SW_", "")
            out[name] = (df_raw["SW"].astype(str) == val).astype(float)

        elif name.startswith("Oil_"):
            val = name.replace("Oil_", "")
            out[name] = (df_raw["Oil"].astype(str) == val).astype(float)

        elif name.startswith("SO_"):
            val = name.replace("SO_", "")
            out[name] = (df_raw["SO"].astype(str) == val).astype(float)

        else:
            raise ValueError(f"Unknown SHAP feature name: {name}")

    return out[shap_feature_names].astype(float)


def load_shap_payload_json(path: str) -> Dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    oh_names = payload["onehot_detail"]["feature_names_onehot"]
    phi_dN = np.asarray(payload["onehot_detail"]["phi_mean_onehot"]["data"], dtype=float)

    X_raw = payload["history"]["X"]["data"]
    y = np.asarray(payload["history"]["y"]["data"], dtype=float)

    df_raw = pd.DataFrame(X_raw, columns=X_COLS_RAW)
    df_raw = clean_candidate_df(df_raw)

    X_oh_df = encode_df_like_shap_json(df_raw, oh_names)
    X_oh = X_oh_df.values

    if list(X_oh_df.columns) != list(oh_names):
        print("[WARN] Encoded columns are not identical to SHAP feature_names_onehot.")
        print("[WARN] Encoded columns:", list(X_oh_df.columns))
        print("[WARN] SHAP columns:", list(oh_names))

    assert X_oh.shape[1] == phi_dN.shape[0], (
        f"dim mismatch X_oh={X_oh.shape}, phi={phi_dN.shape}, "
        f"X columns={list(X_oh_df.columns)}, SHAP columns={list(oh_names)}"
    )

    return {
        "feature_names_onehot": oh_names,
        "history_raw_df": df_raw,
        "history_onehot": X_oh,
        "history_onehot_df": X_oh_df,
        "y": y,
        "phi_dN": phi_dN,
        "payload": payload,
    }


def uniform_discrete_prior() -> Dict[str, Dict[str, float]]:
    variable_values = {
        "SW": CATS_SW,
        "Rwater": VALS_RWATER,
        "ROil": VALS_ROIL,
        "Oil": CATS_OIL,
        "SO": CATS_SO,
        "Ion": VALS_ION,
    }
    prior = {}
    for var, values in variable_values.items():
        w = np.ones(len(values), dtype=float) / len(values)
        prior[var] = {_val_key(v): float(wj) for v, wj in zip(values, w)}
    return prior


def build_prior_from_shap_sign_only_discrete(
    shap_json_path: str,
    *,
    problem: Literal["max", "min"] = "max",
    shap_sign_eps: float = 1e-6,
    min_positive_dims_total: int = 6,
    smooth: float = 1e-6,
) -> Tuple[Dict[str, Dict[str, float]], Dict]:
    """
    Build a fully discrete prior from SHAP signs.

    Rules retained from original template:
      - no topk / topq
      - no argsort ranking
      - no use of y for selecting samples
      - only SHAP positive/negative sign rules
      - eps -> 0 relaxation allowed
      - min_positive_dims relaxation allowed
    """
    if not Path(shap_json_path).exists():
        prior = uniform_discrete_prior()
        debug = {
            "fallback": "shap_json_missing_uniform_discrete_prior",
            "path": str(shap_json_path),
            "M": 0,
            "prior_weights": prior,
        }
        return prior, debug

    pack = load_shap_payload_json(shap_json_path)

    names = pack["feature_names_onehot"]
    phi_dN = np.asarray(pack["phi_dN"], dtype=float)
    df_raw = pack["history_raw_df"]

    N = len(df_raw)
    idx = {n: i for i, n in enumerate(names)}

    variable_values = {
        "SW": CATS_SW,
        "Rwater": VALS_RWATER,
        "ROil": VALS_ROIL,
        "Oil": CATS_OIL,
        "SO": CATS_SO,
        "Ion": VALS_ION,
    }

    def col_name(var: str, val) -> str:
        if var in ["Rwater", "ROil", "Ion"]:
            return var

        return f"{var}_{val}"

    active_indices = []
    active_values = []

    for i in range(N):
        row = df_raw.iloc[i]
        vals_i = {
            "SW": str(row["SW"]),
            "Rwater": float(row["Rwater"]),
            "ROil": float(row["ROil"]),
            "Oil": str(row["Oil"]),
            "SO": str(row["SO"]),
            "Ion": float(row["Ion"]),
        }

        idx_i = []
        val_i = {}

        for var, val in vals_i.items():
            cname = col_name(var, val)
            if cname not in idx:
                raise KeyError(
                    f"Cannot find active one-hot column `{cname}` in SHAP feature names. "
                    f"Check one-hot naming consistency. Available examples: {names[:10]} ..."
                )
            idx_i.append(idx[cname])
            val_i[var] = val

        active_indices.append(idx_i)
        active_values.append(val_i)

    active_phi = np.zeros((N, 6), dtype=float)
    for i in range(N):
        active_phi[i, :] = phi_dN[active_indices[i], i]

    def make_mask(eps: float, min_dims: int) -> np.ndarray:
        if problem == "max":
            return (active_phi > eps).sum(axis=1) >= int(min_dims)
        return (active_phi < -eps).sum(axis=1) >= int(min_dims)

    tried = []

    mask = make_mask(shap_sign_eps, min_positive_dims_total)
    tried.append(("eps", shap_sign_eps, "min_dims", min_positive_dims_total, "M", int(mask.sum())))

    if mask.sum() == 0:
        mask = make_mask(0.0, min_positive_dims_total)
        tried.append(("eps", 0.0, "min_dims", min_positive_dims_total, "M", int(mask.sum())))

    if mask.sum() == 0:
        for md in range(min_positive_dims_total - 1, 0, -1):
            mask = make_mask(0.0, md)
            tried.append(("eps", 0.0, "min_dims", md, "M", int(mask.sum())))
            if mask.sum() > 0:
                break

    idx_hit = np.where(mask)[0].astype(int)
    M = int(idx_hit.size)

    if M == 0:
        prior = uniform_discrete_prior()
        debug = {
            "fallback": "no_sign_hit_uniform_discrete_prior",
            "M": 0,
            "tried": tried,
            "prior_weights": prior,
        }
        return prior, debug

    alpha = np.abs(active_phi[idx_hit]).sum(axis=1).astype(float)
    if (not np.isfinite(alpha).all()) or float(alpha.sum()) <= 1e-12:
        alpha = np.ones_like(alpha, dtype=float)
    alpha = alpha / (alpha.sum() + 1e-12)

    prior_weights = {}

    for var, values in variable_values.items():
        score = np.zeros(len(values), dtype=float)

        for local_k, sample_i in enumerate(idx_hit):
            val = active_values[sample_i][var]

            for j, candidate_val in enumerate(values):
                if var in ["Rwater", "ROil", "Ion"]:
                    same = np.isclose(float(val), float(candidate_val))
                else:
                    same = str(val) == str(candidate_val)

                if same:
                    score[j] += float(alpha[local_k])

        score = score + smooth
        score = score / (score.sum() + 1e-12)
        prior_weights[var] = {_val_key(v): float(wj) for v, wj in zip(values, score)}

    debug = {
        "M": M,
        "tried": tried,
        "prior_weights": prior_weights,
        "active_phi_hit_mean": active_phi[idx_hit].mean(axis=0).tolist(),
        "active_phi_hit_abs_mean": np.abs(active_phi[idx_hit]).mean(axis=0).tolist(),
    }

    return prior_weights, debug



GLOBAL_PIBO_REGISTRY = {}


def register_pibo_prior(acq_model, pack):
    # pack = (prior_weights, tau, transform_info)
    GLOBAL_PIBO_REGISTRY[id(acq_model)] = pack


if "kw_only" not in inspect.signature(_dc.dataclass).parameters:
    _orig_dataclass = _dc.dataclass

    def _dataclass_compat(*args, **kwargs):
        kwargs.pop("kw_only", None)
        return _orig_dataclass(*args, **kwargs)

    _dc.dataclass = _dataclass_compat

warnings.filterwarnings("ignore")

try:
    from typing import Annotated
except Exception:
    from typing_extensions import Annotated

logei = importlib.import_module("botorch.acquisition.logei")

if not hasattr(logei, "qLogEI_PiBO"):
    class qLogEI_PiBO(logei.qLogExpectedImprovement):
        r"""qLogEI(X) * pi(X) ** (beta / current_iter)."""

        def __init__(
            self,
            *args,
            pi_fn: Callable[[Tensor], Tensor],
            beta: float = 1.0,
            current_iter: int = 1,
            reduce: str = "max",
            **kwargs,
        ) -> None:
            assert reduce in ("max", "mean")
            super().__init__(*args, **kwargs)
            self.pi_fn = pi_fn
            self.beta = float(beta)
            self.current_iter = int(current_iter)
            self.reduce = reduce

        def _reduce_pi(self, pi: Tensor, base: Tensor) -> Tensor:
            if pi.ndim == base.ndim + 1:
                pi = pi.mean(dim=-1) if self.reduce == "mean" else pi.max(dim=-1).values
            return pi

        def forward(self, X: Tensor) -> Tensor:
            base = super().forward(X)

            if self.beta == 0:
                return base

            with torch.no_grad():
                pi_raw = self.pi_fn(X)
                pi = self._reduce_pi(pi_raw, base).to(base).clamp_min(1e-12)
                exp = self.beta / max(1, self.current_iter)

            modified = base * pi.pow(exp)

            return modified

    logei.qLogEI_PiBO = qLogEI_PiBO




af_mod = importlib.import_module("bofire.data_models.acquisition_functions.acquisition_function")
types_m = importlib.import_module("bofire.data_models.types")

if not hasattr(af_mod, "qLogEIWithPiBO"):
    class qLogEIWithPiBO(af_mod.SingleObjectiveAcquisitionFunction):
        type: Literal["qLogEIWithPiBO"] = "qLogEIWithPiBO"
        n_mc_samples: types_m.IntPowerOfTwo = 512
        beta: Annotated[float, Field(ge=0)] = 1.0
        reduce: Literal["max", "mean"] = "max"
        prior_mu: Optional[List[float]] = None
        prior_ell: Optional[List[float]] = None

    af_mod.qLogEIWithPiBO = qLogEIWithPiBO

api_mod = importlib.import_module("bofire.data_models.acquisition_functions.api")

if not hasattr(api_mod, "qLogEIWithPiBO"):
    api_mod.qLogEIWithPiBO = af_mod.qLogEIWithPiBO




def _normalize_discrete_value(v: float, values: List[float]) -> float:
    arr = np.asarray(values, dtype=float)
    lo = float(arr.min())
    hi = float(arr.max())
    if abs(hi - lo) < 1e-12:
        return 0.0
    return float((float(v) - lo) / (hi - lo))


def _pi_numeric_discrete_soft(
    Xflat: torch.Tensor,
    dim: int,
    values: List[float],
    weights_dict: Dict[str, float],
    tau: float,
) -> torch.Tensor:
    x = Xflat[:, dim]

    values_norm = [_normalize_discrete_value(float(v), values) for v in values]
    centers = Xflat.new_tensor(values_norm)

    w = [float(weights_dict[_val_key(v)]) for v in values]
    w_t = Xflat.new_tensor(w)
    w_t = w_t / (w_t.sum() + 1e-12)

    tau = float(max(1e-6, tau))
    dist2 = (x[:, None] - centers[None, :]) ** 2
    prob = torch.softmax(-dist2 / (2.0 * tau * tau), dim=-1)

    pi = (prob * w_t[None, :]).sum(dim=-1).clamp_min(1e-12)
    pi = pi * float(len(values))
    return pi


def _pi_categorical_soft_from_onehot(
    Xflat: torch.Tensor,
    start: int,
    categories: List[str],
    weights_dict: Dict[str, float],
    tau: float,
) -> torch.Tensor:
    size = len(categories)
    logits = Xflat[:, start:start + size]

    tau = float(max(1e-6, tau))
    prob = torch.softmax(logits / tau, dim=-1)

    w = [float(weights_dict[str(c)]) for c in categories]
    w_t = Xflat.new_tensor(w)
    w_t = w_t / (w_t.sum() + 1e-12)

    pi = (prob * w_t[None, :]).sum(dim=-1).clamp_min(1e-12)
    pi = pi * float(size)
    return pi



TRANSFORM_INFO = {
    "start_sw": 0,
    "dim_rwater": 3,
    "dim_roil": 4,
    "start_oil": 5,
    "start_so": 5 + len(CATS_OIL),
    "dim_ion": 5 + len(CATS_OIL) + len(CATS_SO),
}




from bofire.strategies.predictives.sobo import SoboStrategy
from bofire.utils.torch_tools import tkwargs
from botorch.sampling.normal import SobolQMCNormalSampler
from bofire.data_models.objectives.api import ConstrainedObjective, Objective

qLogEIWithPiBO_API = api_mod.qLogEIWithPiBO
qLogEI_PiBO = logei.qLogEI_PiBO
_old_get_acqfs = SoboStrategy._get_acqfs


def _best_f_maximize(self) -> torch.Tensor:
    assert self.experiments is not None
    try:
        target_feature = self.domain.outputs.get_by_objective(excludes=ConstrainedObjective)[0]
    except IndexError:
        target_feature = self.domain.outputs.get_by_objective(includes=Objective)[0]

    x_adapt = torch.from_numpy(
        self.domain.outputs.preprocess_experiments_one_valid_output(
            target_feature.key,
            self.experiments,
        )[target_feature.key].values
    ).to(**tkwargs)

    return x_adapt.max()


def _new_get_acqfs(self, n):
    if isinstance(self.acquisition_function, qLogEIWithPiBO_API):
        X_train, X_pending = self.get_acqf_input_tensors()
        objective_callable, constraint_callables, etas = self._get_objective_and_constraints()

        self._pibo_iter = getattr(self, "_pibo_iter", 0) + 1
        best_f = _best_f_maximize(self)
        afm = self.acquisition_function

        pack = GLOBAL_PIBO_REGISTRY.get(id(afm), None)
        def pi_fn(X: torch.Tensor) -> torch.Tensor:
            if pack is None:
                return X.new_ones(X.shape[:-1])

            prior_weights, tau, transform_info = pack

            xshape = X.shape
            Xflat = X.view(-1, xshape[-1])

            # Safety check for transformed dimension.
            expected_min_dim = int(transform_info["dim_ion"]) + 1
            if Xflat.shape[-1] < expected_min_dim:
                raise RuntimeError(
                    f"Transformed X dimension is {Xflat.shape[-1]}, but expected at least {expected_min_dim}. "
                    "Adjust TRANSFORM_INFO according to your BoFire transform layout."
                )

            pi_sw = _pi_categorical_soft_from_onehot(
                Xflat,
                start=transform_info["start_sw"],
                categories=CATS_SW,
                weights_dict=prior_weights["SW"],
                tau=tau,
            )

            pi_rwater = _pi_numeric_discrete_soft(
                Xflat,
                dim=transform_info["dim_rwater"],
                values=VALS_RWATER,
                weights_dict=prior_weights["Rwater"],
                tau=tau,
            )

            pi_roil = _pi_numeric_discrete_soft(
                Xflat,
                dim=transform_info["dim_roil"],
                values=VALS_ROIL,
                weights_dict=prior_weights["ROil"],
                tau=tau,
            )

            pi_oil = _pi_categorical_soft_from_onehot(
                Xflat,
                start=transform_info["start_oil"],
                categories=CATS_OIL,
                weights_dict=prior_weights["Oil"],
                tau=tau,
            )

            pi_so = _pi_categorical_soft_from_onehot(
                Xflat,
                start=transform_info["start_so"],
                categories=CATS_SO,
                weights_dict=prior_weights["SO"],
                tau=tau,
            )

            pi_ion = _pi_numeric_discrete_soft(
                Xflat,
                dim=transform_info["dim_ion"],
                values=VALS_ION,
                weights_dict=prior_weights["Ion"],
                tau=tau,
            )

            pi = (
                    pi_sw
                    * pi_rwater
                    * pi_roil
                    * pi_oil
                    * pi_so
                    * pi_ion
            ).clamp_min(1e-12)

            return pi.view(*xshape[:-1])

        sampler = SobolQMCNormalSampler(afm.n_mc_samples)

        acqf = qLogEI_PiBO(
            model=self.model,
            best_f=best_f,
            sampler=sampler,
            objective=objective_callable,
            X_pending=X_pending,
            constraints=constraint_callables,
            eta=torch.as_tensor(etas, **tkwargs),
            pi_fn=pi_fn,
            beta=float(afm.beta),
            current_iter=max(1, int(self._pibo_iter)),
            reduce=afm.reduce,
        )
        return [acqf]

    return _old_get_acqfs(self, n)


SoboStrategy._get_acqfs = _new_get_acqfs




from bofire.data_models.acquisition_functions.api import qLogEIWithPiBO


def build_problem_spec(history: pd.DataFrame) -> dict:
    impact_text = load_impact_summary()[:4000]

    top_history = []
    if not history.empty:
        tmp = history.sort_values(Y_COL, ascending=False).head(10)
        top_history = tmp[X_COLS_RAW + [Y_COL]].to_dict(orient="records")

    return {
        "objective": "maximize reward",
        "variables": {
            "SW": CATS_SW,
            "Rwater": VALS_RWATER,
            "ROil": VALS_ROIL,
            "Oil": CATS_OIL,
            "SO": CATS_SO,
            "Ion": VALS_ION,
        },
        "prior_knowledge": {
            "impact_summary": impact_text,
            "top_observed_history": top_history,
        },
        "notes": [
            "All variables are discrete.",
            "Avoid repeated combinations.",
            "Use impact_summary and high-reward historical experiments as hints if present.",
            "The generated candidates will be experimentally tested later; reward is unknown now.",
        ],
    }


def recommend_with_llm_warmstart(history_all: pd.DataFrame) -> pd.DataFrame:
    problem_spec = build_problem_spec(get_valid_history(history_all))

    try:
        init_X = generate_init_points_water_oil_with_agitco(
            n_points=int(BO_INIT_N),
            model=WARMSTART_MODEL,
            temperature=0.2,
            max_retries=2,
            problem_spec=problem_spec,
        )
        method = "agitco_llm"
        err = None
    except Exception as e:
        init_X = generate_init_samples(int(BO_INIT_N), seed=0)
        method = "random_fallback"
        err = str(e)
        print(f"[INIT] LLM warmstart failed -> random fallback. err={e}")

    init_X = clean_candidate_df(init_X)
    init_X = remove_already_tested(init_X, history_all)

    if init_X.empty:
        init_X = clean_candidate_df(generate_init_samples(int(BO_INIT_N), seed=1))
        init_X = remove_already_tested(init_X, history_all)

    init_X[Y_COL] = np.nan

    status = {
        "mode": "no_valid_reward_history_llm_warmstart",
        "method": method,
        "llm_model": WARMSTART_MODEL if method == "agitco_llm" else None,
        "error": err,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    (OUT_DIR / "init_status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    init_X.to_csv(OUT_DIR / "initial_recommendations.csv", index=False)
    return init_X


def recommend_with_pibo(history_valid: pd.DataFrame, history_all: pd.DataFrame) -> pd.DataFrame:
    prior_weights, prior_debug = build_prior_from_shap_sign_only_discrete(
        str(SHAP_JSON_PATH),
        problem="max",
        shap_sign_eps=SHAP_SIGN_EPS,
        min_positive_dims_total=MIN_POS_DIMS_TOTAL,
    )

    last_try = None
    if isinstance(prior_debug.get("tried"), list) and len(prior_debug.get("tried", [])) > 0:
        last_try = prior_debug["tried"][-1]

    print(f"[PRIOR] M_hit={prior_debug.get('M')} tried={last_try} fallback={prior_debug.get('fallback')}")
    dbg_block("[PRIOR] prior_weights", json.dumps(prior_weights, ensure_ascii=False, indent=2))

    acq = qLogEIWithPiBO(
        n_mc_samples=512,
        beta=float(BETA),
        reduce=str(REDUCE),
    )

    register_pibo_prior(acq, (prior_weights, float(SOFT_TAU), TRANSFORM_INFO))
    print("[CHECK] SHAP prior registered:", id(acq) in GLOBAL_PIBO_REGISTRY)
    print("[CHECK] Prior variables:", list(prior_weights.keys()))
    print("[CHECK] Prior Oil weights:", prior_weights.get("Oil"))
    print("[CHECK] Prior SW weights:", prior_weights.get("SW"))
    print("[CHECK] Prior Rwater weights:", prior_weights.get("Rwater"))
    sobo_dm = SoboStrategyDM(domain=domain, acquisition_function=acq)
    sobo = strategies.map(sobo_dm)

    # Tell only rows with known reward.
    sobo.tell(history_valid)

    raw_new = sobo.ask(max(int(N_RECOMMEND), 1))
    raw_new = clean_candidate_df(raw_new)
    new_exp = remove_already_tested(raw_new, history_all)

    if new_exp.empty:
        print("[WARN] BO recommendation duplicated existing history. Falling back to random untested candidate.")
        random_pool = clean_candidate_df(generate_init_samples(1000, seed=42))
        random_pool = random_pool.drop_duplicates(subset=X_COLS_RAW).reset_index(drop=True)
        new_exp = remove_already_tested(random_pool, history_all).head(N_RECOMMEND)

    new_exp = new_exp.head(N_RECOMMEND).copy()
    new_exp[Y_COL] = np.nan

    meta = {
        "mode": "pibo_bo_recommendation",
        "history_csv": str(EXPERIMENTS_CSV),
        "shap_json": str(SHAP_JSON_PATH),
        "n_valid_history": int(len(history_valid)),
        "n_all_history": int(len(history_all)),
        "prior_debug": prior_debug,
        "prior_weights": prior_weights,
        "beta": float(BETA),
        "reduce": str(REDUCE),
        "tau": float(SOFT_TAU),
        "transform_info": TRANSFORM_INFO,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    (OUT_DIR / "meta_recommendation.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    new_exp.to_csv(OUT_DIR / "next_recommendation.csv", index=False)

    return new_exp


def main():
    print("[START] Water-oil discrete PiBO real-experiment recommendation")
    print(f"[DATA] Reading historical experiments from: {EXPERIMENTS_CSV.resolve()}")

    history_all = load_experiments_csv(EXPERIMENTS_CSV)
    history_valid = get_valid_history(history_all)

    print(f"[DATA] all rows={len(history_all)}, valid reward rows={len(history_valid)}")

    if len(history_valid) == 0:
        print("[MODE] No valid reward history found. Using LLM warmstart recommendations.")
        rec = recommend_with_llm_warmstart(history_all)
    else:
        print("[MODE] Valid reward history found. Using PiBO recommendation.")
        rec = recommend_with_pibo(history_valid, history_all)

    print("\n[NEXT EXPERIMENT RECOMMENDATION]")
    print(rec.to_string(index=False))

    if APPEND_RECOMMENDATION_TO_HISTORY:
        combined = pd.concat([history_all, rec], ignore_index=True)
        combined.to_csv(OUT_DIR / "BO_with_new_recommendation.csv", index=False)
        print(f"\n[SAVE] Combined CSV saved to: {(OUT_DIR / 'BO_with_new_recommendation.csv').resolve()}")

    print(f"[SAVE] Outputs saved under: {OUT_DIR.resolve()}")
    print("[DONE]")


if __name__ == "__main__":
    main()
