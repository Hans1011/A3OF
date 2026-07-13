
import importlib, torch, warnings
from torch import Tensor
from typing import Callable, Optional, List, Literal, Dict, Tuple
from pydantic import Field
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime
# ===================== [NEW] impact_summary per-run loader =====================
IMPACT_PARENT_DIR = Path(".")  # TODO: set your path
def dbg_block(title: str, text: str, chunk: int = 1200):
    """, clip ,."""
    if not DEBUG:
        return
    text = "" if text is None else str(text)
    print(f"\n========== {title} (len={len(text)}) ==========")
    for i in range(0, len(text), chunk):
        print(text[i:i+chunk])
    print("========== END ==========\n")

def load_impact_summary_for_run(run_idx: int) -> str:
    """
    run_idx: 0..24 -> run_000..run_024
     impact_summary.txt ,.
    """
    impact_path = IMPACT_PARENT_DIR / f"run_{run_idx:03d}" / "impact_summary.txt"
    if not impact_path.exists():
        return ""
    try:
        txt = impact_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        txt = impact_path.read_text(errors="ignore")
    return (txt or "").strip()

import os
import re
from pydantic import BaseModel
from openai import OpenAI

API_BASE = "https://api.agicto.cn/v1"
DEBUG = True

def dbg(msg: str):
    if DEBUG:
        print(msg)

def clip(s: str, n: int = 300) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ...[truncated]"

#  warmstart ()
WARMSTART_MODEL = "deepseek-chat"

class _InitDesignOutReaction(BaseModel):
    points: List[Dict[str, object]]
    notes: Optional[str] = None

def extract_json(s: str) -> str:
    s = (s or "").strip()
    # ```json ... ``` / ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    # { }
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]
    return s

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        xv = float(x)
    except Exception:
        xv = float("nan")
    if not np.isfinite(xv):

        xv = 0.5 * (lo + hi)
    return float(min(max(xv, lo), hi))

def _norm_cont(T: float, V: float) -> np.ndarray:
    # [0,1]^2:T(0~60), V(20~90)
    t = (float(T) - 0.0) / (60.0 - 0.0 + 1e-12)
    v = (float(V) - 20.0) / (90.0 - 20.0 + 1e-12)
    return np.clip(np.array([t, v], dtype=float), 0.0, 1.0)

def generate_init_points_reaction_with_agitco(
    *,
    n_points: int,
    model: str = WARMSTART_MODEL,
    temperature: float = 0.7,
    max_retries: int = 2,
    min_l2_dist_cont: float = 0.10,   # L2

    problem_spec: Optional[dict] = None,

    categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
     DataFrame:columns = ["Temperature","Solvent Volume","Solvent Type"],  n_points
    """
    categories = categories or ["MeOH", "THF", "Dioxane"]

    api_key = "YOUR_API_KEY_HERE"
    if not api_key:
        raise RuntimeError("Missing API key. Set env var AGITCO_API_KEY or OPENAI_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=API_BASE)

    sys_msg = (
        "You are a top-tier Bayesian optimization assistant. "
        "Your task is to propose warmstart initial experiment conditions for a reaction optimization problem. "
        "We want HIGH objective value (maximize Yield), while keeping diversity/space-filling. "
        "Return ONLY a valid JSON object, no markdown, no extra text. "
        "The output MUST start with '{' and end with '}'."
        " You should balance exploration and exploitation. "
        "Do NOT only pick the highest predicted yield, but also explore diverse regions."
    )


    accepted_rows: List[Dict[str, object]] = []

    def _is_far_enough(new_T: float, new_V: float) -> bool:
        if len(accepted_rows) == 0:
            return True
        p = _norm_cont(new_T, new_V)
        for r in accepted_rows:
            q = _norm_cont(float(r["Temperature"]), float(r["Solvent Volume"]))
            if float(np.linalg.norm(p - q)) < float(min_l2_dist_cont):
                return False
        return True

    def _dedupe_key(r: Dict[str, object]) -> Tuple[int, int, str]:
        # : rounding
        T = float(r["Temperature"])
        V = float(r["Solvent Volume"])
        S = str(r["Solvent Type"])
        return (int(round(T * 10)), int(round(V * 10)), S)

    last_err = None

    for attempt in range(max_retries + 1):
        try:
            for round_idx in range(12):  # 12
                need = int(n_points - len(accepted_rows))
                if need <= 0:
                    break

                user_payload = {
                    "task": "warmstart_initial_design_reaction",
                    "n_points": int(need),
                    "variables": {
                        "Temperature": {"type": "continuous", "bounds": [0.0, 60.0], "unit": "C"},
                        "Solvent Volume": {"type": "continuous", "bounds": [20.0, 90.0]},
                        "Solvent Type": {"type": "categorical", "categories": categories},
                    },
                    "objective": {"type": "maximize", "name": "Yield"},
                    "already_selected_points": accepted_rows,
                    "distance_rules": {
                        "min_l2_dist_cont": float(min_l2_dist_cont),
                        "notes": "New points must be at least min_l2_dist_cont away (in normalized continuous space) from any already_selected_points."
                    },
                    "problem_spec": problem_spec or {},
                    "output_schema": {
                        "points": (
                            f"List of exactly {need} points. "
                            "Each point must be an object with keys: "
                            "`Temperature` (float), `Solvent Volume` (float), `Solvent Type` (string)."
                        ),
                        "notes": "Optional string."
                    },
                    "hard_rules": [
                        "Return ONLY valid JSON.",
                        f"Return exactly {need} points in `points` (no more, no less).",
                        "Temperature must be within [0,60].",
                        "Solvent Volume must be within [20,90].",
                        f"Solvent Type must be one of: {categories}.",
                        "Do NOT repeat any already_selected_points (including near-duplicates).",
                        "New points must satisfy the distance_rules.",
                    ],
                    "soft_rules": [
                        "Bias some points toward high yield regions if you can infer them, but keep diversity/coverage.",
                        "Ensure category coverage across the set if possible.",
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
                    top_p=0.9
                )

                if resp is None or not getattr(resp, "choices", None):
                    raise RuntimeError("Agitco returned empty response (no choices).")

                msg = getattr(resp.choices[0], "message", None)
                content = getattr(msg, "content", None)
                if not content or not str(content).strip():
                    raise RuntimeError("Agitco message.content empty.")
                raw = str(content).strip()
                dbg_block("[AGITCO] raw_response", raw)

                data = json.loads(extract_json(raw))
                out = _InitDesignOutReaction.model_validate(data)
                pts = out.points
                dbg_block("[AGITCO] parsed_json", json.dumps(data, ensure_ascii=False, indent=2))
                dbg(f"[AGITCO] parsed points count = {len(pts)}")

                # ///
                seen = set(_dedupe_key(r) for r in accepted_rows)

                for p in pts:
                    if len(accepted_rows) >= n_points:
                        break

                    T = _clamp(p.get("Temperature", None), 0.0, 60.0)
                    V = _clamp(p.get("Solvent Volume", None), 20.0, 90.0)
                    S = str(p.get("Solvent Type", "")).strip()

                    if S not in categories:
                        # :
                        continue

                    candidate = {"Temperature": T, "Solvent Volume": V, "Solvent Type": S}
                    k = _dedupe_key(candidate)
                    if k in seen:
                        continue
                    if not _is_far_enough(T, V):
                        continue

                    accepted_rows.append(candidate)
                    seen.add(k)
                dbg_block("[AGITCO] accepted_rows_so_far", json.dumps(accepted_rows, ensure_ascii=False, indent=2))

            if len(accepted_rows) != n_points:
                raise ValueError(f"LLM failed to provide {n_points} valid points. got={len(accepted_rows)}")

            df = pd.DataFrame(accepted_rows, columns=["Temperature", "Solvent Volume", "Solvent Type"])
            if len(df) != n_points:
                raise ValueError("Internal error: dataframe length mismatch.")
            dbg_block("[AGITCO] final_init_df", df.to_csv(index=False))
            return df

        except Exception as e:
            last_err = e
            dbg(f"[AGITCO] attempt failed: {e}")
            accepted_rows = []

    raise RuntimeError(f"Agitco warmstart failed after retries. Last error: {last_err}")

# ===================== [NEW END] Agitco LLM warmstart helper =====================


T0 = 25
T1 = 100
e0 = np.exp((T1 + 0) / T0)
e60 = np.exp((T1 + 60) / T0)
de = e60 - e0

boiling_points = {"MeOH": 64.7, "THF": 66.0, "Dioxane": 101.0}
density = {"MeOH": 0.792, "THF": 0.886, "Dioxane": 1.03}
solvent_descriptors = pd.DataFrame({"boiling_points": boiling_points, "density": density})

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


import bofire.strategies.api as strategies
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDM
from bofire.data_models.features.api import ContinuousOutput, ContinuousInput, CategoricalInput
from bofire.data_models.objectives.api import MaximizeObjective
from bofire.data_models.domain.api import Domain, Inputs, Outputs

temperature_feature = ContinuousInput(key="Temperature", bounds=[0.0, 60.0], unit="°C")
solvent_amount_feature = ContinuousInput(key="Solvent Volume", bounds=[20.0, 90.0])
solvent_type_feature = CategoricalInput(key="Solvent Type", categories=["MeOH", "THF", "Dioxane"])

inputs = Inputs(features=[temperature_feature, solvent_type_feature, solvent_amount_feature])

objective = MaximizeObjective(w=1.0)
yield_feature = ContinuousOutput(key="Yield", objective=objective)
outputs = Outputs(features=[yield_feature])

domain = Domain(inputs=inputs, outputs=outputs)

X_COLS_RAW = ["Temperature", "Solvent Volume", "Solvent Type"]
Y_COL = "Yield"
CATS_SOLVENT = list(solvent_type_feature.categories)

def generate_init_samples(n: int, seed: Optional[int] = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    T = rng.uniform(temperature_feature.lower_bound, temperature_feature.upper_bound, size=n)
    V = rng.uniform(solvent_amount_feature.lower_bound, solvent_amount_feature.upper_bound, size=n)
    S = rng.choice(CATS_SOLVENT, size=n)
    return pd.DataFrame({"Temperature": T, "Solvent Volume": V, "Solvent Type": S})

# ============================================================
# Benchmark
# ============================================================
def create_experiments(domain, nsamples=100, A=25, B=90, candidates=None):
    Tf = domain.inputs.get_by_key("Temperature")
    Vf = domain.inputs.get_by_key("Solvent Volume")
    typef = domain.inputs.get_by_key("Solvent Type")
    yf = domain.outputs.get_by_key("Yield")

    if candidates is None:
        T = np.random.uniform(low=Tf.lower_bound, high=Tf.upper_bound, size=(nsamples,))
        V = np.random.uniform(low=Vf.lower_bound, high=Vf.upper_bound, size=(nsamples,))
        solvent_types = [typef.categories[np.random.randint(0, len(typef.categories))] for _ in range(nsamples)]
    else:
        nsamples = len(candidates)
        T = candidates["Temperature"].values.astype(float)
        V = candidates["Solvent Volume"].values.astype(float)
        solvent_types = candidates["Solvent Type"].values.astype(str)

    # core()
    Tfact = calc_Tfact(T)
    rhofact = calc_rhofact(solvent_types, Tfact)
    Vfact = calc_volume_fact(V)

    y_core = A * Tfact + B * rhofact
    y_core = 0.5 * y_core + 0.5 * y_core * Vfact

    samples = pd.DataFrame(
        {
            Tf.key: T,
            Vf.key: V,
            typef.key: solvent_types,
            yf.key: y_core,
            "valid_" + yf.key: np.ones(nsamples),
        }
    )
    samples.index = pd.RangeIndex(nsamples)
    return samples

def evaluate_experiments(domain, candidates):
    return create_experiments(domain, candidates=candidates)


def one_hot_df_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    oh_s = pd.get_dummies(df_raw["Solvent Type"].astype(str), prefix="SolventType")
    out = pd.concat(
        [
            df_raw[["Temperature", "Solvent Volume"]].astype(float).reset_index(drop=True),
            oh_s.reset_index(drop=True),
        ],
        axis=1,
    )

    for c in CATS_SOLVENT:
        col = f"SolventType_{c}"
        if col not in out.columns:
            out[col] = 0.0

    cols = ["Temperature", "Solvent Volume"] + [f"SolventType_{c}" for c in CATS_SOLVENT]
    return out[cols].astype(float)

def load_shap_payload_json(path: str) -> Dict:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    oh_names = payload["onehot_detail"]["feature_names_onehot"]  
    phi_dN = np.asarray(payload["onehot_detail"]["phi_mean_onehot"]["data"], dtype=float)  

    X_raw = payload["history"]["X"]["data"]  
    y = np.asarray(payload["history"]["y"]["data"], dtype=float)  

    df_raw = pd.DataFrame(X_raw, columns=["Temperature", "Solvent Volume", "Solvent Type"])
    X_oh = one_hot_df_raw(df_raw).values  

    assert X_oh.shape[1] == phi_dN.shape[0], f"dim mismatch X_oh={X_oh.shape}, phi={phi_dN.shape}"
    return {
        "feature_names_onehot": oh_names,  
        "history_raw_df": df_raw,          
        "history_onehot": X_oh,            
        "y": y,                            
        "phi_dN": phi_dN,                  
    }

def _index_map(names: List[str]) -> Dict[str, int]:
    return {n: i for i, n in enumerate(names)}

def _normalize_cont_to_01(X_cont_raw: np.ndarray) -> np.ndarray:
    # 2 bounds [0,1]
    lb = np.array([0.0, 20.0], dtype=float)
    ub = np.array([60.0, 90.0], dtype=float)
    w = ub - lb
    Xn = (X_cont_raw - lb[None, :]) / (w[None, :] + 1e-12)
    return np.clip(Xn, 0.0, 1.0)

def build_prior_from_shap_sign_only_gmm_bic(
    shap_json_path: str,
    *,
    problem: Literal["max", "min"] = "max",
    shap_sign_eps: float = 1e-6,
    min_positive_dims_cont: int = 2,   
    kmax: int = 6,
    reg_covar: float = 1e-6,
    ell_min: float = 0.03,
    ell_max: float = 0.75,
    smooth: float = 1e-12,
) -> Tuple[List[List[float]], List[List[float]], List[float], List[float], Dict]:
    """
    :
      mus_cont_norm   : [K,2]   ([0,1])
      ells_cont_norm  : [K,2]
      w_cont          : [K]
      w_solvent       : [3]     (MeOH/THF/Dioxane)
      debug           : dict
    """
    pack = load_shap_payload_json(shap_json_path)
    names = pack["feature_names_onehot"]
    X_oh = np.asarray(pack["history_onehot"], dtype=float)  
    df_raw = pack["history_raw_df"]
    phi_dN = np.asarray(pack["phi_dN"], dtype=float)        
    N, d = X_oh.shape
    assert phi_dN.shape == (d, N)

    idx = _index_map(names)

    cont_cols = ["Temperature", "Solvent Volume"]
    cont_idx = [idx[c] for c in cont_cols]

    solv_cols = [f"SolventType_{c}" for c in CATS_SOLVENT]
    solv_idx = [idx[c] for c in solv_cols]

    X_cont_raw = X_oh[:, cont_idx]              
    X_cont = _normalize_cont_to_01(X_cont_raw)  

    solv_raw = df_raw["Solvent Type"].astype(str).values
    solv_id = np.array([CATS_SOLVENT.index(s) if s in CATS_SOLVENT else 0 for s in solv_raw], dtype=int)

    phi_cont_N2 = phi_dN[cont_idx, :].T  

    phi_solv_self = np.zeros(N, dtype=float)
    for i in range(N):
        phi_solv_self[i] = float(phi_dN[solv_idx[solv_id[i]], i])

    def make_mask(eps: float, min_dims: int) -> np.ndarray:
        if problem == "max":
            cont_ok = (phi_cont_N2 > eps).sum(axis=1) >= int(min_dims)
            solv_ok = (phi_solv_self > eps)
        else:
            cont_ok = (phi_cont_N2 < -eps).sum(axis=1) >= int(min_dims)
            solv_ok = (phi_solv_self < -eps)
        return cont_ok & solv_ok

    tried = []
    mask = make_mask(shap_sign_eps, min_positive_dims_cont)
    tried.append(("eps", shap_sign_eps, "min_dims", min_positive_dims_cont, "M", int(mask.sum())))

    if mask.sum() == 0:
        mask = make_mask(0.0, min_positive_dims_cont)
        tried.append(("eps", 0.0, "min_dims", min_positive_dims_cont, "M", int(mask.sum())))

    if mask.sum() == 0:
        for md in range(min_positive_dims_cont, 0, -1):
            mask = make_mask(0.0, md)
            tried.append(("eps", 0.0, "min_dims", md, "M", int(mask.sum())))
            if mask.sum() > 0:
                break

    idx_hit = np.where(mask)[0].astype(int)
    M = int(idx_hit.size)

    if M == 0:
        mu0 = X_cont.mean(axis=0)
        std0 = np.maximum(X_cont.std(axis=0), 0.10)
        mus = [mu0.tolist()]
        ells = [np.clip(std0, ell_min, ell_max).tolist()]
        w_cont = [1.0]
        w_sol = (np.ones(len(CATS_SOLVENT)) / len(CATS_SOLVENT)).tolist()
        debug = {"fallback": "no_sign_hit_uniform", "tried": tried, "M": 0, "K": 1}
        return mus, ells, w_cont, w_sol, debug

    Xg = X_cont[idx_hit]  


    alpha = (np.abs(phi_cont_N2[idx_hit]).sum(axis=1) + np.abs(phi_solv_self[idx_hit])).astype(float)
    if (not np.isfinite(alpha).all()) or float(alpha.sum()) <= 1e-12:
        alpha = np.ones_like(alpha, dtype=float)
    alpha = alpha / (alpha.sum() + 1e-12)

    sol_score = np.zeros(len(CATS_SOLVENT), dtype=float)
    for k, i in enumerate(idx_hit):
        sol_score[solv_id[i]] += float(alpha[k])
    sol_score = sol_score + smooth
    w_sol = (sol_score / (sol_score.sum() + 1e-12)).tolist()


    if M < 2:
        mu0 = Xg[0]
        ell0 = np.maximum(X_cont.std(axis=0), 0.08)
        ell0 = np.clip(ell0, ell_min, ell_max)

        mus_cont = [mu0.tolist()]
        ells_cont = [ell0.tolist()]
        w_cont = [1.0]

        debug = {
            "M": M,
            "K": 1,
            "fallback": "only_1_sign_hit_no_gmm",
            "tried": tried,
            "w_solvent": [float(x) for x in w_sol],
            "w_cont": [1.0],
        }
        return mus_cont, ells_cont, w_cont, w_sol, debug

    from sklearn.mixture import GaussianMixture

    Kmax = int(min(max(1, kmax), M))
    best_bic = np.inf
    best_gmm = None
    for k in range(1, Kmax + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="diag",
            reg_covar=reg_covar,
            random_state=0,
            n_init=3,
            max_iter=500,
        )
        gmm.fit(Xg)
        bic = gmm.bic(Xg)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm

    resp = best_gmm.predict_proba(Xg)  # [M,K]
    w_eff = alpha[:, None] * resp
    w_k = w_eff.sum(axis=0)
    keep = w_k > 1e-12
    if not np.any(keep):
        keep = np.ones_like(w_k, dtype=bool)
    w_eff = w_eff[:, keep]
    w_k = w_k[keep]
    K = int(w_k.shape[0])
    w_k = (w_k / (w_k.sum() + 1e-12)).astype(float)

    mus_cont, ells_cont = [], []
    for kk in range(K):
        wk_vec = w_eff[:, kk]
        denom = float(wk_vec.sum() + 1e-12)
        mu = (wk_vec[:, None] * Xg).sum(axis=0) / denom
        diff = Xg - mu[None, :]
        var = (wk_vec[:, None] * (diff ** 2)).sum(axis=0) / denom
        ell = np.sqrt(np.maximum(var, 1e-18))
        ell = np.clip(ell, ell_min, ell_max)
        mus_cont.append(mu.tolist())
        ells_cont.append(ell.tolist())

    w_cont = w_k.tolist()

    debug = {
        "M": M,
        "K": K,
        "best_bic": float(best_bic),
        "tried": tried,
        "w_solvent": [float(x) for x in w_sol],
        "w_cont": [float(x) for x in w_cont],
    }
    return mus_cont, ells_cont, w_cont, w_sol, debug

GLOBAL_PIBO_REGISTRY = {}
def register_pibo_prior(acq_model, pack):
    # pack = (mus_cont, ells_cont, w_cont, w_solvent, tau)
    GLOBAL_PIBO_REGISTRY[id(acq_model)] = pack

import inspect, dataclasses as _dc
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
                pi = self.pi_fn(X)
                pi = self._reduce_pi(pi, base).to(base).clamp_min(1e-12)
                exp = self.beta / max(1, self.current_iter)
            return base * pi.pow(exp)

    logei.qLogEI_PiBO = qLogEI_PiBO


af_mod  = importlib.import_module("bofire.data_models.acquisition_functions.acquisition_function")
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
            target_feature.key, self.experiments
        )[target_feature.key].values
    ).to(**tkwargs)
    return x_adapt.max()

def _pi_gmm_cont(Xflat: torch.Tensor, mus, ells, ws) -> torch.Tensor:
    # Xflat: [M,d], 2
    Xc = Xflat[:, :2]
    ws_t = Xflat.new_tensor(ws)
    ws_t = ws_t / (ws_t.sum() + 1e-12)

    comps = []
    for mu_k, ell_k in zip(mus, ells):
        mu = Xflat.new_tensor(mu_k)   # [2]
        ell = Xflat.new_tensor(ell_k) # [2]
        Z = (Xc - mu) / (ell + 1e-12)
        comps.append(torch.exp(-0.5 * (Z * Z).sum(dim=-1)))
    Pis = torch.stack(comps, dim=-1)  # [M,K]
    mix = (Pis * ws_t).sum(dim=-1).clamp_min(1e-12)
    mix = mix * float(max(1, len(mus)))  # ≈1
    return mix

def _pi_cat_soft_from_onehot(Xflat: torch.Tensor, start: int, size: int, w: List[float], tau: float) -> torch.Tensor:
    logits = Xflat[:, start:start+size]
    tau = float(max(1e-6, tau))
    p = torch.softmax(logits / tau, dim=-1)  # [M,size]
    w_t = Xflat.new_tensor(w)
    pi = (p * w_t[None, :]).sum(dim=-1).clamp_min(1e-12)
    pi = pi * float(size)  # ≈1
    return pi

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

            mus_cont, ells_cont, w_cont, w_sol, tau = pack

            xshape = X.shape
            Xflat = X.view(-1, xshape[-1])

            pi_cont = _pi_gmm_cont(Xflat, mus_cont, ells_cont, w_cont)
            # solvent onehot: start=2 size=3
            pi_sol = _pi_cat_soft_from_onehot(Xflat, start=2, size=3, w=w_sol, tau=tau)

            pi = (pi_cont * pi_sol).clamp_min(1e-12)
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

SHAP_PARENT_DIR = Path(r"1.4-chemical-BO1")  # ✅
OUT_PARENT_DIR  = Path(r"1.18-BO1-chemical3")
OUT_PARENT_DIR.mkdir(parents=True, exist_ok=True)

NUM_RUNS = 25
BO_INIT_N = 3
BO_MAX_ITER = 50

TARGET_BEST = 68.604575
TARGET_TOL  = 0.2
STOP_WHEN_REACH_TARGET = True

SHAP_SIGN_EPS = 1e-6
MIN_POS_DIMS_CONT = 2
GMM_KMAX = 6
SOFT_TAU = 0.05
BETA = 2.0
REDUCE = "mean"

def run_one_pibo_reaction(run_idx: int, shap_json_path: Path, out_dir: Path):

    status = {
        "run": int(run_idx),
        "init_method": None,
        "llm_model": None,
        "error": None,
    }

    # LLM ( PiBO)
    # [NEW] run impact_summary.txt ( warmstart )
    impact_text = load_impact_summary_for_run(run_idx)
    impact_text = impact_text[:4000]  # prompt ,/

    problem_spec = {
        "objective": "maximize Yield",
        "variables": {
            "Temperature": "0~60",
            "Solvent Volume": "20~90",
            "Solvent Type": CATS_SOLVENT,
        },
        "prior_knowledge": {
            "impact_summary": impact_text
        },

        "notes": [
            f"Generate {BO_INIT_N} diverse initial points.",
            "Avoid giving nearly identical continuous settings.",
            "Try to include some category variety if possible.",
            "Use impact_summary as additional hints if present."
        ]
    }

    # :
    print(f"[RUN {run_idx:03d}] impact_found={bool(impact_text)} chars={len(impact_text)}")

    try:
        init_X = generate_init_points_reaction_with_agitco(
            n_points=int(BO_INIT_N),
            model=WARMSTART_MODEL,
            temperature=0.2,
            max_retries=2,
            min_l2_dist_cont=0.15,
            problem_spec=problem_spec,
            categories=CATS_SOLVENT,
        )
        status["init_method"] = "agitco_llm"
        status["llm_model"] = WARMSTART_MODEL
        print(f"[RUN {run_idx:03d}] LLM warmstart OK. model={WARMSTART_MODEL}, n={len(init_X)}")
    except Exception as e:
        init_X = generate_init_samples(int(BO_INIT_N), seed=run_idx)
        status["init_method"] = "random_fallback"
        status["error"] = str(e)
        print(f"[RUN {run_idx:03d}] LLM warmstart FAILED -> fallback random. err={e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "init_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    init_X.to_csv(out_dir / "init_points.csv", index=False)


    init_xy = evaluate_experiments(domain, init_X)
    all_data = init_xy.copy()

    # build prior
    mus_cont, ells_cont, w_cont, w_sol, dbg = build_prior_from_shap_sign_only_gmm_bic(
        str(shap_json_path),
        problem="max",
        shap_sign_eps=SHAP_SIGN_EPS,
        min_positive_dims_cont=MIN_POS_DIMS_CONT,
        kmax=GMM_KMAX,
    )
    last_try = dbg.get("tried")[-1] if isinstance(dbg.get("tried"), list) and len(dbg["tried"]) > 0 else dbg.get("tried")
    print(f"[RUN {run_idx:03d}] peaks(K)={len(mus_cont)}  M_hit={dbg.get('M')}  tried={last_try}")

    # acq + register
    acq = qLogEIWithPiBO(n_mc_samples=512, beta=float(BETA), reduce=str(REDUCE))
    register_pibo_prior(acq, (mus_cont, ells_cont, w_cont, w_sol, float(SOFT_TAU)))

    sobo_dm = SoboStrategyDM(domain=domain, acquisition_function=acq)
    sobo = strategies.map(sobo_dm)
    sobo.tell(init_xy)

    trace_rows = []
    best_so_far = float(all_data[Y_COL].max())

    for _, row in init_xy.iterrows():
        trace_rows.append({
            "iter": 0,
            "Temperature": float(row["Temperature"]),
            "Solvent Volume": float(row["Solvent Volume"]),
            "Solvent Type": str(row["Solvent Type"]),
            "Yield": float(row["Yield"]),
            "best_so_far": best_so_far,
            "is_initial": True,
        })

    stop_iter = None
    stop_reason = None
    if STOP_WHEN_REACH_TARGET and (best_so_far >= (TARGET_BEST - TARGET_TOL)):
        stop_iter = 0
        stop_reason = "hit_target_in_init"

    if stop_iter is None:
        for it in range(BO_MAX_ITER):
            new_exp = sobo.ask(1)
            new_xy = evaluate_experiments(domain, new_exp)
            sobo.tell(new_xy)

            all_data = pd.concat([all_data, new_xy], ignore_index=True)

            fx = float(new_xy[Y_COL].iloc[0])
            best_so_far = max(best_so_far, fx)

            xrow = new_xy.iloc[0]
            trace_rows.append({
                "iter": it + 1,
                "Temperature": float(xrow["Temperature"]),
                "Solvent Volume": float(xrow["Solvent Volume"]),
                "Solvent Type": str(xrow["Solvent Type"]),
                "Yield": fx,
                "best_so_far": best_so_far,
                "is_initial": False,
            })

            if STOP_WHEN_REACH_TARGET and (best_so_far >= (TARGET_BEST - TARGET_TOL)):
                stop_iter = it + 1
                stop_reason = "hit_target"
                break

    # save
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trace_rows).to_csv(out_dir / "bo_trace.csv", index=False)
    all_data.to_csv(out_dir / "bo_history.csv", index=False)

    meta = {
        "run_idx": int(run_idx),
        "shap_json": str(shap_json_path),
        "stopped_early": stop_iter is not None,
        "stop_iter": stop_iter,
        "stop_reason": stop_reason,
        "final_best": float(best_so_far),
        "final_n_points": int(len(all_data)),
        "prior_debug": dbg,
        "cat_w_solvent": {c: float(w) for c, w in zip(CATS_SOLVENT, w_sol)},
        "n_prior_peaks": int(len(mus_cont)),
        "beta": float(BETA),
        "reduce": str(REDUCE),
        "tau": float(SOFT_TAU),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "init_status": status,
    }
    (out_dir / "meta_run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[RUN {run_idx:03d}] [SAVE] -> {out_dir.resolve()}  best={best_so_far:.6f}")

# :run_000..run_024
for i in range(NUM_RUNS):
    shap_json = SHAP_PARENT_DIR / f"run_{i:03d}" / "shap_results.json"
    if not shap_json.exists():
        raise FileNotFoundError(f"Missing: {shap_json}")
    out_dir = OUT_PARENT_DIR / f"run_{i:03d}"
    run_one_pibo_reaction(i, shap_json, out_dir)

print(f"[DONE] All {NUM_RUNS} runs saved under: {OUT_PARENT_DIR.resolve()}")
