
import importlib, torch, warnings
from torch import Tensor
from typing import Callable, Optional, List, Literal
from pydantic import Field
from pathlib import Path
import json
import numpy as np
import pandas as pd


import os
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

GRAPHRAG_CHAT_MODEL = "gpt-5-nano"
JSON_MODE = False


MULTIHOP_MODEL = "deepseek-chat"
FUSION_MODEL = MULTIHOP_MODEL


WARMSTART_MODEL = MULTIHOP_MODEL

class _InitDesignOut(BaseModel):
    points: List[List[float]]
    notes: Optional[str] = None

def _clamp_and_dedupe_points(points: np.ndarray, bounds: List[List[float]]) -> np.ndarray:
    b = np.asarray(bounds, dtype=float)  # [d,2]
    pts = np.asarray(points, dtype=float)
    pts = np.clip(pts, b[:, 0], b[:, 1])
    pts_r = np.round(pts, 10)
    _, idx = np.unique(pts_r, axis=0, return_index=True)
    pts = pts[np.sort(idx)]
    return pts


IMPACT_PARENT_DIR = Path(".")  # TODO: set your path  # <-

import re

def extract_json(s: str) -> str:
    s = (s or "").strip()

    # 1) ```json ... ``` / ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # 2) , { }
    l = s.find("{")
    r = s.rfind("}")
    if l != -1 and r != -1 and r > l:
        s = s[l:r+1]

    return s

def load_impact_summary_for_run(run_idx: int) -> str:
    """
    run_idx: 0..99 -> run_000..run_099
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

def generate_init_points_with_agitco(
    *,
    n_points: int,
    feature_names: List[str],
    bounds: List[List[float]],
    model: str = WARMSTART_MODEL,
    temperature: float = 0.2,
    max_retries: int = 2,
    problem_spec: Optional[dict] = None,
    # [NEW] “”( L2 )
    min_l2_dist: float = 0.05,
) -> np.ndarray:
    """
     Agitco  warmstart .
     exactly n_points( LHS ):
    -  LLM /, LLM .
    """
    api_key = "YOUR_API_KEY_HERE"
    if not api_key:
        raise RuntimeError("Missing API key. Set env var OPENAI_API_KEY or AGITCO_API_KEY.")

    client = OpenAI(api_key=api_key, base_url=API_BASE)

    d = len(feature_names)
    b = np.asarray(bounds, dtype=float)

    sys_msg = (
        "You are a top-tier Bayesian optimization assistant working on black-box test functions. "
        "Your goal is to propose warmstart initial design points that are likely to achieve small objective values, "
        "while still maintaining diversity and space-filling coverage. "
        "You must use any prior knowledge provided (e.g., impact_summary) to bias sampling toward promising regions. "
        "Return ONLY a valid JSON object, no markdown, no extra text. "
        "The output MUST start with '{' and end with '}'."
    )

    def _min_pairwise_l2(pts: np.ndarray) -> float:
        if pts.shape[0] < 2:
            return float("inf")
        # O(n^2) ,n<=10
        md = float("inf")
        for i in range(pts.shape[0]):
            for j in range(i + 1, pts.shape[0]):
                md = min(md, float(np.linalg.norm(pts[i] - pts[j])))
        return md

    accepted = np.empty((0, d), dtype=float)

    last_err = None
    # : LLM
    for attempt in range(max_retries + 1):
        try:
            # :, n_points
            for round_idx in range(10):  # 10 ,
                need = int(n_points - accepted.shape[0])
                if need <= 0:
                    break

                # :, need
                user_payload = {
                    "task": "warmstart_initial_design",
                    "n_points": int(need),
                    "input_order": feature_names,
                    "bounds": {feature_names[i]: [float(b[i, 0]), float(b[i, 1])] for i in range(d)},
                    "problem_spec": problem_spec or {},
                    "already_selected_points": accepted.tolist(),  # [NEW]
                    "distance_rules": {
                        "min_l2_dist": float(min_l2_dist),
                        "notes": "New points must be at least min_l2_dist away from any already_selected_points."
                    },
                    "output_schema": {
                        "points": f"List of exactly {need} points, each is a list of {d} floats in input_order.",
                        "notes": "Optional string."
                    },
                    "hard_rules": [
                        "Return ONLY valid JSON.",
                        "Return exactly n_points points in `points` (no more, no less).",
                        "Each point must have exactly d floats.",
                        "All coordinates must be within bounds.",
                        "Do NOT repeat any already_selected_points.",
                        "Ensure new points are not near-duplicates of already_selected_points.",
                    ],
                    "soft_rules": [
                        "Maximize diversity / space-filling coverage.",
                        "Use any prior knowledge in problem_spec if provided.",
                    ],
                }

                dbg(f"[AGITCO] warmstart attempt={attempt} round={round_idx} need={need} model={model}")
                dbg(f"[AGITCO] sys_msg: {clip(sys_msg, 800)}")
                dbg(f"[AGITCO] user_payload(json): {clip(json.dumps(user_payload, ensure_ascii=False), 2000)}")

                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    temperature=temperature,
                )

                # ===== [NEW] robust parse =====
                if resp is None:
                    raise RuntimeError("Agitco returned None response.")

                choices = getattr(resp, "choices", None)
                if not choices or len(choices) == 0:
                    # (SDK model_dump / dict)
                    try:
                        raw = resp.model_dump()
                    except Exception:
                        try:
                            raw = resp.__dict__
                        except Exception:
                            raw = str(resp)
                    raise RuntimeError(f"Agitco response has no choices. raw={raw}")

                msg = getattr(choices[0], "message", None)
                if msg is None:
                    raise RuntimeError(f"Agitco choice.message is None. choice0={choices[0]}")

                content = getattr(msg, "content", None)
                if not content or not str(content).strip():
                    raise RuntimeError(f"Agitco message.content empty. msg={msg}")

                content = str(content).strip()
                dbg(f"[AGITCO] raw: {clip(content)}")

                data = json.loads(extract_json(content))
                out = _InitDesignOut.model_validate(data)
                pts_new = np.asarray(out.points, dtype=float)

                if pts_new.ndim != 2 or pts_new.shape[1] != d:
                    raise ValueError(f"LLM returned shape {pts_new.shape}, expected (n,{d}).")

                # clamp + ()
                pts_new = _clamp_and_dedupe_points(pts_new, bounds)

                # accepted ( round 1e-10)
                if accepted.shape[0] > 0 and pts_new.shape[0] > 0:
                    all_pts = np.vstack([accepted, pts_new])
                    all_pts = _clamp_and_dedupe_points(all_pts, bounds)
                    accepted = all_pts[:accepted.shape[0]]  # accepted
                    # “”:(:)
                    final_new = []
                    for p in all_pts:
                        # p accepted , accepted ,
                        if accepted.shape[0] < n_points:
                            # ( rounding)
                            p_r = np.round(p, 10)
                            acc_r = np.round(accepted, 10)
                            if acc_r.shape[0] == 0 or not np.any(np.all(acc_r == p_r, axis=1)):
                                final_new.append(p)
                    pts_new = np.asarray(final_new, dtype=float)

                for p in pts_new:
                    if accepted.shape[0] >= n_points:
                        break
                    if accepted.shape[0] > 0:
                        dists = np.linalg.norm(accepted - p[None, :], axis=1)
                        if float(dists.min()) < float(min_l2_dist):
                            continue
                    accepted = np.vstack([accepted, p[None, :]])

            # : n_points
            if accepted.shape[0] != n_points:
                raise ValueError(f"LLM failed to provide {n_points} valid diverse points. got={accepted.shape[0]}")

            # /
            accepted = _clamp_and_dedupe_points(accepted, bounds)

            if accepted.shape[0] != n_points:
                raise ValueError(f"After dedupe, got {accepted.shape[0]} points, expected {n_points}.")

            # :
            if _min_pairwise_l2(accepted) < float(min_l2_dist):
                raise ValueError("Generated points are too close to each other.")

            return accepted

        except Exception as e:
            last_err = e
            dbg(f"[AGITCO] attempt failed: {e}")
            accepted = np.empty((0, d), dtype=float)

    raise RuntimeError(f"Agitco warmstart failed after retries. Last error: {last_err}")


def load_shap_payload(path: str, feature_names: list):
    """
    :
    - shap_results.json()
    - shap_per_sample.csv()

     payload:
      payload["history"]["X"]["data"] : [N,d]
      payload["history"]["y"]["data"] : [N]
      payload["phi_mean"]["data"]     : [d,N]
    """
    path = Path(path)

    # ---------- JSON() ----------
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload

    # ---------- CSV() ----------
    if path.suffix.lower() == ".csv":
        # tab ;
        try:
            df = pd.read_csv(path)
            if len(df.columns) == 1 and "\t" in df.columns[0]:
                df = pd.read_csv(path, sep="\t")
        except Exception:
            df = pd.read_csv(path, sep="\t")

        shap_cols = [f"shap_{f}" for f in feature_names]

        for c in feature_names + shap_cols + ["reward"]:
            if c not in df.columns:
                raise ValueError(f"CSV missing required column: {c}")

        X = df[feature_names].astype(float).values          # [N,d]
        y = df["reward"].astype(float).values               # [N]
        phi_Nd = df[shap_cols].astype(float).values         # [N,d]
        phi_dN = phi_Nd.T                                   # [d,N]

        payload = {
            "feature_names": feature_names,
            "target_name": "reward",
            "history": {
                "X": {"data": X.tolist()},
                "y": {"data": y.tolist()},
            },
            "phi_mean": {
                "data": phi_dN.tolist()
            }
        }
        return payload

    raise ValueError(f"Unsupported file type: {path.suffix}")


def build_clustered_gmm_prior_from_shap(
    shap_path: str,
    feature_names: list,
    bounds: list,
    *,
    problem: Literal["min", "max"] = "min",
    shap_sign_eps: float = 0.0,           
    fallback_top_frac_by_y: float = 0.2,  

    k_max: int = 10,                      
    gmm_reg_covar: float = 1e-6,          
    
    ell_min_w_frac: float = 0.05,         
    ell_max_w_frac: float = 0.75,         
):
    payload = load_shap_payload(shap_path, feature_names)
    assert list(payload["feature_names"]) == list(feature_names), "feature_names mismatch"

    bounds_arr = np.asarray(bounds, dtype=float)
    X = np.asarray(payload["history"]["X"]["data"], dtype=float)   
    y = np.asarray(payload["history"]["y"]["data"], dtype=float)   
    phi = np.asarray(payload["phi_mean"]["data"], dtype=float)     
    d = X.shape[1]
    assert phi.shape == (d, X.shape[0])

    if problem == "min":
        maskG = np.all(phi < -shap_sign_eps, axis=0)   
    elif problem == "max":
        maskG = np.all(phi > +shap_sign_eps, axis=0)   
    else:
        raise ValueError("problem must be 'min' or 'max'")

    idxG = np.where(maskG)[0].astype(int)

    if idxG.size == 0:
        N = X.shape[0]
        m = max(1, int(np.floor(fallback_top_frac_by_y * N)))
        idxG = np.argsort(y)[:m] if problem == "min" else np.argsort(-y)[:m]
        idxG = idxG.astype(int)

    Xg = X[idxG]          
    phi_g = phi[:, idxG]  
    M = Xg.shape[0]

    w_point = np.sum(np.abs(phi_g), axis=0)  # [M]
    if (not np.isfinite(w_point).all()) or float(w_point.sum()) <= 1e-12:
        w_point = np.ones_like(w_point, dtype=float)
    w_point = w_point.astype(float)
    w_point = w_point / (w_point.sum() + 1e-12)  # normalize(,)

    Kmax = int(min(max(1, k_max), M))  # K <= M
    if Kmax == 1:
        resp = np.ones((M, 1), dtype=float)
    else:
        resp = None
        try:
            from sklearn.mixture import GaussianMixture

            best_bic = np.inf
            best_gmm = None
            for k in range(1, Kmax + 1):
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type="diag",  # ;diag
                    reg_covar=gmm_reg_covar,
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
        except Exception:

            try:
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score

                best_score = -1e9
                best_labels = None
                best_k = 1

                for k in range(2, Kmax + 1):
                    km = KMeans(n_clusters=k, random_state=0, n_init=10)
                    labels = km.fit_predict(Xg)
                    # / silhouette ,
                    try:
                        score = silhouette_score(Xg, labels)
                    except Exception:
                        continue
                    if score > best_score:
                        best_score = score
                        best_labels = labels
                        best_k = k

                if best_labels is None:
                    # :K=1
                    resp = np.ones((M, 1), dtype=float)
                else:
                    resp = np.zeros((M, best_k), dtype=float)
                    resp[np.arange(M), best_labels] = 1.0
            except Exception:
                # :K=1
                resp = np.ones((M, 1), dtype=float)

    K = resp.shape[1]


    w_eff = w_point[:, None] * resp  
    w_k = w_eff.sum(axis=0)          


    keep = w_k > 1e-12
    if not np.any(keep):
        keep = np.ones_like(w_k, dtype=bool)

    w_eff = w_eff[:, keep]
    w_k = w_k[keep]
    K = w_k.shape[0]

    w_k = (w_k / (w_k.sum() + 1e-12)).astype(float)  

    mus = []
    ells_list = []

    Δ = (bounds_arr[:, 1] - bounds_arr[:, 0]).astype(float)
    min_w = (ell_min_w_frac * Δ).astype(float)
    max_w = (ell_max_w_frac * Δ).astype(float)

    for k in range(K):
        wk_vec = w_eff[:, k]                       
        denom = float(wk_vec.sum() + 1e-12)

        mu_k = (wk_vec[:, None] * Xg).sum(axis=0) / denom  

        diff = Xg - mu_k[None, :]
        var_k = (wk_vec[:, None] * (diff ** 2)).sum(axis=0) / denom
        var_k = np.maximum(var_k, 1e-18)
        ell_k = np.sqrt(var_k)

        ell_k = np.minimum(np.maximum(ell_k, min_w), max_w)

        mus.append(mu_k.astype(float).tolist())
        ells_list.append(ell_k.astype(float).tolist())

    weights = w_k.astype(float).tolist()
    print(
        f"[CLUSTER] M={M}, K={len(mus)}, "
        f"weights={np.round(weights, 3)}"
    )
    return mus, ells_list, weights


GLOBAL_PIBO_GMM_REGISTRY = {}
def register_gmm_prior(acq_model, mus, ells_list, weights):
    GLOBAL_PIBO_GMM_REGISTRY[id(acq_model)] = (mus, ells_list, weights)


import inspect, dataclasses as _dc
if "kw_only" not in inspect.signature(_dc.dataclass).parameters:
    _orig_dataclass = _dc.dataclass
    def _dataclass_compat(*args, **kwargs):
        kwargs.pop("kw_only", None)
        return _orig_dataclass(*args, **kwargs)
    _dc.dataclass = _dataclass_compat


warnings.filterwarnings("ignore", message="divide by zero encountered in log2")
warnings.filterwarnings("ignore", message="invalid value encountered in log2")
warnings.filterwarnings("ignore", message="invalid value encountered in cast")

try:
    from typing import Annotated  # py>=3.10
except Exception:
    from typing_extensions import Annotated


logei = importlib.import_module("botorch.acquisition.logei")
if not hasattr(logei, "qLogEI_PiBO"):
    class qLogEI_PiBO(logei.qLogExpectedImprovement):
        r"""qLogEI with πBO: qLogEI(X) * pi(X) ** (beta / current_iter)."""
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

        def set_iter(self, n: int) -> None:
            self.current_iter = int(n)

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
                pi = self._reduce_pi(pi, base).to(base)
                pi = torch.clamp(pi, min=1e-12)
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

def _best_f_minimize(self) -> torch.Tensor:
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
    return x_adapt.min()

def _new_get_acqfs(self, n):
    if isinstance(self.acquisition_function, qLogEIWithPiBO_API):
        X_train, X_pending = self.get_acqf_input_tensors()
        objective_callable, constraint_callables, etas = self._get_objective_and_constraints()
        self._pibo_iter = getattr(self, "_pibo_iter", 0) + 1
        best_f = _best_f_minimize(self)
        afm = self.acquisition_function

        # ()
        _gmm_pack = GLOBAL_PIBO_GMM_REGISTRY.get(id(afm), None)

        def _pi_mix_gaussians(X: torch.Tensor, mus, ells, ws) -> torch.Tensor:
            xshape = X.shape
            Xflat = X.view(-1, xshape[-1])
            ws = X.new_tensor(ws, dtype=X.dtype)
            ws = ws / (ws.sum() + 1e-12)
            pis = []
            for mu_k, ell_k in zip(mus, ells):
                mu_k  = X.new_tensor(mu_k)
                ell_k = X.new_tensor(ell_k)
                Z = (Xflat - mu_k) / (ell_k + 1e-12)
                r2 = (Z**2).sum(dim=-1)
                pis.append(torch.exp(-0.5 * r2))
            Pis = torch.stack(pis, dim=-1)   # [M,K]
            mix = (Pis * ws).sum(dim=-1).clamp_min(1e-12)
            return mix.view(*xshape[:-1])

        def pi_fn(X: torch.Tensor) -> torch.Tensor:
            # :
            if _gmm_pack is not None:
                mus, ells_list, weights = _gmm_pack
                return _pi_mix_gaussians(X, mus, ells_list, weights)
            # :
            if afm.prior_mu is not None and afm.prior_ell is not None:
                mu  = X.new_tensor(afm.prior_mu)
                ell = X.new_tensor(afm.prior_ell)
                Z   = (X - mu) / (ell + 1e-12)
                r2  = (Z**2).sum(dim=-1)
                return torch.exp(-0.5 * r2).clamp_min(1e-12)
            # :
            return X.new_ones(X.shape[:-1])

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


import matplotlib.pyplot as plt
from scipy.stats.qmc import LatinHypercube
import bofire.strategies.api as strategies
from bofire.data_models.acquisition_functions.api import qLogEIWithPiBO
from bofire.data_models.strategies.api import SoboStrategy as SoboStrategyDM
from bofire.data_models.features.api import ContinuousOutput, ContinuousInput
from bofire.data_models.objectives.api import MinimizeObjective
from bofire.data_models.domain.api import Domain, Inputs, Outputs

# ========== Hartmann6 ==========
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
        s += alpha[i] * np.exp(-np.sum(A[i] * (x - P[i])**2))
    return -s  # minimize

# ========== Latin Hypercube ==========
def generate_lhs_samples(num_samples, dim=6):
    sampler = LatinHypercube(d=dim)
    return sampler.random(n=num_samples)

SHAP_PARENT_DIR = Path(".")  # TODO: set your path
OUT_PARENT_DIR = Path(".")  # TODO: set your path
OUT_PARENT_DIR.mkdir(parents=True, exist_ok=True)


def run_one_pibo(run_idx: int, shap_csv_path: Path, out_dir: Path):
    num_initial_samples = 10

    feature_names = [f"x{i + 1}" for i in range(6)]
    bounds = [(0.0, 1.0)] * 6

    impact_text = load_impact_summary_for_run(run_idx)
    impact_text = impact_text[:4000]  # 4000 chars
    print(impact_text)
    problem_spec = {
        "objective": "minimize Hartmann6 (black-box).",
        "prior_knowledge": {
            "impact_summary": impact_text
        },
        "notes": [
            "Generate 10 diverse, space-filling points in [0,1]^6.",
            "If impact_summary suggests important regions or trends, bias points moderately toward them, but keep coverage."
        ]
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {"run": run_idx, "init_method": None, "llm_model": None, "error": None, "impact_found": bool(impact_text)}

    try:
        init_pts = generate_init_points_with_agitco(
            n_points=num_initial_samples,
            feature_names=feature_names,
            bounds=bounds,
            model=WARMSTART_MODEL,
            temperature=0.2,
            max_retries=2,
            problem_spec=problem_spec,  
        )

        initial_data = pd.DataFrame(init_pts[:num_initial_samples], columns=[f"x{i+1}" for i in range(6)])
        status["init_method"] = "agitco_llm"
        status["llm_model"] = WARMSTART_MODEL
        initial_data.to_csv(out_dir / "init_points.csv", index=False)
        (out_dir / "init_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RUN {run_idx:03d}] Agitco warmstart OK. model={WARMSTART_MODEL}, n={len(initial_data)}")

    except Exception as e:
        lhs_samples = generate_lhs_samples(num_initial_samples, dim=6)
        initial_data = pd.DataFrame(lhs_samples, columns=[f"x{i+1}" for i in range(6)])
        status["init_method"] = "lhs_fallback"
        status["error"] = str(e)
        initial_data.to_csv(out_dir / "init_points.csv", index=False)
        (out_dir / "init_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[RUN {run_idx:03d}] Agitco warmstart FAILED -> fallback LHS. err={e}")

    initial_data["reward"] = initial_data.apply(lambda row: hartmann6(row.values[:6]), axis=1)

    inputs = Inputs(features=[ContinuousInput(key=f"x{i+1}", bounds=(0, 1)) for i in range(6)])
    objective = MinimizeObjective(w=1.0)
    reward = ContinuousOutput(key="reward", objective=objective)
    outputs = Outputs(features=[reward])
    domain = Domain(inputs=inputs, outputs=outputs)

    feature_names = [f"x{i+1}" for i in range(6)]
    bounds = [(0.0, 1.0)] * 6

    prior_mus, prior_ells_list, prior_weights = build_clustered_gmm_prior_from_shap(
        str(shap_csv_path), feature_names, bounds,
        problem="min",
        shap_sign_eps=0.0,
        fallback_top_frac_by_y=0.2,
        k_max=10,
        gmm_reg_covar=1e-6,
        ell_min_w_frac=0.05,
        ell_max_w_frac=0.75,
    )
    print(f"[RUN {run_idx:03d}] shap={shap_csv_path}  num_clusters={len(prior_mus)}")

    acq = qLogEIWithPiBO(
        n_mc_samples=512,
        beta=2.0,
        reduce="max",
    )
    register_gmm_prior(acq, prior_mus, prior_ells_list, prior_weights)

    sobo_dm = SoboStrategyDM(domain=domain, acquisition_function=acq)
    sobo_strategy = strategies.map(sobo_dm)

    sobo_strategy.tell(initial_data)
    all_data = initial_data.copy()


    trace_rows = []
    best_so_far = float(all_data["reward"].min())

    for _, row in initial_data.iterrows():
        trace_rows.append({
            "iter": 0,
            **{f"x{j+1}": float(row[f"x{j+1}"]) for j in range(6)},
            "reward": float(row["reward"]),
            "best_so_far": best_so_far,
            "is_initial": True,
        })

    MAX_ITER = 100
    best_known_min = -3.322368011415515  # Hartmann6
    threshold = 0.95 * best_known_min

    log_values = list(initial_data["reward"])

    for iteration in range(MAX_ITER):
        new_experiment = sobo_strategy.ask(1)
        new_experiment["reward"] = new_experiment.apply(lambda row: hartmann6(row.values[:6]), axis=1)
        sobo_strategy.tell(new_experiment)

        all_data = pd.concat([all_data, new_experiment], ignore_index=True)

        fx = float(new_experiment["reward"].iloc[0])
        log_values.append(fx)
        print(f"[RUN {run_idx:03d}] Iter {iteration+1}, f(x) = {fx}")

        x_row = new_experiment.iloc[0]
        best_so_far = min(best_so_far, fx)
        trace_rows.append({
            "iter": iteration + 1,
            **{f"x{j+1}": float(x_row[f"x{j+1}"]) for j in range(6)},
            "reward": fx,
            "best_so_far": best_so_far,
            "is_initial": False,
        })

        if fx <= threshold:
            print(f"[RUN {run_idx:03d}] Stopping at iteration {iteration+1} with value {fx}")
            break

    trace_df = pd.DataFrame(trace_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "bo_trace.csv"
    trace_df.to_csv(out_csv, index=False)
    print(f"[RUN {run_idx:03d}] [SAVE] BO trace -> {out_csv.resolve()}")

    # : 100 , png
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(log_values) + 1), log_values, marker="o")
    plt.axhline(y=best_known_min, color="red", linestyle="--", label="Global minimum")
    plt.axhline(y=threshold, color="green", linestyle="--", label="95% threshold")
    plt.xlabel("Iteration")
    plt.ylabel("Hartmann6 value (minimize)")
    plt.title(f"PiBO qLogEI (clustered multi-peak prior) - run {run_idx:03d}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    fig_path = out_dir / "trace.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"[RUN {run_idx:03d}] [SAVE] trace plot -> {fig_path.resolve()}")

NUM_RUNS = 100

for i in range(1, NUM_RUNS + 1):
    shap_csv = SHAP_PARENT_DIR / f"run_{i:03d}" / "shap_per_sample.csv"
    if not shap_csv.exists():
        raise FileNotFoundError(f"Missing: {shap_csv}")

    out_dir = OUT_PARENT_DIR / f"run_{i:03d}"
    run_one_pibo(i, shap_csv, out_dir)

print(f"[DONE] All {NUM_RUNS} runs saved under: {OUT_PARENT_DIR.resolve()}")
