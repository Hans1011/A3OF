"""
A3OF Bayesian optimization module for droplet microfluidic recipes.
It can run as a standalone single-recommendation script or be imported by Main.py.
"""

import os
import numpy as np
import pandas as pd

try:
    import bofire.strategies.api as strategies
    from bofire.data_models.acquisition_functions.api import qEI
    from bofire.data_models.strategies.api import SoboStrategy
    from bofire.data_models.domain.api import Domain, Inputs, Outputs
    from bofire.data_models.features.api import (
        ContinuousInput, DiscreteInput, CategoricalInput, ContinuousOutput
    )
    from bofire.data_models.objectives.api import MaximizeObjective
    BOFIRE_AVAILABLE = True
except ImportError:
    BOFIRE_AVAILABLE = False

# ============================================================
# 1. Search space
# ============================================================

# Water-phase surfactants.
SW = CategoricalInput(
    key="SW",
    categories=["ODEA", "LDEA", "CDEA"],
    allowed=[True, True, True]
)

# Water-phase surfactant ratios.
Rwater = DiscreteInput(
    key="Rwater",
    values=[0, 0.0002, 0.001, 0.005]
)

# Oil-phase surfactant ratios.
ROil = DiscreteInput(
    key="ROil",
    values=[0, 0.0002, 0.001, 0.005]
)

# Oil phase options.
Oil = CategoricalInput(
    key="Oil",
    categories=[
        "PMX-10", "PMX-20", "PMX-50", "PMX-100", "PMX-200",
        "PMX-500", "PMX-1000", "PMX-30000", "PMX-60000",
        "7500", "FC-40", "PFPE", "mineral", "PMX-350"
    ],
    allowed=[True] * 14
)


SO = CategoricalInput(
    key="SO",
    categories=["AEO-5", "Pico-surf", "Perfluoro", "TEGO-410"],
    allowed=[True, True, True, True]
)

# Ion concentration in mol/L.
Ion = DiscreteInput(
    key="Ion",
    values=[0, 0.5, 1, 1.5, 2, 2.5, 3]
)

# Optimization target.
reward = ContinuousOutput(
    key="reward",
    objective=MaximizeObjective(w=1.0)
)

# ============================================================
# 2. BoFire domain
# ============================================================

def build_bo_domain():
    """Build and return the BoFire optimization domain."""
    input_features = Inputs(features=[SW, Rwater, ROil, Oil, SO, Ion])
    output_features = Outputs(features=[reward])
    return Domain(inputs=input_features, outputs=output_features)


# ============================================================
# 3. Bayesian optimization
# ============================================================

def get_next_recommendation(csv_path="BO-test.csv"):
    """
    Read historical experiments from CSV, run Bayesian optimization,
    and return the next recommended parameter set.
    """
    if not BOFIRE_AVAILABLE:
        raise RuntimeError("BoFire is not installed, so Bayesian optimization cannot run.")
    return bo_recommend(csv_path)


def bo_recommend(csv_path):
    """
    Interface called by Main.py.
    Run BO and fall back to random recommendation when data are insufficient.
    Return a one-row DataFrame.
    """
    domain = build_bo_domain()
    experiments = pd.read_csv(csv_path)
    domain_cols = [f.key for f in domain.inputs.features] + [domain.outputs.features[0].key]

    # Keep only rows with an observed reward.
    experiments_filtered = experiments[domain_cols].dropna(subset=["reward"]).copy()

    if len(experiments_filtered) < 3:
        print("Warning: fewer than 3 valid rows; using random recommendation.")
        return random_recommend()

    if not BOFIRE_AVAILABLE:
        print("Warning: BoFire is unavailable; using random recommendation.")
        return random_recommend()

    strategy_model = SoboStrategy(domain=domain, acquisition_function=qEI())
    strategy = strategies.map(strategy_model)
    strategy.tell(experiments_filtered)
    new_exp = strategy.ask(1)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', None)
    print(f"BO recommendation: {new_exp.to_dict('records')[0]}")
    return new_exp


def random_recommend():
    """Randomly sample one parameter set when BO is unavailable or undertrained."""
    domain = build_bo_domain()
    row = {}
    for f in domain.inputs.features:
        if hasattr(f, 'categories'):
            row[f.key] = np.random.choice(f.categories)
        elif hasattr(f, 'values'):
            row[f.key] = np.random.choice(f.values)
    print(f"Random recommendation: {row}")
    return pd.DataFrame([row])


def append_recommendation_to_csv(csv_path="BO-test.csv"):
    """
    Read the CSV, recommend the next parameter set, append it to the CSV,
    and return the recommended parameters as a dict.
    """
    new_exp = get_next_recommendation(csv_path)

    # Append recommendation to CSV.
    experiments = pd.read_csv(csv_path)
    experiments = pd.concat([experiments, new_exp], ignore_index=True)
    experiments.to_csv(csv_path, index=False)

    print(f"New recommendation appended to {csv_path}; total rows: {len(experiments)}")

    # Return the newly appended row as a parameter dict.
    last_row = experiments.iloc[-1]
    params = {
        "SW":   str(last_row["SW"]).strip(),
        "SCW":  float(last_row["Rwater"]),
        "SCO":  float(last_row["ROil"]),
        "OT":   str(last_row["Oil"]).strip(),
        "SO":   str(last_row["SO"]).strip(),
        "Ion":  float(last_row["Ion"]),
    }
    return params


# ============================================================
# 4. Standalone entry
# ============================================================

if __name__ == "__main__":
    append_recommendation_to_csv("BO-test.csv")
