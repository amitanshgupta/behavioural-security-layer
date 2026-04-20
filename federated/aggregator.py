import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("fl_aggregator")


def fedavg(
    updates     : list,
    weighted    : bool = True,
) -> np.ndarray:
    """
    Federated Averaging (FedAvg) algorithm.
    McMahan et al. (2017)

    Aggregates weight updates from all clients by computing
    a weighted average proportional to each client's data size.

    Args:
        updates  : list of dicts from client.train_local()
                   each must have 'weights' and 'n_samples'
        weighted : if True, weight by n_samples (standard FedAvg)
                   if False, simple mean (equal weighting)

    Returns:
        Aggregated weight array
    """
    if not updates:
        raise ValueError("No updates to aggregate")

    total_samples = sum(u["n_samples"] for u in updates)
    weight_arrays = [u["weights"][0] for u in updates]

    if weighted:
        aggregated = np.zeros_like(weight_arrays[0], dtype=float)
        for u, w in zip(updates, weight_arrays):
            client_weight = u["n_samples"] / total_samples
            aggregated   += client_weight * w
        log.info(
            f"FedAvg (weighted) across {len(updates)} nodes "
            f"({total_samples:,} total samples)"
        )
    else:
        aggregated = np.mean(weight_arrays, axis=0)
        log.info(
            f"FedAvg (uniform) across {len(updates)} nodes"
        )

    return aggregated


def compute_aggregation_stats(
    updates     : list,
    aggregated  : np.ndarray,
) -> dict:
    """
    Compute statistics about the aggregation round.
    Measures convergence by checking variance across client updates.

    Returns dict with:
      - mean_local_f1    : average F1 across all clients
      - weight_variance  : variance of weight arrays (convergence signal)
      - weight_drift     : max deviation of any client from aggregated mean
    """
    local_f1s    = [u["local_f1"] for u in updates]
    weight_arrays= np.array([u["weights"][0] for u in updates])

    weight_variance = float(np.var(weight_arrays, axis=0).mean())
    deviations      = [
        float(np.linalg.norm(w - aggregated))
        for w in weight_arrays
    ]
    weight_drift = max(deviations)

    stats = {
        "mean_local_f1"  : round(float(np.mean(local_f1s)), 4),
        "std_local_f1"   : round(float(np.std(local_f1s)), 4),
        "weight_variance": round(weight_variance, 6),
        "weight_drift"   : round(weight_drift, 4),
        "n_clients"      : len(updates),
        "total_samples"  : sum(u["n_samples"] for u in updates),
    }

    log.info(
        f"Aggregation stats - "
        f"mean_F1={stats['mean_local_f1']:.4f} "
        f"weight_variance={stats['weight_variance']:.6f} "
        f"weight_drift={stats['weight_drift']:.4f}"
    )
    return stats


def check_convergence(
    round_stats : list,
    threshold   : float = 0.001,
) -> bool:
    """
    Check if federated training has converged.
    Convergence criterion from proposal: val loss change < 0.001

    Uses weight_variance as proxy — when variance stops decreasing
    significantly, clients have converged to similar models.

    Args:
        round_stats : list of stats dicts from previous rounds
        threshold   : minimum variance change to continue training

    Returns:
        True if converged, False if should continue
    """
    if len(round_stats) < 2:
        return False

    prev_var = round_stats[-2]["weight_variance"]
    curr_var = round_stats[-1]["weight_variance"]
    change   = abs(prev_var - curr_var)

    converged = change < threshold
    log.info(
        f"Convergence check - "
        f"variance_change={change:.6f} "
        f"({'CONVERGED' if converged else 'CONTINUE'})"
    )
    return converged


if __name__ == "__main__":
    import pandas as pd
    import joblib
    from utils.constants import PROCESSED_DIR, METADATA_DIR
    from federated.client import run_clients

    lgb_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    features = lgb_data["features"]
    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")

    updates    = run_clients(train_df, features, apply_dp=True)
    aggregated = fedavg(updates)
    stats      = compute_aggregation_stats(updates, aggregated)

    print(f"\n--- FedAvg Aggregation Results ---")
    print(f"Aggregated weight shape : {aggregated.shape}")
    print(f"Mean local F1           : {stats['mean_local_f1']:.4f}")
    print(f"Weight variance         : {stats['weight_variance']:.6f}")
    print(f"Weight drift (max)      : {stats['weight_drift']:.4f}")
    print(f"Total samples           : {stats['total_samples']:,}")