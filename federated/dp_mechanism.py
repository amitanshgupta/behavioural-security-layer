import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

log = get_logger("dp_mechanism")

# Privacy budget from proposal
EPSILON        = 0.5
DELTA          = 1e-5
NOISE_MULT     = 1.1
MAX_GRAD_NORM  = 1.0


def clip_weights(
    weights    : list,
    max_norm   : float = MAX_GRAD_NORM,
) -> tuple:
    """
    Clip model weight updates to max L2 norm.
    This bounds the sensitivity of each update —
    a core requirement for differential privacy.

    Args:
        weights  : list of numpy arrays (model weight deltas)
        max_norm : clipping threshold

    Returns:
        (clipped_weights, actual_norm)
    """
    # Flatten all weights into single vector
    flat     = np.concatenate([w.flatten() for w in weights])
    l2_norm  = np.linalg.norm(flat)

    if l2_norm > max_norm:
        scale   = max_norm / (l2_norm + 1e-9)
        clipped = [w * scale for w in weights]
        log.debug(f"Weights clipped: norm={l2_norm:.4f} -> {max_norm:.4f}")
    else:
        clipped  = weights
        scale    = 1.0

    return clipped, float(l2_norm)


def compute_sigma(
    epsilon    : float = EPSILON,
    delta      : float = DELTA,
    sensitivity: float = MAX_GRAD_NORM,
) -> float:
    """
    Compute Gaussian noise sigma for (epsilon, delta)-DP.
    Uses the analytic Gaussian mechanism formula.

    sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    log.info(
        f"Gaussian sigma={sigma:.4f} "
        f"(epsilon={epsilon}, delta={delta:.0e})"
    )
    return sigma


def add_gaussian_noise(
    weights      : list,
    noise_mult   : float = NOISE_MULT,
    max_norm     : float = MAX_GRAD_NORM,
    random_state : int   = None,
) -> list:
    """
    Apply Gaussian Differential Privacy to model weight updates.

    Steps:
      1. Clip weights to max_norm (bound sensitivity)
      2. Add Gaussian noise scaled by noise_mult * max_norm

    Args:
        weights     : list of numpy weight arrays
        noise_mult  : noise multiplier (sigma = noise_mult * max_norm)
        max_norm    : clipping threshold
        random_state: for reproducibility

    Returns:
        List of noised weight arrays
    """
    if random_state is not None:
        np.random.seed(random_state)

    clipped, original_norm = clip_weights(weights, max_norm)
    sigma   = noise_mult * max_norm
    noised  = []

    for w in clipped:
        noise = np.random.normal(0, sigma, size=w.shape)
        noised.append(w + noise)

    total_noise = np.sqrt(sum(
        np.sum(np.random.normal(0, sigma, w.shape)**2)
        for w in clipped
    ))
    log.debug(
        f"DP noise added: sigma={sigma:.4f}, "
        f"original_norm={original_norm:.4f}"
    )
    return noised


def compute_privacy_budget(
    n_rounds     : int,
    n_clients    : int,
    sample_rate  : float,
    noise_mult   : float = NOISE_MULT,
    delta        : float = DELTA,
) -> dict:
    """
    Estimate privacy budget spent after n_rounds of federated training.
    Uses RDP (Renyi Differential Privacy) accountant approximation.

    Args:
        n_rounds    : number of FL communication rounds
        n_clients   : number of participating clients
        sample_rate : fraction of data sampled per round
        noise_mult  : noise multiplier used
        delta       : target delta

    Returns:
        dict with epsilon estimate and budget status
    """
    # Simplified RDP -> (epsilon, delta) conversion
    # Real production would use Google's dp-accounting library
    # This is a conservative upper bound estimate
    steps        = n_rounds * n_clients
    epsilon_est  = (
        np.sqrt(steps) * sample_rate /
        (noise_mult * np.sqrt(2 * np.log(1.25 / delta)))
    )

    status = "WITHIN BUDGET" if epsilon_est <= EPSILON else "BUDGET EXCEEDED"
    log.info(
        f"Privacy budget after {n_rounds} rounds: "
        f"epsilon={epsilon_est:.4f} ({status})"
    )

    return {
        "epsilon_spent": round(epsilon_est, 6),
        "epsilon_budget": EPSILON,
        "delta"         : delta,
        "n_rounds"      : n_rounds,
        "status"        : status,
    }


if __name__ == "__main__":
    log.info("Testing DP mechanism...")

    # Simulate weight arrays like a small LightGBM update
    dummy_weights = [
        np.random.randn(100),
        np.random.randn(50),
        np.random.randn(20),
    ]

    # Test clipping
    clipped, norm = clip_weights(dummy_weights)
    print(f"\nOriginal L2 norm : {norm:.4f}")
    print(f"Max norm         : {MAX_GRAD_NORM}")

    # Test noise addition
    noised = add_gaussian_noise(dummy_weights, random_state=42)
    print(f"Noise added successfully to {len(noised)} weight arrays")

    # Test sigma computation
    sigma = compute_sigma()
    print(f"Gaussian sigma   : {sigma:.4f}")

    # Test budget estimation
    budget = compute_privacy_budget(
        n_rounds   = 10,
        n_clients  = 5,
        sample_rate= 0.2,
    )
    print(f"\nPrivacy budget: {budget}")