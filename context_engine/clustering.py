import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import (
    PROCESSED_DIR,
    CLUSTERING_FEATURES,
    N_CLUSTERS,
    METADATA_DIR,
)
from utils.logger import get_logger

log = get_logger("clustering")

# Paths for saving fitted objects
SCALER_PATH  = METADATA_DIR / "cluster_scaler.pkl"
KMEANS_PATH  = METADATA_DIR / "kmeans_model.pkl"
MAPPING_PATH = METADATA_DIR / "cluster_profiles.csv"


def load_features(split: str = "train") -> pd.DataFrame:
    path = PROCESSED_DIR / f"nslkdd_{split}_features.csv"
    log.info(f"Loading {path}")
    return pd.read_csv(path)


def select_clustering_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only features defined in CLUSTERING_FEATURES that exist in df."""
    available = [f for f in CLUSTERING_FEATURES if f in df.columns]
    missing   = [f for f in CLUSTERING_FEATURES if f not in df.columns]
    if missing:
        log.warning(f"Missing clustering features (skipped): {missing}")
    log.info(f"Using {len(available)} features for clustering")
    return df[available].copy()


def fit_scaler(X: pd.DataFrame) -> tuple[StandardScaler, np.ndarray]:
    """Fit StandardScaler and return scaler + scaled array."""
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info("Fitted StandardScaler on clustering features")
    return scaler, X_scaled


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = range(3, 12),
) -> int:
    """
    Evaluate K-Means for k in k_range using silhouette score.
    Returns the k with the highest silhouette score.
    Uses a 20% sample for speed on large datasets.
    """
    log.info(f"Searching optimal k in range {list(k_range)}...")

    sample_size = min(10_000, len(X_scaled))
    idx         = np.random.choice(len(X_scaled), sample_size, replace=False)
    X_sample    = X_scaled[idx]

    scores = {}
    for k in k_range:
        km    = KMeans(n_clusters=k, random_state=42, n_init=5)
        lbls  = km.fit_predict(X_sample)
        score = silhouette_score(X_sample, lbls)
        scores[k] = score
        log.info(f"  k={k:>2} → silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    log.info(f"Optimal k selected: {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k


def fit_kmeans(
    X_scaled: np.ndarray,
    k: int,
) -> KMeans:
    """Fit final K-Means model on full dataset."""
    log.info(f"Fitting K-Means with k={k} on {len(X_scaled):,} samples...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    km.fit(X_scaled)
    log.info("K-Means fitting complete")
    return km


def build_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
) -> pd.DataFrame:
    """
    For each cluster compute:
      - size and percentage of total
      - mean of each clustering feature
      - attack rate (% label_binary=1)
      - dominant protocol (most common protocol_type value)
    """
    df = df.copy()
    df["cluster"] = cluster_labels

    profiles = []
    for c in sorted(df["cluster"].unique()):
        grp  = df[df["cluster"] == c]
        prof = {"cluster": c, "size": len(grp),
                "pct_total": round(len(grp) / len(df) * 100, 2)}

        # Feature means
        for feat in CLUSTERING_FEATURES:
            if feat in grp.columns:
                prof[f"mean_{feat}"] = round(grp[feat].mean(), 4)

        # Attack rate
        if "label_binary" in grp.columns:
            prof["attack_rate"] = round(grp["label_binary"].mean() * 100, 2)

        profiles.append(prof)

    profile_df = pd.DataFrame(profiles)
    log.info(f"Cluster profiles built for {len(profiles)} clusters")
    return profile_df


def run_clustering(
    optimize_k: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Master function.

    Args:
        optimize_k : if True, search for best k via silhouette.
                     if False, use N_CLUSTERS from constants.

    Returns:
        (train_df_with_clusters, cluster_profiles_df)
    """
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = load_features("train")
    X        = select_clustering_features(train_df)

    # Scale
    scaler, X_scaled = fit_scaler(X)
    joblib.dump(scaler, SCALER_PATH)
    log.info(f"Scaler saved -> {SCALER_PATH}")

    # Choose k
    k = find_optimal_k(X_scaled) if optimize_k else N_CLUSTERS
    log.info(f"Using k={k}")

    # Fit
    km = fit_kmeans(X_scaled, k)
    joblib.dump(km, KMEANS_PATH)
    log.info(f"KMeans model saved -> {KMEANS_PATH}")

    # Assign labels
    train_df["cluster"] = km.labels_

    # Profiles
    profiles = build_cluster_profiles(train_df, km.labels_)
    profiles.to_csv(MAPPING_PATH, index=False)
    log.info(f"Profiles saved -> {MAPPING_PATH}")

    # Summary
    log.info("\nCluster summary:")
    for _, row in profiles.iterrows():
        log.info(
            f"  Cluster {int(row['cluster']):>2} | "
            f"size={int(row['size']):>7,} ({row['pct_total']:>5.1f}%) | "
            f"attack_rate={row['attack_rate']:>6.1f}%"
        )

    return train_df, profiles


def assign_clusters_to_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Load fitted scaler + KMeans and assign clusters to test split.
    Call this after run_clustering() has been run at least once.
    """
    scaler = joblib.load(SCALER_PATH)
    km     = joblib.load(KMEANS_PATH)

    X        = select_clustering_features(test_df)
    X_scaled = scaler.transform(X)

    test_df["cluster"] = km.predict(X_scaled)
    log.info(f"Clusters assigned to test split. "
             f"Distribution:\n{test_df['cluster'].value_counts().sort_index().to_string()}")
    return test_df


if __name__ == "__main__":
    train_df, profiles = run_clustering(optimize_k=True)

    print("\nCluster Profiles:")
    print(profiles[["cluster", "size", "pct_total", "attack_rate"]].to_string(index=False))

    # Assign to test too
    test_df = load_features("test")
    test_df = assign_clusters_to_test(test_df)
    print(f"\nTest cluster distribution:")
    print(test_df["cluster"].value_counts().sort_index().to_string())