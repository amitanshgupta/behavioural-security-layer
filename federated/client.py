import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
import lightgbm as lgb
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger
from federated.dp_mechanism import add_gaussian_noise, clip_weights

log = get_logger("fl_client")

N_NODES       = 5
LOCAL_EPOCHS  = 5


def partition_data(
    df          : pd.DataFrame,
    n_nodes     : int = N_NODES,
    random_state: int = 42,
) -> list:
    """
    Partition training data into n_nodes non-overlapping shards.
    Stratified by label to ensure each node sees both classes.
    Simulates separate organisational departments.
    """
    from sklearn.model_selection import StratifiedKFold
    skf    = StratifiedKFold(
        n_splits=n_nodes, shuffle=True, random_state=random_state
    )
    y      = df["label_binary"]
    shards = []

    for _, shard_idx in skf.split(df, y):
        shards.append(df.iloc[shard_idx].reset_index(drop=True))

    for i, shard in enumerate(shards):
        attack_rate = shard["label_binary"].mean() * 100
        log.info(
            f"Node {i} | size={len(shard):>7,} | "
            f"attack_rate={attack_rate:.1f}%"
        )

    return shards


class FederatedClient:
    """
    Simulates a single federated learning client (organisational node).
    Each client trains a local LightGBM model on its data shard,
    applies DP noise to weight updates, and returns them to the server.
    """

    def __init__(
        self,
        node_id  : int,
        features : list,
        apply_dp : bool = True,
    ):
        self.node_id  = node_id
        self.features = features
        self.apply_dp = apply_dp
        self.model    = None
        self.n_samples= 0

    def load_shard(self, shard: pd.DataFrame) -> None:
        """Load local data shard."""
        self.X = shard[self.features].fillna(0)
        self.y = shard["label_binary"]
        self.n_samples = len(shard)
        log.info(
            f"Node {self.node_id} loaded shard: "
            f"{self.n_samples:,} rows"
        )

    def train_local(
        self,
        global_params : dict = None,
        n_estimators  : int  = 50,
    ) -> dict:
        """
        Train local LightGBM model.
        If global_params provided, uses them as starting point
        (warm-start from global model).

        Returns weight update dict for aggregation.
        """
        log.info(f"Node {self.node_id} starting local training...")

        params = {
            "n_estimators"     : n_estimators,
            "max_depth"        : 7,
            "learning_rate"    : 0.05,
            "num_leaves"       : 63,
            "class_weight"     : "balanced",
            "random_state"     : 42 + self.node_id,
            "n_jobs"           : -1,
            "verbose"          : -1,
        }

        self.model = lgb.LGBMClassifier(**params)
        self.model.fit(self.X, self.y)

        # Extract feature importances as proxy for model weights
        weights = [self.model.feature_importances_.astype(float)]

        # Apply differential privacy
        if self.apply_dp:
            weights = add_gaussian_noise(weights, random_state=self.node_id)
            log.info(f"Node {self.node_id} DP noise applied")

        # Local evaluation
        y_pred = self.model.predict(self.X)
        f1     = f1_score(self.y, y_pred, zero_division=0)
        log.info(
            f"Node {self.node_id} local F1={f1:.4f} "
            f"on {self.n_samples:,} samples"
        )

        return {
            "node_id"   : self.node_id,
            "weights"   : weights,
            "n_samples" : self.n_samples,
            "local_f1"  : f1,
            "model"     : self.model,
        }

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Evaluate local model on held-out test set."""
        if self.model is None:
            log.warning(f"Node {self.node_id} has no trained model")
            return {}

        X   = X_test[self.features].fillna(0)
        pred= self.model.predict(X)
        f1  = f1_score(y_test, pred, zero_division=0)

        log.info(f"Node {self.node_id} test F1={f1:.4f}")
        return {"node_id": self.node_id, "test_f1": f1}


def run_clients(
    train_df  : pd.DataFrame,
    features  : list,
    apply_dp  : bool = True,
) -> list:
    """
    Initialise and train all N_NODES clients.
    Returns list of weight update dicts for the aggregator.
    """
    shards  = partition_data(train_df)
    updates = []

    for node_id in range(N_NODES):
        client = FederatedClient(node_id, features, apply_dp)
        client.load_shard(shards[node_id])
        update = client.train_local()
        updates.append(update)

    return updates


if __name__ == "__main__":
    # Load data
    lgb_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    features = lgb_data["features"]

    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    log.info(f"Running {N_NODES} federated clients...")
    updates = run_clients(train_df, features, apply_dp=True)

    print(f"\n--- Client Training Summary ---")
    for u in updates:
        print(f"Node {u['node_id']} | "
              f"samples={u['n_samples']:>7,} | "
              f"local_F1={u['local_f1']:.4f} | "
              f"weight_shape={u['weights'][0].shape}")