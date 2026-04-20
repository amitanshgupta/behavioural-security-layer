import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.constants import PROCESSED_DIR, METADATA_DIR
from utils.logger import get_logger
from federated.client import run_clients, N_NODES
from federated.aggregator import fedavg, compute_aggregation_stats, check_convergence
from federated.dp_mechanism import compute_privacy_budget
from sklearn.metrics import f1_score, roc_auc_score, classification_report

log = get_logger("fl_server")

FL_ROUNDS     = 10
CONVERGENCE_T = 0.001
SERVER_PATH   = METADATA_DIR / "fl_server_results.pkl"


class FederatedServer:
    """
    Central federated learning coordinator.
    Orchestrates multiple rounds of:
      1. Broadcast global model params to clients
      2. Clients train locally with DP
      3. Collect and aggregate updates via FedAvg
      4. Check convergence
      5. Track privacy budget
    """

    def __init__(
        self,
        features    : list,
        n_rounds    : int  = FL_ROUNDS,
        apply_dp    : bool = True,
        n_nodes     : int  = N_NODES,
    ):
        self.features    = features
        self.n_rounds    = n_rounds
        self.apply_dp    = apply_dp
        self.n_nodes     = n_nodes
        self.round_stats = []
        self.global_weights = None
        self.best_round     = 0
        self.best_f1        = 0.0

    def run_round(
        self,
        train_df  : pd.DataFrame,
        round_num : int,
    ) -> dict:
        """Execute one full FL round."""
        log.info(f"\n--- FL Round {round_num}/{self.n_rounds} ---")

        # All clients train locally
        updates = run_clients(train_df, self.features, self.apply_dp)

        # Aggregate via FedAvg
        aggregated = fedavg(updates)
        stats      = compute_aggregation_stats(updates, aggregated)
        stats["round"] = round_num

        self.global_weights = aggregated
        self.round_stats.append(stats)

        # Track best round by F1
        if stats["mean_local_f1"] > self.best_f1:
            self.best_f1    = stats["mean_local_f1"]
            self.best_round = round_num

        return stats, updates

    def run_federation(
        self,
        train_df  : pd.DataFrame,
        test_df   : pd.DataFrame,
    ) -> dict:
        """
        Run full federated training for n_rounds.
        Checks convergence after each round.
        Tracks privacy budget throughout.
        """
        log.info(
            f"Starting federated training: "
            f"{self.n_rounds} rounds, "
            f"{self.n_nodes} nodes, "
            f"DP={'ON' if self.apply_dp else 'OFF'}"
        )

        all_updates   = []
        round_results = []

        for round_num in range(1, self.n_rounds + 1):
            stats, updates = self.run_round(train_df, round_num)
            all_updates.append(updates)
            round_results.append(stats)

            # Check convergence after round 2
            if round_num >= 2:
                converged = check_convergence(self.round_stats)
                if converged:
                    log.info(
                        f"Converged at round {round_num} — "
                        f"stopping early"
                    )
                    break

        # Privacy budget after all rounds
        budget = compute_privacy_budget(
            n_rounds    = len(round_results),
            n_clients   = self.n_nodes,
            sample_rate = 1.0 / self.n_nodes,
        )

        # Evaluate last round's client models on test set
        last_updates = all_updates[-1]
        test_results = self._evaluate_on_test(
            last_updates, test_df
        )

        # Compare federated vs centralised
        central_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
        central_f1   = self._evaluate_central(
            central_data, test_df
        )

        gap = abs(central_f1 - test_results["mean_test_f1"])
        log.info(
            f"\nFederated vs Centralised gap: {gap:.4f} "
            f"({'WITHIN' if gap <= 0.03 else 'EXCEEDS'} 3% target)"
        )

        results = {
            "round_results"  : round_results,
            "n_rounds_run"   : len(round_results),
            "best_round"     : self.best_round,
            "best_f1"        : self.best_f1,
            "privacy_budget" : budget,
            "test_results"   : test_results,
            "central_f1"     : central_f1,
            "fed_central_gap": round(gap, 4),
            "global_weights" : self.global_weights,
        }

        # Save
        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(results, SERVER_PATH)
        log.info(f"FL results saved -> {SERVER_PATH}")

        return results

    def _evaluate_on_test(
        self,
        updates  : list,
        test_df  : pd.DataFrame,
    ) -> dict:
        """Evaluate each client's local model on the shared test set."""
        X_test = test_df[self.features].fillna(0)
        y_test = test_df["label_binary"]

        test_f1s = []
        for u in updates:
            model  = u["model"]
            y_pred = model.predict(X_test)
            f1     = f1_score(y_test, y_pred, zero_division=0)
            test_f1s.append(f1)
            log.info(f"Node {u['node_id']} test F1={f1:.4f}")

        mean_f1 = float(np.mean(test_f1s))
        log.info(f"Mean federated test F1: {mean_f1:.4f}")

        return {
            "per_node_f1" : test_f1s,
            "mean_test_f1": round(mean_f1, 4),
        }

    def _evaluate_central(
        self,
        central_data: dict,
        test_df     : pd.DataFrame,
    ) -> float:
        """Evaluate centralised model on same test set for comparison."""
        feats  = central_data["features"]
        shared = [f for f in feats if f in test_df.columns]
        X_test = test_df[shared].fillna(0)
        y_test = test_df["label_binary"]

        y_pred = central_data["model"].predict(X_test)
        f1     = f1_score(y_test, y_pred, zero_division=0)
        log.info(f"Centralised model test F1: {f1:.4f}")
        return round(f1, 4)

    def print_summary(self, results: dict) -> None:
        """Print a clean training summary."""
        print(f"\n{'='*60}")
        print(f"FEDERATED LEARNING SUMMARY")
        print(f"{'='*60}")
        print(f"Rounds completed    : {results['n_rounds_run']}")
        print(f"Best round          : {results['best_round']}")
        print(f"Best local F1       : {results['best_f1']:.4f}")
        print(f"Mean test F1 (fed)  : {results['test_results']['mean_test_f1']:.4f}")
        print(f"Centralised F1      : {results['central_f1']:.4f}")
        print(f"Fed vs Central gap  : {results['fed_central_gap']:.4f} "
              f"({'OK' if results['fed_central_gap'] <= 0.03 else 'HIGH'})")
        print(f"\nPrivacy Budget:")
        pb = results["privacy_budget"]
        print(f"  Epsilon spent     : {pb['epsilon_spent']:.4f} / {pb['epsilon_budget']}")
        print(f"  Delta             : {pb['delta']:.0e}")
        print(f"  Status            : {pb['status']}")
        print(f"\nPer-round F1:")
        for r in results["round_results"]:
            print(f"  Round {r['round']:>2} | "
                  f"mean_F1={r['mean_local_f1']:.4f} | "
                  f"variance={r['weight_variance']:.4f}")


if __name__ == "__main__":
    lgb_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    features = lgb_data["features"]

    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    server  = FederatedServer(
        features = features,
        n_rounds = FL_ROUNDS,
        apply_dp = True,
    )
    results = server.run_federation(train_df, test_df)
    server.print_summary(results)