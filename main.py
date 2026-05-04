import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from utils.logger import get_logger
from utils.constants import PROCESSED_DIR, METADATA_DIR

log = get_logger("main")


def run_phase1():
    """Phase 1 — Data ingestion, cleaning, feature engineering."""
    log.info("=" * 60)
    log.info("PHASE 1 — Data Pipeline")
    log.info("=" * 60)

    from ingestion.load_nslkdd import load_nslkdd
    from preprocessing.cleaner import clean
    from preprocessing.validator import validate
    from feature_engineering.feature_pipeline import run_feature_pipeline

    for split in ("train", "test"):
        log.info(f"\n--- {split.upper()} ---")
        df      = load_nslkdd(split)
        df      = clean(df, split)
        valid   = validate(df, split)
        if not valid:
            log.error(f"Validation failed for {split} — stopping")
            return False
        run_feature_pipeline(df, split=split)

    log.info("Phase 1 complete")
    return True


def run_phase2():
    """Phase 2 — Context Engine."""
    log.info("=" * 60)
    log.info("PHASE 2 — Context Engine")
    log.info("=" * 60)

    import joblib
    from context_engine.clustering import run_clustering, assign_clusters_to_test
    from context_engine.baseline_model import run_baseline_model
    from context_engine.drift_detection import run_drift_detection
    from context_engine.context_adjuster import run_context_adjustment
    from utils.constants import CLUSTERING_FEATURES

    # Clustering
    train_df, profiles = run_clustering(optimize_k=False)
    log.info(f"Clustering done — {len(profiles)} clusters")

    # Baselines
    scaler   = joblib.load(METADATA_DIR / "cluster_scaler.pkl")
    km       = joblib.load(METADATA_DIR / "kmeans_model.pkl")
    X        = train_df[[f for f in CLUSTERING_FEATURES
                         if f in train_df.columns]]
    train_df["cluster"] = km.predict(scaler.transform(X))

    static, rolling, thresholds = run_baseline_model(train_df)

    # Drift + context adjustment
    train_df, _ = run_drift_detection(train_df)
    train_df    = run_context_adjustment(train_df)

    out = PROCESSED_DIR / "nslkdd_train_context.csv"
    train_df.to_csv(out, index=False)
    log.info(f"Context-annotated train saved -> {out}")
    log.info("Phase 2 complete")
    return True


def run_phase3():
    """Phase 3 — Model Training."""
    log.info("=" * 60)
    log.info("PHASE 3 — Detection Models")
    log.info("=" * 60)

    from models.isolation_forest import run_isolation_forest
    from models.lightgbm_model import run_lightgbm
    from models.genai_detector import run_genai_detector
    from models.model_pipeline import run_pipeline

    log.info("Training Isolation Forest...")
    run_isolation_forest()

    log.info("Training LightGBM...")
    run_lightgbm()

    log.info("Training GenAI Detector...")
    run_genai_detector()

    log.info("Running ensemble pipeline...")
    run_pipeline()

    log.info("Phase 3 complete")
    return True


def run_phase4():
    """Phase 4 — Explainability."""
    log.info("=" * 60)
    log.info("PHASE 4 — Explainability")
    log.info("=" * 60)

    from explainability.shap_explainer import run_shap_explainer
    from explainability.alert_generator import run_alert_generator

    run_shap_explainer(sample_n=1000)
    run_alert_generator(n_alerts=10)

    log.info("Phase 4 complete")
    return True


def run_phase5():
    """Phase 5 — Federated Learning."""
    log.info("=" * 60)
    log.info("PHASE 5 — Federated Learning")
    log.info("=" * 60)

    import joblib
    import pandas as pd
    from federated.server import FederatedServer

    lgb_data = joblib.load(METADATA_DIR / "lightgbm_model.pkl")
    features = lgb_data["features"]
    train_df = pd.read_csv(PROCESSED_DIR / "nslkdd_train_features.csv")
    test_df  = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")

    server  = FederatedServer(features=features, n_rounds=10, apply_dp=True)
    results = server.run_federation(train_df, test_df)
    server.print_summary(results)

    log.info("Phase 5 complete")
    return True


def run_phase6():
    """Phase 6 — Evaluation."""
    log.info("=" * 60)
    log.info("PHASE 6 — Evaluation")
    log.info("=" * 60)

    import pandas as pd
    from evaluation.metrics import evaluate_all_models, save_metrics_report
    from evaluation.experiments import run_all_experiments
    from evaluation.ablation import run_ablation

    test_df    = pd.read_csv(PROCESSED_DIR / "nslkdd_test_features.csv")
    comparison = evaluate_all_models(test_df)
    save_metrics_report(comparison)

    run_all_experiments()
    run_ablation()

    log.info("Phase 6 complete")
    return True


def run_cert_pipeline():
    """CERT data cleaning + feature engineering."""
    log.info("=" * 60)
    log.info("CERT — Data Pipeline")
    log.info("=" * 60)

    from preprocessing.cert_cleaner import run_cert_cleaner
    from feature_engineering.cert_features import run_cert_feature_pipeline

    run_cert_cleaner(save=True)
    run_cert_feature_pipeline(save=True)

    log.info("CERT pipeline complete")
    return True


def run_cert_models_phase():
    """Train models on CERT + unified network data."""
    log.info("=" * 60)
    log.info("CERT — Model Training")
    log.info("=" * 60)

    from models.cert_model import run_cert_models
    from models.genai_detector_cert import run_genai_cert
    from models.network_ids_model import run_network_ids
    from models.combined_model import run_combined_model

    run_cert_models()
    run_genai_cert()
    run_network_ids()
    run_combined_model()

    log.info("CERT model training complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral Security Layer — Full Pipeline"
    )
    parser.add_argument(
        "--phase",
        type    = int,
        choices = [1, 2, 3, 4, 5, 6, 7, 8],
        default = None,
        help    = "Run a specific phase (1-8). Omit to run all."
    )
    parser.add_argument(
        "--from-phase",
        type    = int,
        choices = [1, 2, 3, 4, 5, 6, 7, 8],
        default = 1,
        help    = "Start from a specific phase and run to end."
    )
    args = parser.parse_args()

    phases = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
    7: run_cert_pipeline,
    8: run_cert_models_phase,
}

    if args.phase:
        log.info(f"Running Phase {args.phase} only")
        phases[args.phase]()
    else:
        start = args.from_phase
        log.info(f"Running phases {start} -> 8")
        for phase_num in range(start, 9):
            success = phases[phase_num]()
            if not success:
                log.error(f"Phase {phase_num} failed — stopping")
                break

    log.info("\nPipeline finished.")


if __name__ == "__main__":
    main()