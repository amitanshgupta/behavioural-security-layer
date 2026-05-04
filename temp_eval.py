import pkg_resources

packages = [
    "pandas", "numpy", "scikit-learn", "lightgbm",
    "torch", "shap", "lime", "joblib", "scipy",
    "matplotlib", "imbalanced-learn", "nbformat",
    "flower", "opacus",
]

lines = []
for pkg in packages:
    try:
        version = pkg_resources.get_distribution(pkg).version
        lines.append(f"{pkg}=={version}")
    except Exception:
        lines.append(f"# {pkg} — not installed")

for line in lines:
    print(line)