"""Contemporaneous appliance-consumption regression with temporal validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".mpl-cache"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ENVIRONMENT = [item for i in range(1, 10) for item in (f"T{i}", f"RH_{i}")] + [
    "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"
]
CALENDAR = ["hour_sin", "hour_cos", "day_of_week"]


def load_data(path: Path) -> pd.DataFrame:
    """Validate rather than silently impute, aggregate, or delete observations."""
    frame = pd.read_excel(path, engine="openpyxl")
    required = ["date", "Appliances", "lights", *ENVIRONMENT]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    frame = frame[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame["date"].isna().any() or frame["date"].duplicated().any():
        raise ValueError("Missing or duplicate timestamps; resolve them before training.")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Rows must be in chronological order.")
    numeric = frame.drop(columns="date").apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Missing or infinite measurements; an explicit cleaning policy is required.")
    if (numeric["Appliances"] < 0).any():
        raise ValueError("Negative target measurements need investigation.")
    if len(frame) < 100:
        raise ValueError("At least 100 observations are required for temporal validation.")
    if not frame["date"].diff().dropna().eq(pd.Timedelta(minutes=10)).all():
        raise ValueError("Expected regular 10-minute observations; investigate gaps first.")
    frame[numeric.columns] = numeric
    return frame


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Deterministic timestamp encoding; no fitted statistics or target access."""
    features = frame[[*ENVIRONMENT, "lights"]].copy()
    dates = pd.to_datetime(frame["date"])
    hours = dates.dt.hour + dates.dt.minute / 60
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["day_of_week"] = dates.dt.dayofweek
    return features


def make_model(kind: str, *, components: int | None = None, lights: bool = True) -> Pipeline:
    if kind not in {"mean", "calendar", "ols", "pcr"}:
        raise ValueError(f"Unknown model kind: {kind}")
    blocks = [("hour", "passthrough", ["hour_sin", "hour_cos"]),
              ("weekday", OneHotEncoder(categories=[np.arange(7)], drop="first",
                                         sparse_output=False), ["day_of_week"])]
    if kind in {"ols", "pcr"}:
        environmental_steps = [("scale", StandardScaler())]
        if kind == "pcr":
            if components is None or not 1 <= components <= len(ENVIRONMENT):
                raise ValueError("PCR requires a valid component count.")
            environmental_steps.append(("pca", PCA(n_components=components, svd_solver="full")))
        blocks.append(("environment", Pipeline(environmental_steps), ENVIRONMENT))
        if lights:
            blocks.append(("lights", StandardScaler(), ["lights"]))
    return Pipeline([
        ("features", ColumnTransformer(blocks, remainder="drop", verbose_feature_names_out=True)),
        ("regression", DummyRegressor(strategy="mean") if kind == "mean" else LinearRegression()),
    ])


def temporal_splits(n: int):
    """Four expanding folds: 0–40→40–50%, ... 0–70→70–80%; final 20% withheld."""
    boundaries = [int(n * p / 10) for p in range(4, 9)]
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        yield np.arange(start), np.arange(start, stop)


def metrics(actual, predicted) -> dict:
    return {"rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "mae": float(mean_absolute_error(actual, predicted)),
            "r2": float(r2_score(actual, predicted))}


def candidates() -> list[dict]:
    options = [{"name": "mean", "kind": "mean"}, {"name": "calendar", "kind": "calendar"}]
    for lights in [False, True]:
        suffix = "with_lights" if lights else "without_lights"
        options.append({"name": f"ols_{suffix}", "kind": "ols", "lights": lights})
        for k in [3, 6, 9, 12, 18, 24]:
            options.append({"name": f"pcr_{k}_{suffix}", "kind": "pcr", "components": k,
                            "lights": lights})
    return options


def build(spec: dict) -> Pipeline:
    return make_model(spec["kind"], components=spec.get("components"), lights=spec.get("lights", True))


def validate(features, target, dates) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    records = []
    for spec in candidates():
        print(f"Validating {spec['name']}", flush=True)
        for fold, (train, valid) in enumerate(temporal_splits(len(features)), start=1):
            model = build(spec)
            model.fit(features.iloc[train], target.iloc[train])
            result = metrics(target.iloc[valid], model.predict(features.iloc[valid]))
            records.append({"model": spec["name"], "fold": fold,
                            "train_rows": len(train), "validation_rows": len(valid),
                            "train_end": str(dates.iloc[train[-1]]),
                            "validation_start": str(dates.iloc[valid[0]]),
                            "validation_end": str(dates.iloc[valid[-1]]), **result})
    folds = pd.DataFrame(records)
    summary = folds.groupby("model").agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                                         mae_mean=("mae", "mean"), r2_mean=("r2", "mean"))
    summary = summary.sort_values("rmse_mean").reset_index()
    # Only OLS/PCR candidates are eligible; baselines remain explicit reference models.
    eligible = summary[summary.model.str.startswith(("ols_", "pcr_"))]
    best = eligible.iloc[0].model
    return folds, summary, next(spec for spec in candidates() if spec["name"] == best)


def save_diagnostics(model, x_train, y_train, train_dates, output: Path):
    """Classical influence measures are exploratory, not iid significance tests."""
    design = np.column_stack([np.ones(len(x_train)), model.named_steps["features"].transform(x_train)])
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tolerance = max(design.shape) * np.finfo(float).eps * singular[0]
    rank = int((singular > tolerance).sum())
    leverage = np.square(u[:, :rank]).sum(axis=1)
    residual = y_train.to_numpy() - model.predict(x_train)
    mse = np.square(residual).sum() / (len(residual) - rank)
    denominator = np.maximum(1 - leverage, np.finfo(float).eps)
    standardized = residual / np.sqrt(max(mse, np.finfo(float).eps) * denominator)
    cook = standardized**2 * leverage / (rank * denominator)
    details = pd.DataFrame({"date": train_dates.to_numpy(), "residual": residual,
                            "leverage": leverage, "standardized_residual": standardized,
                            "cooks_distance": cook})
    details.nlargest(20, "cooks_distance").to_csv(output / "influential_training_rows.csv", index=False)
    pd.DataFrame({"feature": ["intercept", *model.named_steps["features"].get_feature_names_out()],
                  "coefficient": [model.named_steps["regression"].intercept_,
                                  *model.named_steps["regression"].coef_]}).to_csv(
                                      output / "coefficients.csv", index=False)
    pca_info = None
    environment = model.named_steps["features"].named_transformers_["environment"]
    if "pca" in environment.named_steps:
        pca = environment.named_steps["pca"]
        pca_info = pca.explained_variance_ratio_.tolist()
        pd.DataFrame(pca.components_.T, index=ENVIRONMENT,
                     columns=[f"PC{i+1}" for i in range(pca.n_components_)]).to_csv(output / "pca_loadings.csv")
    return {"design_rank": rank, "design_columns": design.shape[1],
            "condition_number": float(singular[0] / singular[rank - 1]),
            "training_residual_lag1_correlation": float(pd.Series(residual).autocorr()),
            "pca_explained_variance_ratio": pca_info}


def plot_results(predictions: pd.DataFrame, summary: pd.DataFrame, output: Path):
    dates = pd.to_datetime(predictions.date)
    actual, predicted = predictions.actual, predictions.predicted
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # A week of raw data avoids hiding peaks in an aggregate.
    mask = dates < dates.iloc[0] + pd.Timedelta(days=7)
    axes[0, 0].plot(dates[mask], actual[mask], label="Actual", linewidth=.8)
    axes[0, 0].plot(dates[mask], predicted[mask], label="Predicted", linewidth=.8)
    axes[0, 0].set_title("First test week"); axes[0, 0].legend()
    axes[0, 0].tick_params(axis="x", rotation=25)
    axes[0, 1].scatter(actual, predicted, s=5, alpha=.2)
    limits = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    axes[0, 1].plot(limits, limits, color="black", linestyle="--")
    axes[0, 1].set(xlabel="Actual consumption", ylabel="Predicted consumption", title="Test predictions")
    residual = actual - predicted
    axes[1, 0].scatter(predicted, residual, s=5, alpha=.2)
    axes[1, 0].axhline(0, color="black", linestyle="--")
    axes[1, 0].set(xlabel="Predicted consumption", ylabel="Actual − predicted", title="Test residuals")
    axes[1, 1].plot(range(1, 145), [residual.autocorr(lag) for lag in range(1, 145)])
    axes[1, 1].set(xlabel="Lag (10-minute intervals)", ylabel="Residual correlation",
                   title="Residual temporal dependence (descriptive)")
    fig.tight_layout(); fig.savefig(output / "test_diagnostics.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 7))
    ordered = summary.sort_values("rmse_mean", ascending=False)
    ax.barh(ordered.model, ordered.rmse_mean, xerr=ordered.rmse_std)
    ax.set(xlabel="Mean validation RMSE (error bars: fold SD)", title="Chronological model comparison")
    fig.tight_layout(); fig.savefig(output / "validation_comparison.png", dpi=160); plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(frame.columns.astype(str)) + " |"
    rule = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
                               for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def run(data_path: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    frame = load_data(data_path)
    features, target, dates = make_features(frame), frame.Appliances, frame.date
    folds, summary, selected = validate(features, target, dates)
    folds.to_csv(output / "validation_folds.csv", index=False)
    summary.to_csv(output / "validation_summary.csv", index=False)
    cutoff = int(len(frame) * .8)
    x_train, x_test = features.iloc[:cutoff], features.iloc[cutoff:]
    y_train, y_test = target.iloc[:cutoff], target.iloc[cutoff:]
    model = build(selected).fit(x_train, y_train)
    predicted = model.predict(x_test)
    test_records = [{"model": selected["name"], **metrics(y_test, predicted)}]
    for kind in ["mean", "calendar"]:
        baseline = make_model(kind).fit(x_train, y_train)
        test_records.append({"model": kind, **metrics(y_test, baseline.predict(x_test))})
    test = pd.DataFrame(test_records)
    test.to_csv(output / "test_metrics.csv", index=False)
    predictions = pd.DataFrame({"date": dates.iloc[cutoff:].to_numpy(), "actual": y_test.to_numpy(),
                                "predicted": predicted})
    predictions["residual"] = predictions.actual - predictions.predicted
    predictions.to_csv(output / "test_predictions.csv", index=False)
    threshold = float(y_train.quantile(.9))
    groups = []
    for name, mask in {"high_consumption": y_test >= threshold,
                       "other_consumption": y_test < threshold}.items():
        if mask.any():
            groups.append({"group": name, "rows": int(mask.sum()),
                           **metrics(y_test[mask], predicted[mask.to_numpy()])})
    group_results = pd.DataFrame(groups)
    group_results.to_csv(output / "test_consumption_groups.csv", index=False)
    weekly = pd.DataFrame([
        {"week_start": str(start), "rows": len(block), **metrics(block.actual, block.predicted)}
        for start, block in predictions.set_index("date").resample("7D") if len(block)
    ])
    weekly.to_csv(output / "test_weekly_metrics.csv", index=False)
    diagnostics = save_diagnostics(model, x_train, y_train, dates.iloc[:cutoff], output)
    plot_results(predictions, summary, output)
    artifact = {"model": model, "spec": selected, "environment_columns": ENVIRONMENT,
                "calendar_columns": CALENDAR, "task": "contemporaneous estimation"}
    joblib.dump(artifact, output / "model.joblib")
    # Verify that the saved artifact actually reproduces predictions.
    restored = joblib.load(output / "model.joblib")
    np.testing.assert_allclose(restored["model"].predict(x_test), predicted, rtol=1e-12, atol=1e-12)
    import sklearn
    metadata = {"source_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
                "rows": len(frame), "training_rows": cutoff, "test_rows": len(y_test),
                "train_start": str(dates.iloc[0]), "train_end": str(dates.iloc[cutoff-1]),
                "test_start": str(dates.iloc[cutoff]), "test_end": str(dates.iloc[-1]),
                "selected_model": selected, "selection_rule": "Lowest mean validation RMSE among OLS/PCR",
                "high_consumption_threshold": threshold,
                "negative_test_predictions": int((predicted < 0).sum()),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "diagnostics": diagnostics}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    improvement = 100 * (1 - test.iloc[0].rmse / test[test.model == "mean"].iloc[0].rmse)
    calendar_rmse = float(test[test.model == "calendar"].iloc[0].rmse)
    calendar_comparison = (
        "The calendar-only baseline has lower test RMSE than the selected sensor model. "
        "The added sensor/lighting information has not demonstrated a test-period advantage; "
        "the model was not retuned after observing this result."
        if calendar_rmse < test.iloc[0].rmse else
        "The selected sensor model has lower test RMSE than the calendar-only baseline. "
        "This is a descriptive comparison on one recording, not evidence of statistical significance."
    )
    report = f"""# Appliance consumption regression: results

The selected model is **{selected['name']}**, chosen using four expanding chronological validation folds.
The final 20% of observations was withheld from fitting and selection. This is contemporaneous
estimation, not advance forecasting or a causal analysis. Units have not been independently verified.

## Data and evaluation

- {len(frame):,} regular 10-minute observations; no missing required measurements.
- Final refit: {dates.iloc[0]} through {dates.iloc[cutoff-1]} ({cutoff:,} rows).
- Test: {dates.iloc[cutoff]} through {dates.iloc[-1]} ({len(y_test):,} rows).
- Calendar variables: cyclic hour and weekday; no raw timestamp or response lags.
- rv1/rv2 excluded: identical columns with undocumented predictive meaning.
- Standardization and PCA were fitted within each training fold, then refitted on development data.
- Both versions with and without contemporaneous lighting consumption were compared in validation.

## Chronological validation

{markdown_table(summary)}

## Final test

{markdown_table(test)}

The selected model's RMSE improvement over the training-mean baseline is {improvement:.2f}%.
All metrics are in the original target scale. Only the selected regression model and the two
predefined baselines were evaluated on the final test period. No clipping was applied:
{metadata['negative_test_predictions']} predictions are negative.

{calendar_comparison}
The small validation differences between leading candidates should not be interpreted as decisive.
The test R² of {test.iloc[0].r2:.4f} indicates limited explanatory performance; the final test data
was not used to tune a replacement model.

## High-consumption performance

High consumption means at least {threshold:.4f}, the development-period 90th percentile.
This threshold was not estimated on the test data.

{markdown_table(group_results)}

## Interpretation and limitations

PCA removes correlated directions; it does not guarantee better predictions and remains unsupervised.
Selection was driven by validation RMSE, not explained variance. The full 24-component PCR candidate
is an OLS-equivalent reference, subject to numerical precision. Calendar and lighting are outside PCA.
Coefficients for standardized variables are changes per training standard deviation; PCA coefficients
refer to component scores rather than individual sensors. These are conditional associations, not causal effects.

Temporal drift and unmeasured appliance-use behaviour can limit accuracy. Inspect the test time plot,
residual-versus-fitted plot, weekly metrics, and high-consumption errors rather than relying on one score.
Residual lag-1 correlation on the refitted training period is
{diagnostics['training_residual_lag1_correlation']:.4f}. Classical independent-error significance claims
are not justified. Cook's distance and standardized residuals are exploratory diagnostics only;
no observations were deleted using these measures. OLS predictions can be negative.

Dataset semantics, measurement units, household coverage, and target provenance require external
documentation. This run does not demonstrate generalization to other households or seasons.
The log-target model and next-interval forecasting extension have not been implemented or evaluated.

## Artifacts

See `validation_folds.csv`, `test_predictions.csv`, `test_weekly_metrics.csv`, `coefficients.csv`,
`influential_training_rows.csv`, `metadata.json`, and the diagnostic plots. `model.joblib` contains
the fitted pipeline, whose save/load prediction consistency was verified.
"""
    (output / "report.md").write_text(report)
    print("\nSelected:", selected["name"])
    print(test.to_string(index=False))
    print(f"Report: {output / 'report.md'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the original Excel workbook")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.data, args.output)


if __name__ == "__main__":
    main()
