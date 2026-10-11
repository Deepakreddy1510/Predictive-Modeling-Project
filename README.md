# Appliance energy regression

A course project comparing ordinary least squares (OLS) with principal component regression
(standardized environmental measurements → PCA → OLS). The response is `Appliances` in the
uploaded `energydata_appliances_continuous.xlsx` workbook. This estimates consumption from
measurements available in the same interval; it is **not advance forecasting**.

## Run

Open [appliance_energy_regression.ipynb](appliance_energy_regression.ipynb) in Jupyter or Google Colab
to run the complete workflow interactively. The supplied workbook is included under `data/`.
For a notebook downloaded on its own, download the workbook too and place it beside the notebook.

Python 3.12 was used. From this repository:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python energy_regression.py --data data/energydata_appliances_continuous.xlsx --output outputs
```

Use the included workbook under `data/`, or pass another compatible workbook path directly.
Dependencies are already available in the current workspace, so `python` can also run these commands.
No network access, credentials, or external services are needed for training.

## Experiment

- Required columns: timestamp, response, lighting, and 24 temperature/humidity/weather measurements.
- Validate timestamps, sampling intervals, numeric values, and target sign. Do not silently impute or remove spikes.
- Ignore `rv1`/`rv2`: they are identical in the supplied workbook and their predictive meaning is undocumented.
- Encode hour with sine/cosine and weekday with six dummy variables. Do not use raw timestamps or response lags.
- Compare mean and calendar baselines, OLS with/without lighting, and PCR with 3, 6, 9, 12, 18, or 24 components.
- Fit standardization/PCA within each training fold. Keep calendar and lighting outside PCA.
- Four expanding folds: train on the first 40%, validate on 40–50%; then 0–50% → 50–60%,
  0–60% → 60–70%, and 0–70% → 70–80%. The final 20% is never used for selection.
- Select the OLS/PCR candidate with minimum mean validation RMSE. Refit on the first 80%.
- Evaluate only that selected model and predefined baselines on the final 20%.
- Report RMSE, MAE, and R², plus weekly and high-consumption errors. No clipping or automatic outlier removal.

OLS minimizes squared error on the original target scale. PCA truncation trades information for
potentially more stable estimates; component count is selected by predictive error, not variance explained.
The full 24-component candidate is a coordinate-change reference and should match OLS predictions.

The course PDF covers OLS on pages 7–17, leverage/influence diagnostics on pages 1–6, and PCA
on pages 40–53. PCR combines these course methods. Chronological validation and metrics are evaluation
choices, not claims about additional algorithms taught in the notes.

## Outputs

`outputs/report.md` is the generated results report. CSV files contain fold results, final predictions,
coefficients, weekly errors, and influential training rows. PNG figures compare candidates and diagnose
test errors. `metadata.json` records the dataset hash, split dates, selected configuration, and versions.
`model.joblib` contains the fitted preprocessing/regression pipeline. Only load trusted joblib files.

## Result on the supplied workbook

Validation selected three environmental principal components, with calendar and lighting features.
On 3,947 held-out readings, RMSE was **86.94**, MAE **51.89**, and R² **0.088**. The calendar-only
baseline achieved RMSE **86.51** and MAE **47.95**, slightly better than the selected sensor model.
Thus the environmental features did not demonstrate an advantage in the final period. High-consumption
spikes were poorly predicted. The model was not retuned on the test set. See `RESULTS.md` for the full report.

To predict on another compatible workbook:

```python
from pathlib import Path
import joblib
from energy_regression import load_data, make_features

artifact = joblib.load("outputs/model.joblib")
new_data = load_data(Path("data/energydata_appliances_continuous.xlsx"))
predictions = artifact["model"].predict(make_features(new_data))
```

This example uses the strict experiment loader, which requires the target column even though
`make_features` and the fitted pipeline do not use it for prediction. For unlabelled observations,
construct a DataFrame with the documented input columns and pass it through `make_features`.

## Limits

Sensor locations, physical units, household coverage, and target provenance need external documentation.
Repeated readings are not independent observations. Correlated or heteroscedastic errors invalidate
ordinary iid inference; coefficient p-values are intentionally not reported. Cook's distance and
standardized residuals are exploratory flags, not reasons to delete observations. Associations are not
causal effects, and performance over this recording does not establish transfer to other households.
No log-target model, forecast extension, or nonlinear feature expansion is claimed as evaluated.
