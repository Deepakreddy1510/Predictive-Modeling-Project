# Appliance consumption regression: results

The selected model is **pcr_3_with_lights**, chosen using four expanding chronological validation folds.
The final 20% of observations was withheld from fitting and selection. This is contemporaneous
estimation, not advance forecasting or a causal analysis. Units have not been independently verified.

## Data and evaluation

- 19,735 regular 10-minute observations; no missing required measurements.
- Final refit: 2016-01-11 17:00:00 through 2016-04-30 08:10:00 (15,788 rows).
- Test: 2016-04-30 08:20:00 through 2016-05-27 18:00:00 (3,947 rows).
- Calendar variables: cyclic hour and weekday; no raw timestamp or response lags.
- rv1/rv2 excluded: identical columns with undocumented predictive meaning.
- Standardization and PCA were fitted within each training fold, then refitted on development data.
- Both versions with and without contemporaneous lighting consumption were compared in validation.

## Chronological validation

| model | rmse_mean | rmse_std | mae_mean | r2_mean |
| --- | --- | --- | --- | --- |
| pcr_3_with_lights | 95.1402 | 10.2592 | 51.7641 | 0.1061 |
| pcr_6_with_lights | 95.3544 | 10.6315 | 51.8457 | 0.1026 |
| pcr_12_with_lights | 95.3588 | 11.4382 | 51.3813 | 0.1041 |
| pcr_3_without_lights | 95.4139 | 9.7088 | 52.1288 | 0.0998 |
| calendar | 95.4264 | 9.5043 | 52.5763 | 0.0991 |
| pcr_6_without_lights | 95.5049 | 10.1157 | 51.9184 | 0.0988 |
| pcr_12_without_lights | 95.7699 | 10.9280 | 51.2650 | 0.0952 |
| pcr_9_with_lights | 96.2984 | 11.4343 | 51.5388 | 0.0862 |
| pcr_9_without_lights | 96.6860 | 10.8942 | 52.0390 | 0.0777 |
| pcr_18_without_lights | 97.1576 | 15.5145 | 58.4154 | 0.0685 |
| pcr_18_with_lights | 97.1885 | 15.6843 | 58.4723 | 0.0683 |
| ols_with_lights | 97.5582 | 14.4137 | 58.1373 | 0.0618 |
| pcr_24_with_lights | 97.5582 | 14.4137 | 58.1373 | 0.0618 |
| ols_without_lights | 97.8159 | 14.4889 | 58.5045 | 0.0567 |
| pcr_24_without_lights | 97.8159 | 14.4889 | 58.5045 | 0.0567 |
| mean | 100.9656 | 11.8946 | 58.5250 | -0.0047 |

## Final test

| model | rmse | mae | r2 |
| --- | --- | --- | --- |
| pcr_3_with_lights | 86.9379 | 51.8907 | 0.0880 |
| mean | 91.0506 | 52.6761 | -0.0003 |
| calendar | 86.5136 | 47.9482 | 0.0969 |

The selected model's RMSE improvement over the training-mean baseline is 4.52%.
All metrics are in the original target scale. Only the selected regression model and the two
predefined baselines were evaluated on the final test period. No clipping was applied:
0 predictions are negative.

The calendar-only baseline has lower test RMSE (86.5136) than the selected sensor model (86.9379). The added sensor/lighting information has not demonstrated a test-period advantage; the model was not retuned after observing this result.
The small validation differences between leading candidates should not be interpreted as decisive. The test R² of 0.0880 indicates limited explanatory performance.

## High-consumption performance

High consumption means at least 210.1935, the development-period 90th percentile.
This threshold was not estimated on the test data.

| group | rows | rmse | mae | r2 |
| --- | --- | --- | --- | --- |
| high_consumption | 323 | 262.0979 | 224.9910 | -3.0275 |
| other_consumption | 3624 | 45.9257 | 36.4626 | -1.4733 |

## Interpretation and limitations

PCA removes correlated directions; it does not guarantee better predictions and remains unsupervised.
Selection was driven by validation RMSE, not explained variance. The full 24-component PCR candidate
is an OLS-equivalent reference, subject to numerical precision. Calendar and lighting are outside PCA.
Coefficients for standardized variables are changes per training standard deviation; PCA coefficients
refer to component scores rather than individual sensors. These are conditional associations, not causal effects.

Temporal drift and unmeasured appliance-use behaviour can limit accuracy. Inspect the test time plot,
residual-versus-fitted plot, weekly metrics, and high-consumption errors rather than relying on one score.
Residual lag-1 correlation on the refitted training period is
0.7087. Classical independent-error significance claims
are not justified. Cook's distance and standardized residuals are exploratory diagnostics only;
no observations were deleted using these measures. OLS predictions can be negative.

Dataset semantics, measurement units, household coverage, and target provenance require external
documentation. This run does not demonstrate generalization to other households or seasons.
The log-target model and next-interval forecasting extension have not been implemented or evaluated.

## Artifacts

See `validation_folds.csv`, `test_predictions.csv`, `test_weekly_metrics.csv`, `coefficients.csv`,
`influential_training_rows.csv`, `metadata.json`, and the diagnostic plots. `model.joblib` contains
the fitted pipeline, whose save/load prediction consistency was verified.
