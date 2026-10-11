import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from energy_regression import ENVIRONMENT, load_data, make_features, make_model, metrics, temporal_splits


def sample(rows=120):
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(size=(rows, len(ENVIRONMENT))), columns=ENVIRONMENT)
    frame["date"] = pd.date_range("2016-01-01", periods=rows, freq="10min")
    frame["lights"] = rng.integers(0, 4, rows)
    frame["Appliances"] = 100 + 3 * frame.T1 - 2 * frame.RH_1 + frame.lights
    return frame


class RegressionTests(unittest.TestCase):
    def test_temporal_folds_never_use_final_test_or_future(self):
        folds = list(temporal_splits(19735))
        self.assertEqual(len(folds), 4)
        for train, valid in folds:
            self.assertLess(train[-1], valid[0])
            self.assertLess(valid[-1], int(19735 * .8))
        self.assertEqual(folds[-1][1][-1] + 1, int(19735 * .8))

    def test_features_do_not_read_target_and_hour_wraps(self):
        frame = sample()
        first = make_features(frame)
        frame["Appliances"] = -99999
        pd.testing.assert_frame_equal(first, make_features(frame))
        frame.loc[0, "date"] = pd.Timestamp("2016-01-01 23:50")
        frame.loc[1, "date"] = pd.Timestamp("2016-01-02 00:00")
        cyclic = make_features(frame)[["hour_sin", "hour_cos"]]
        self.assertLess(np.linalg.norm(cyclic.iloc[0] - cyclic.iloc[1]), .05)

    def test_pca_and_scaling_fit_only_on_training(self):
        frame = sample()
        x = make_features(frame)
        model = make_model("pcr", components=3).fit(x.iloc[:80], frame.Appliances.iloc[:80])
        scaler = model.named_steps["features"].named_transformers_["environment"].named_steps["scale"]
        np.testing.assert_allclose(scaler.mean_, x.iloc[:80][ENVIRONMENT].mean())
        original = scaler.mean_.copy()
        x.loc[80:, ENVIRONMENT] = 10000
        model.predict(x.iloc[80:])
        np.testing.assert_array_equal(scaler.mean_, original)

    def test_ols_recovers_known_linear_relation_and_full_pca_matches(self):
        frame = sample(300)
        x = make_features(frame)
        a = make_model("ols").fit(x.iloc[:200], frame.Appliances.iloc[:200])
        b = make_model("pcr", components=24).fit(x.iloc[:200], frame.Appliances.iloc[:200])
        pa, pb = a.predict(x.iloc[200:]), b.predict(x.iloc[200:])
        np.testing.assert_allclose(pa, frame.Appliances.iloc[200:], atol=1e-8)
        np.testing.assert_allclose(pa, pb, atol=1e-8)

    def test_invalid_workbooks_fail_instead_of_silent_cleaning(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.xlsx"
            frame = sample()
            frame.to_excel(path, index=False)
            self.assertEqual(len(load_data(path)), len(frame))
            frame.loc[1, "T1"] = np.nan
            frame.to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, "Missing or infinite"):
                load_data(path)
            frame = sample()
            frame.loc[1, "date"] = frame.loc[0, "date"]
            frame.to_excel(path, index=False)
            with self.assertRaisesRegex(ValueError, "duplicate timestamps"):
                load_data(path)

    def test_metric_units_and_negative_r2(self):
        result = metrics([0, 2], [2, 0])
        self.assertEqual(result["rmse"], 2)
        self.assertEqual(result["mae"], 2)
        self.assertEqual(result["r2"], -3)


if __name__ == "__main__":
    unittest.main()
