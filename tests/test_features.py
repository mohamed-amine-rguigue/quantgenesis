import os
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.features import build_features, FEATURE_COLS, LABEL_COL


class BuildFeaturesTest(unittest.TestCase):
    def make_sample_frame(self):
        dates = pd.date_range("2020-01-01", periods=80, freq="D")
        close = np.linspace(100, 120, len(dates))
        return pd.DataFrame({
            "date": dates,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1000, 2000, len(dates)),
        })

    def test_build_features_returns_expected_columns_and_labels(self):
        df = self.make_sample_frame()
        out = build_features(df)

        self.assertIn(LABEL_COL, out.columns)
        self.assertTrue(all(col in out.columns for col in FEATURE_COLS))
        self.assertTrue(out.empty or out.shape[0] > 0)


if __name__ == "__main__":
    unittest.main()
