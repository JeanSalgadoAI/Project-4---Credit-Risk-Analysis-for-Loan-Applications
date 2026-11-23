"""Verifica separação, ausência de vazamento e cálculo de referências."""
import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
from main import ROOT, prepare, split_indices, preprocessor, classify_metrics, regress_metrics, forecast_stores, ppc_summary

class EvaluationTests(unittest.TestCase):
    def test_train_only_preprocessing(self):
        train = pd.DataFrame({"x": [0., 2.], "category": ["a", "b"]})
        test = pd.DataFrame({"x": [1000.], "category": ["unseen"]})
        prep = preprocessor(train, list(train), ["category"])
        prep.fit(train)
        transformed = prep.transform(test)
        self.assertEqual(prep.named_transformers_["numeric"].named_steps["scale"].mean_[0], 1.)
        self.assertEqual(transformed.shape[0], 1)
        self.assertNotIn("unseen", prep.named_transformers_["category"].named_steps["encode"].categories_[0])

    def test_target_leak_rejected(self):
        df = pd.DataFrame({"target": [0, 1]})
        with self.assertRaises(ValueError):
            prepare(df, {"target": "target", "features": ["target"]})
        with self.assertRaises(ValueError):
            prepare(df, {"target": "y", "features": ["target"], "forbidden_features": ["target"]})

    def test_project_split(self):
        cfg = json.loads((ROOT / "project.json").read_text())
        df = pd.read_csv(ROOT / cfg["data"])
        frame, features = prepare(df, cfg)
        if cfg["mode"] in ("classification", "text", "regression"):
            train, test = split_indices(frame, cfg)
            self.assertFalse(set(train) & set(test))
            self.assertEqual(len(train) + len(test), len(frame))
            if cfg.get("split") == "group":
                self.assertFalse(set(frame.iloc[train][cfg["group"]]) & set(frame.iloc[test][cfg["group"]]))
            if cfg.get("split") == "time":
                self.assertLess(pd.to_datetime(frame.iloc[train][cfg["date"]]).max(), pd.to_datetime(frame.iloc[test][cfg["date"]]).min())
            if cfg.get("derive_target"):
                self.assertNotIn(cfg["derive_target"]["source"], features)

    def test_known_metrics(self):
        m = classify_metrics([0, 0, 1, 1], [0, 0, 0, 0], [.5]*4)
        self.assertEqual(m["balanced_accuracy"], .5)
        self.assertEqual(m["recall"], 0.)
        self.assertEqual(m["average_precision"], .5)
        self.assertEqual(regress_metrics([1,3],[2,2])["mae"], 1.)

    def test_fixed_origin_does_not_use_future_sales(self):
        import matplotlib.pyplot as plt
        frame = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=10), "Store": [1]*10, "Sales": range(10)})
        _, first, fig = forecast_stores(frame, {})
        plt.close(fig)
        frame.loc[8:, "Sales"] = 9999
        _, second, fig = forecast_stores(frame, {})
        plt.close(fig)
        for col in ["prediction_last", "prediction_mean4", "prediction_mean"]:
            np.testing.assert_array_equal(first[col], second[col])

    def test_published_metrics_match_predictions(self):
        result = json.loads((ROOT / "reports/metrics.json").read_text())
        detail = pd.read_csv(ROOT / "reports/results.csv")
        if result["task"] == "regression":
            self.assertAlmostEqual(result["model_metrics"]["mae"], regress_metrics(detail.actual, detail.prediction)["mae"])
        elif result["task"] == "classification":
            got = classify_metrics(detail.actual, detail.prediction, detail.probability_1)
            for key in ["accuracy", "f1", "roc_auc", "average_precision"]:
                self.assertAlmostEqual(result["model_metrics"][key], got[key])

if __name__ == "__main__":
    unittest.main()
