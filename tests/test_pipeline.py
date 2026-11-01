import hashlib
import json
import os
import sys
import unittest
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parents[1].joinpath("scripts")))
import ml_pipeline


class DataPreparationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            feature: np.arange(12, dtype=float) + index
            for index, feature in enumerate(ml_pipeline.FEATURES)
        })
        self.frame[ml_pipeline.TARGET] = [0, 1] * 6

    def test_removes_duplicates_and_non_feature_columns(self):
        raw = pd.concat([self.frame, self.frame.iloc[[0]]], ignore_index=True)
        raw["Engine_ID"] = "example"
        cleaned = ml_pipeline.prepare_frame(raw)
        self.assertEqual(len(cleaned), 12)
        self.assertEqual(list(cleaned.columns), ml_pipeline.FEATURES + [ml_pipeline.TARGET])

    def test_rejects_missing_sensor_column(self):
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            ml_pipeline.prepare_frame(self.frame.drop(columns=["Fuel pressure"]))

    def test_rejects_missing_and_non_finite_values(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                invalid = self.frame.copy()
                invalid.loc[0, "Coolant temp"] = value
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    ml_pipeline.prepare_frame(invalid)

    def test_rejects_negative_rpm_and_pressure(self):
        for feature in ml_pipeline.FEATURES[:4]:
            with self.subTest(feature=feature):
                invalid = self.frame.copy()
                invalid.loc[0, feature] = -1
                with self.assertRaisesRegex(ValueError, "must not be negative"):
                    ml_pipeline.prepare_frame(invalid)

    def test_preserves_zero_rpm_and_temperature_outliers(self):
        self.frame.loc[0, "Engine rpm"] = 0
        self.frame.loc[0, "lub oil temp"] = -10
        self.frame.loc[0, "Coolant temp"] = 195.5
        cleaned = ml_pipeline.prepare_frame(self.frame)
        self.assertEqual(cleaned.loc[0, "Engine rpm"], 0)
        self.assertEqual(cleaned.loc[0, "lub oil temp"], -10)
        self.assertEqual(cleaned.loc[0, "Coolant temp"], 195.5)

    def test_rejects_wrong_target_labels_and_insufficient_classes(self):
        invalid = self.frame.copy()
        invalid[ml_pipeline.TARGET] = 2
        with self.assertRaisesRegex(ValueError, "class labels"):
            ml_pipeline.prepare_frame(invalid)
        with self.assertRaisesRegex(ValueError, "five distinct records"):
            ml_pipeline.prepare_frame(self.frame.iloc[:6])


class DeploymentTests(unittest.TestCase):
    def test_train_and_test_partitions_are_disjoint_and_complete(self):
        train = pd.read_csv(ml_pipeline.ARTIFACTS.joinpath("train.csv"))
        test = pd.read_csv(ml_pipeline.ARTIFACTS.joinpath("test.csv"))
        prepared = pd.read_csv(ml_pipeline.ARTIFACTS.joinpath("prepared.csv"))
        self.assertEqual(len(train) + len(test), len(prepared))
        self.assertTrue(set(map(tuple, train.to_numpy())).isdisjoint(set(map(tuple, test.to_numpy()))))
        self.assertEqual(set(train[ml_pipeline.TARGET]), {0, 1})
        self.assertEqual(set(test[ml_pipeline.TARGET]), {0, 1})

    def test_model_and_test_metrics_are_valid(self):
        model = joblib.load(ml_pipeline.MODEL_PATH)
        self.assertEqual(list(model.feature_names_in_), ml_pipeline.FEATURES)
        self.assertEqual(set(model.classes_), {0, 1})
        report = json.loads(ml_pipeline.ROOT.joinpath("reports", "metrics.json").read_text(encoding="utf-8"))
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertTrue(0 <= report["test_metrics"][metric] <= 1)
        self.assertEqual(
            report["data_profile"]["source_sha256"],
            hashlib.sha256(ml_pipeline.DATA_PATH.read_bytes()).hexdigest(),
        )

    def test_generated_app_and_dependencies_match_the_validated_sources(self):
        self.assertEqual(
            ml_pipeline.ROOT.joinpath("streamlit_app.py").read_text(encoding="utf-8"),
            ml_pipeline.notebook_app_source(),
        )
        requirements = ml_pipeline.ROOT.joinpath("requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(requirements, [f"{name}=={version(name)}" for name in ml_pipeline.PACKAGES])

    def test_streamlit_collects_dataframe_and_predicts_with_saved_model(self):
        sample = pd.read_csv(ml_pipeline.ARTIFACTS.joinpath("test.csv")).iloc[0]
        model = joblib.load(ml_pipeline.MODEL_PATH)
        app = AppTest.from_file(str(ml_pipeline.ROOT.joinpath("streamlit_app.py")), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(len(app.number_input), 6)
        for feature in ml_pipeline.FEATURES:
            app.number_input(feature).set_value(float(sample[feature]))
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        expected = pd.DataFrame(
            [{feature: float(sample[feature]) for feature in ml_pipeline.FEATURES}],
            columns=ml_pipeline.FEATURES,
        )
        pd.testing.assert_frame_equal(app.session_state["input_df"], expected)
        self.assertEqual(app.session_state["predicted_condition"], str(model.predict(expected)[0]))
        np.testing.assert_allclose(
            app.session_state["class_probabilities"]["Probability"], model.predict_proba(expected)[0]
        )
        self.assertEqual(len(app.get("download_button")), 1)
        pd.testing.assert_frame_equal(
            pd.read_csv(StringIO(app.session_state["input_df"].to_csv(index=False))), expected
        )

    def test_replacing_the_model_invalidates_the_streamlit_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            app_path = root.joinpath("streamlit_app.py")
            app_path.write_text(ml_pipeline.notebook_app_source(), encoding="utf-8")
            model_path = root.joinpath("models", "best_engine_model.pkl")
            model_path.parent.mkdir()
            sample = pd.DataFrame(
                [{feature: 1.0 for feature in ml_pipeline.FEATURES}],
                columns=ml_pipeline.FEATURES,
            )
            joblib.dump(DecisionTreeClassifier(random_state=42).fit(sample, [0]), model_path)
            original_time = model_path.stat().st_mtime_ns
            app = AppTest.from_file(str(app_path), default_timeout=30).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["predicted_condition"], "0")
            joblib.dump(DecisionTreeClassifier(random_state=42).fit(sample, [1]), model_path)
            new_time = original_time + 1_000_000_000
            os.utime(model_path, ns=(new_time, new_time))
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["predicted_condition"], "1")


if __name__ == "__main__":
    unittest.main()
