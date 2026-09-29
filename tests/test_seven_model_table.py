"""Check that the seven-model table never fills incompatible historical cells."""

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "seven_model_table", ROOT / "scripts" / "build_seven_model_table.py")
table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table)


class SevenModelTableTest(unittest.TestCase):
    def test_template_preserves_old_values_and_leaves_new_models_pending(self):
        rows = table.rows_template()
        by_id = {row["model_id"]: row for row in rows}
        self.assertEqual(by_id["unet"]["macro_dice"], .7935)
        self.assertEqual(by_id["afbms"]["lesion_f1_26conn_micro"], .5385)
        self.assertAlmostEqual(by_id["resunet"]["small_lesion_matched_dice"],
                               .3684612447499045)
        self.assertIsNone(by_id["attention"]["small_lesion_matched_dice"])
        self.assertIsNone(by_id["doubleblock"]["et_dice"])

    def test_pooled_f1_matches_two_historical_rows(self):
        # Formal ET summaries: ResUNet TP=49 FP=46 FN=44;
        # AFBMS-ResUNet TP=49 FP=40 FN=44.
        self.assertAlmostEqual(table.pooled_lesion_f1(49, 46, 44), .5212765957)
        self.assertAlmostEqual(table.pooled_lesion_f1(49, 40, 44), .5384615385)
        self.assertEqual(table.format_value("N/A (0 matched)",
                                            "small_lesion_matched_dice"),
                         "N/A (0 matched)")

    def test_mismatched_old_anchor_is_rejected(self):
        rows = table.rows_template()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            table.put_matched(rows, "unet", .5, .2, "wrong cohort")
        self.assertIsNone(rows[0]["small_lesion_matched_dice"])

    def test_legacy_summary_needs_identical_case_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = root / "summary.csv"
            with summary.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=(
                    "model_id", "stratum", "seed", "split", "test_cases",
                    "gt_lesions", "matched_lesion_dice",
                    "gt_anchored_lesion_dice"))
                writer.writeheader()
                writer.writerow({"model_id": "unet", "stratum": "small",
                                 "seed": 55, "split": "test", "test_cases": 37,
                                 "gt_lesions": 31, "matched_lesion_dice": .4,
                                 "gt_anchored_lesion_dice": .0687})
            with (root / "test_cases.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("split", "case_id"))
                writer.writeheader()
                writer.writerow({"split": "test", "case_id": "wrong"})
            with self.assertRaisesRegex(ValueError, "test cases/order differ"):
                table.read_legacy_summary(summary, ["expected"], table.rows_template())

    def test_training_must_be_finished_before_test_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "train_log.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=("epoch", "bad_epochs"))
                writer.writeheader()
                writer.writerow({"epoch": 47, "bad_epochs": 10})
            protocol = {"epochs": 200, "early_stopping_patience": 25}
            with self.assertRaisesRegex(RuntimeError, "training not complete"):
                table.require_finished_training(root, protocol)
            with (root / "train_log.csv").open("a", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerow((62, 25))
            table.require_finished_training(root, protocol)


if __name__ == "__main__":
    unittest.main()
