#!/usr/bin/env python3
# /// script
# dependencies = [
#     "pandas>=2.0,<3.0",
#     "numpy>=1.26,<2.0",
#     "scikit-learn>=1.4,<1.6",
# ]
# requires-python = ">=3.10"
# ///
"""Verify student MP2 output against golden baseline checkpoint.

Compares the student's preprocessed DataFrame against the golden checkpoint_for_mp3,
printing specific, actionable error messages for each mismatch.

Usage:
    uv run "3. notebooks/scripts/verify_checkpoint.py" <student_checkpoint.pkl>

    If no argument given, checks the internal checkpoint at checkpoints/mp2_checkpoint.pkl.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GOLDEN_DIR = PROJECT_ROOT / "2. data" / "checkpoints"


class CheckResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed += 1
            print(f"  \u2705 {name}")
        else:
            self.failed += 1
            msg = f"  \u274c {name}"
            if detail:
                msg += f"\n     → {detail}"
            print(msg)

    def summary(self) -> bool:
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"  {self.passed}/{total} checks passed")
        if self.failed > 0:
            print(f"  {self.failed} issues found — see details above")
        else:
            print("  Your checkpoint matches the golden baseline!")
        print(f"{'=' * 60}")
        return self.failed == 0


def main():
    # Determine student checkpoint path
    if len(sys.argv) > 1:
        student_path = Path(sys.argv[1])
    else:
        student_path = PROJECT_ROOT / "checkpoints" / "mp2_checkpoint.pkl"

    golden_path = GOLDEN_DIR / "checkpoint_for_mp3.pkl"

    if not golden_path.exists():
        print(f"ERROR: Golden checkpoint not found at {golden_path}")
        print("Run: uv run scripts/build_checkpoints.py")
        sys.exit(1)

    if not student_path.exists():
        print(f"ERROR: Student checkpoint not found at {student_path}")
        print("Run your MP2 solution notebook first to generate the checkpoint.")
        sys.exit(1)

    print("=" * 60)
    print("Checkpoint Verification: Student vs Golden Baseline")
    print("=" * 60)
    print(f"  Student:  {student_path}")
    print(f"  Golden:   {golden_path}")
    print()

    with open(golden_path, "rb") as f:
        golden = pickle.load(f)
    with open(student_path, "rb") as f:
        student = pickle.load(f)

    v = CheckResult()

    # Check keys
    golden_keys = set(golden.keys())
    student_keys = set(student.keys())
    missing_keys = golden_keys - student_keys
    v.check(
        "Checkpoint contains required keys",
        len(missing_keys) == 0,
        f"Missing keys: {missing_keys}. Your checkpoint should contain: {sorted(golden_keys)}"
    )

    # Check X_train shape
    if "X_train" in student:
        g_shape = golden["X_train"].shape
        s_shape = student["X_train"].shape
        v.check(
            f"X_train shape matches ({g_shape})",
            s_shape == g_shape,
            f"Your X_train has shape {s_shape}, expected {g_shape}. "
            + (f"Row mismatch: check your outlier removal step (IQR on avg_basket_value). "
               if s_shape[0] != g_shape[0] else "")
            + (f"Column mismatch: check your encoding step (one-hot + ordinal). "
               f"Expected {g_shape[1]} features after encoding."
               if len(s_shape) > 1 and len(g_shape) > 1 and s_shape[1] != g_shape[1] else "")
        )

    # Check X_test shape
    if "X_test" in student:
        g_shape = golden["X_test"].shape
        s_shape = student["X_test"].shape
        v.check(
            f"X_test shape matches ({g_shape})",
            s_shape == g_shape,
            f"Your X_test has shape {s_shape}, expected {g_shape}."
        )

    # Check y_train
    if "y_train" in student:
        g_len = len(golden["y_train"])
        s_len = len(student["y_train"])
        v.check(
            f"y_train length matches ({g_len})",
            s_len == g_len,
            f"Your y_train has {s_len} samples, expected {g_len}. "
            "Check that you removed the same outlier rows from both X and y."
        )

        g_rate = golden["y_train"].mean()
        s_rate = student["y_train"].mean()
        v.check(
            f"y_train lapse rate matches ({g_rate:.3f})",
            abs(s_rate - g_rate) < 0.01,
            f"Your y_train lapse rate is {s_rate:.3f}, expected {g_rate:.3f}. "
            "Check stratified split: train_test_split(..., stratify=y)."
        )

    # Check y_test
    if "y_test" in student:
        g_rate = golden["y_test"].mean()
        s_rate = student["y_test"].mean()
        v.check(
            f"y_test lapse rate matches ({g_rate:.3f})",
            abs(s_rate - g_rate) < 0.01,
            f"Your y_test lapse rate is {s_rate:.3f}, expected {g_rate:.3f}."
        )

    # Check feature names
    if "feature_names" in student and "feature_names" in golden:
        g_feats = sorted(golden["feature_names"])
        s_feats = sorted(student["feature_names"])
        v.check(
            f"Feature names match ({len(g_feats)} features)",
            g_feats == s_feats,
            _feature_diff_detail(g_feats, s_feats)
        )

    # Check column order (sorted alphabetically)
    if "X_train" in student and hasattr(student["X_train"], "columns"):
        g_cols = list(golden["X_train"].columns)
        s_cols = list(student["X_train"].columns)
        v.check(
            "Column order matches (alphabetically sorted)",
            g_cols == s_cols,
            f"First difference at position {_first_diff(g_cols, s_cols)}. "
            "Did you sort columns alphabetically? df = df[sorted(df.columns)]"
            if g_cols != s_cols else ""
        )

    # Check null values
    if "X_train" in student and hasattr(student["X_train"], "isnull"):
        s_nulls = student["X_train"].isnull().sum().sum()
        v.check(
            "X_train has no null values",
            s_nulls == 0,
            f"Your X_train has {s_nulls} null values. "
            "Check imputation steps (median for numeric, mode for categorical)."
        )

    # Check scaling (means near 0, stds near 1) — excluding cluster column
    if "X_train" in student and hasattr(student["X_train"], "mean"):
        cols_to_check = [c for c in student["X_train"].columns if c != "cluster"]
        means = student["X_train"][cols_to_check].mean()
        max_mean = means.abs().max()
        v.check(
            "X_train appears scaled (means ≈ 0, excluding cluster)",
            max_mean < 0.1,
            f"Max column mean = {max_mean:.4f}. "
            "Did you apply StandardScaler? Fit on train, transform both train and test."
        )

    # Check numeric values approximate match (first few rows)
    if "X_train" in student and "X_train" in golden:
        try:
            g_vals = golden["X_train"].values[:5, :5].flatten()
            s_vals = student["X_train"].values[:5, :5].flatten()
            if len(g_vals) == len(s_vals):
                max_diff = np.max(np.abs(g_vals - s_vals))
                v.check(
                    "Numeric values approximately match (first 5×5 block)",
                    max_diff < 0.01,
                    f"Max difference = {max_diff:.6f}. "
                    "Values should match if preprocessing steps were done in the correct order."
                )
        except Exception:
            pass

    v.summary()
    sys.exit(0 if v.failed == 0 else 1)


def _feature_diff_detail(golden: list, student: list) -> str:
    g_set = set(golden)
    s_set = set(student)
    missing = g_set - s_set
    extra = s_set - g_set
    parts = []
    if missing:
        parts.append(f"Missing features: {sorted(missing)}")
    if extra:
        parts.append(f"Extra features: {sorted(extra)}")
    if not parts:
        parts.append("Same features but different order")
    return " | ".join(parts)


def _first_diff(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


if __name__ == "__main__":
    main()
