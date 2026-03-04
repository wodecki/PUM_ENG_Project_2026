# MP3: Baseline Modeling & Algorithm Comparison

## Scenario

Your data is clean and ready for modeling. MajsterPlus needs to know: **can we predict which customers will lapse?** You'll train two different algorithms — Logistic Regression (simple, interpretable) and Random Forest (complex, powerful) — and compare their performance.

This mini-project covers the **Modeling** phase of CRISP-DM.

## Learning Objectives

By the end of this mini-project, you should be able to:
- Train classification models with scikit-learn
- Evaluate models using confusion matrix, classification report, and ROC-AUC
- Compare models using ROC curves on the same plot
- Analyze feature importance from tree-based models
- Detect overfitting by comparing train vs. test performance

## What You Receive

- `3. notebooks/mp3_starter.ipynb` — starter notebook with section headers and hints
- Checkpoint from MP2 (`checkpoints/mp2_checkpoint.pkl` or `2. data/checkpoints/checkpoint_for_mp3.pkl`)

**If you didn't complete MP2**, load the golden checkpoint:
```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp3.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## What You Do

| Step | Task | Pre-filled? | Estimated Time |
|------|------|-------------|---------------|
| 1 | Load checkpoint | Yes | 5 min |
| 2 | **TODO**: Train LogisticRegression, evaluate (confusion matrix, report, ROC-AUC) | **TODO** | 30 min |
| 3 | **TODO**: Train RandomForest, evaluate with same metrics | **TODO** | 25 min |
| 4 | **TODO**: Plot ROC curves for both models on same chart | **TODO** | 15 min |
| 5 | **TODO**: Extract and visualize RF feature importances (top 15) | **TODO** | 15 min |
| 6 | **TODO**: Compare train vs. test accuracy (overfitting check) | **TODO** | 10 min |
| 7 | **TODO**: Create comparison summary table | **TODO** | 15 min |
| 8 | Save checkpoint for MP4 | Yes | 5 min |
| | **Total** | | **~2 hours** |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand:

- [ ] How to interpret a confusion matrix in the MajsterPlus business context (TP/FP/FN/TN meaning)
- [ ] What the modest AUC improvement from MP1 baseline (~0.83) to MP3 LR (~0.84) suggests about predictive signal
- [ ] LogisticRegression test accuracy
- [ ] RandomForest recall for class 1 (lapsed)
- [ ] RandomForest's most important feature
- [ ] Train accuracy for both models (overfitting check)

## Hints and Common Pitfalls

1. **Use exact hyperparameters**:
   - `LogisticRegression(random_state=42, max_iter=1000)`
   - `RandomForestClassifier(random_state=42, n_estimators=100)`
   - Different hyperparameters → different results → wrong MCQ answers.

2. **Confusion matrix interpretation**: Rows are actual classes, columns are predicted classes (in sklearn's default). So `confusion_matrix(y_test, y_pred)` gives:
   ```
   [[TN, FP],
    [FN, TP]]
   ```

3. **ROC curves**: Use `RocCurveDisplay.from_predictions()` for clean plotting. Plot both on the same `ax` for comparison.

4. **Feature importance**: `rf.feature_importances_` gives importance for each feature in the order they appear in the training data. Map them to feature names.

5. **Overfitting check**: Random Forest achieving 100% train accuracy is common and expected — it memorizes training data by default. The key question is how much worse it performs on the test set.

6. **Precision vs. Recall**: Don't confuse them:
   - **Precision** = TP / (TP + FP) — "Of those predicted lapsed, how many truly are?"
   - **Recall** = TP / (TP + FN) — "Of those truly lapsed, how many did we catch?"

7. **Save your models** in the checkpoint — you'll need them in MP4 and MP5.

## If Your Output Differs

If your metrics don't match expected values:
1. Verify you loaded the correct checkpoint (MP2 output or golden baseline)
2. Check hyperparameters match exactly (random_state=42, max_iter=1000, n_estimators=100)
3. If still stuck, load `2. data/checkpoints/checkpoint_for_mp4.pkl` to continue to MP4

## Reproducibility

- Random seed: 42 (for all models)
- LogisticRegression: `max_iter=1000`
- RandomForest: `n_estimators=100`
