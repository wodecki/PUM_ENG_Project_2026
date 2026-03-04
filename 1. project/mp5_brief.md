# MP5: Model Comparison & Final Recommendation

## Scenario

You've built and evaluated two models. Now MajsterPlus asks for your **final recommendation**: Which model should we deploy? At what threshold? What about fairness across customer demographics? The VP of Marketing needs a written recommendation she can present to the board.

This mini-project covers the **Evaluation** phase of CRISP-DM — synthesis, comparison, and recommendation.

## Learning Objectives

By the end of this mini-project, you should be able to:
- Train and evaluate additional algorithms (GradientBoosting, VotingClassifier)
- Compare models across multiple criteria (statistics, profit, fairness, interpretability)
- Assess model fairness by analyzing performance across demographic subgroups
- Evaluate model interpretability (coefficients vs. feature importances)
- Write a structured, evidence-based business recommendation

## What You Receive

- `3. notebooks/mp5_starter.ipynb` — minimal scaffolding, mostly section headers
- Checkpoint from MP4 (`checkpoints/mp4_checkpoint.pkl` or `2. data/checkpoints/checkpoint_for_mp5.pkl`)

**If you didn't complete MP4**, load the golden checkpoint:
```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp5.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## What You Do

| Step | Task | Pre-filled? | Estimated Time |
|------|------|-------------|---------------|
| 1 | Load checkpoint | Yes | 5 min |
| 2-3 | Reuse LR + RF models from checkpoint | Yes | 5 min |
| 4 | **TODO**: Train GradientBoostingClassifier | **TODO** | 15 min |
| 5 | **TODO**: Create VotingClassifier (optional) | **TODO** | 15 min |
| 6 | **TODO**: Multi-criteria comparison table | **TODO** | 20 min |
| 7 | **TODO**: Business profit comparison (all models, optimal thresholds) | **TODO** | 20 min |
| 8 | **TODO**: Fairness analysis (recall/precision by gender) | **TODO** | 20 min |
| 9 | **TODO**: Interpretability assessment | **TODO** | 15 min |
| 10 | **TODO**: Write final recommendation | **TODO** | 20 min |
| | **Total** | | **~2.5 hours** |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand:

- [ ] Why the model with the highest ROC-AUC does not necessarily produce the highest business profit
- [ ] What additional criteria (beyond AUC) should guide model selection when AUCs are near-identical
- [ ] VotingClassifier ROC-AUC and which model has the highest ROC-AUC overall
- [ ] Recall gap between gender groups for GradientBoosting
- [ ] Which model you would recommend for deployment and why (profit, interpretability, fairness)

## Hints and Common Pitfalls

1. **GradientBoosting**: Use `GradientBoostingClassifier(random_state=42)` with default hyperparameters. Different random_state = different results.

2. **VotingClassifier**: Use `voting="soft"` to average probabilities (not hard votes). Include LR, RF, and GB as estimators.

3. **Best ROC-AUC ≠ best business model.** This is the key insight of this MP. The model with the highest AUC may not generate the most profit when the cost matrix is applied. Compare at each model's optimal threshold.

4. **Fairness analysis**: The `gender_test` series is in the checkpoint. Use it to split predictions by gender (M/K) and compute recall and precision for each subgroup. A gap > 0.05 between groups warrants discussion.

5. **Interpretability ranking**:
   - Logistic Regression: most interpretable (direct coefficients with sign and magnitude)
   - GradientBoosting/RandomForest: feature importances show *what* matters but not *how*
   - VotingClassifier: least interpretable (combines multiple models)

6. **Final recommendation structure**:
   - Which model and why
   - What threshold
   - Expected financial impact
   - Caveats and limitations
   - Suggested next steps

7. **The "simpler model wins" pattern**: Don't be surprised if LogisticRegression generates the highest profit. With tight margins (60 PLN per TP vs. 80 PLN per FP), a conservative, precise model can outperform a complex one.

8. **Optional PyCaret section** is ungraded. Only explore it if you have time and interest.

## If Your Output Differs

If your metrics don't match:
1. Check `GradientBoostingClassifier(random_state=42)` — default hyperparameters
2. Verify VotingClassifier uses `voting="soft"` with exactly [lr, rf, gb] as estimators
3. Ensure you're using the same CAMPAIGN_COST (80) and EXPECTED_REVENUE from the checkpoint

## Reproducibility

- Random seed: 42 (for GradientBoosting)
- VotingClassifier: soft voting, estimators = [LR, RF, GB]
- Same cost matrix as MP4 (campaign cost = 80 PLN, revenue from checkpoint)
