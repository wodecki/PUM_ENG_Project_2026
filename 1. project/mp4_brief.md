# MP4: Model Evaluation & Business Impact

## Scenario

You have trained two models. But MajsterPlus doesn't care about ROC-AUC — they care about **money**. The marketing VP asks: "If we run a reactivation campaign targeting your model's predictions, how much profit will we make? And should we contact everyone, or just the high-risk customers?"

This mini-project covers the **Evaluation** phase of CRISP-DM — translating statistical metrics into business value.

## Learning Objectives

By the end of this mini-project, you should be able to:
- Construct a cost matrix that maps predictions to financial outcomes
- Calculate profit per record at a given classification threshold
- Compare model-based targeting against naive strategies (contact everyone / nobody)
- Optimize the classification threshold for maximum profit
- Interpret cumulative gains (lift) curves

## What You Receive

- `3. notebooks/mp4_starter.ipynb` — starter notebook with business context and section headers
- Checkpoint from MP3 (`checkpoints/mp3_checkpoint.pkl` or `2. data/checkpoints/checkpoint_for_mp4.pkl`)

**If you didn't complete MP3**, load the golden checkpoint:
```python
import pickle
with open("../2. data/checkpoints/checkpoint_for_mp4.pkl", "rb") as f:
    checkpoint = pickle.load(f)
```

## Business Parameters

| Parameter | Value |
|-----------|-------|
| Campaign cost per customer | **80 PLN** (voucher + operational costs) |
| Voucher value | 50 PLN |
| Expected revenue per reactivation | **Median total_spend of lapsed customers in the test set** |

**Cost matrix:**

| | Actually Lapsed | Actually Active |
|---|---|---|
| **Contacted** (predicted lapsed) | Revenue − 80 PLN (TP) | −80 PLN (FP) |
| **Not contacted** (predicted active) | 0 PLN (FN) | 0 PLN (TN) |

## What You Do

| Step | Task | Pre-filled? | Estimated Time |
|------|------|-------------|---------------|
| 1 | Load checkpoint | Yes | 5 min |
| 2 | Set up business parameters (cost, revenue) | Partially | 10 min |
| 3 | **TODO**: Define compute_profit() function | **TODO** | 15 min |
| 4 | **TODO**: Calculate profit at threshold=0.5 for both models | **TODO** | 15 min |
| 5 | **TODO**: Calculate baseline profits (contact everyone / nobody) | **TODO** | 10 min |
| 6 | **TODO**: Threshold optimization (sweep 0.05–0.95) | **TODO** | 25 min |
| 7 | **TODO**: Identify optimal threshold | **TODO** | 10 min |
| 8 | **TODO**: Create lift (cumulative gains) curve | **TODO** | 20 min |
| 9 | **TODO**: Estimate annual profit for full customer base | **TODO** | 10 min |
| 10 | Save checkpoint for MP5 | Yes | 5 min |
| | **Total** | | **~2 hours** |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand:

- [ ] The expected revenue per reactivation (median lapsed test spend)
- [ ] Profit from "contact everyone" strategy on the test set
- [ ] LogisticRegression profit at threshold 0.5
- [ ] Why a threshold below 0.5 can be optimal — the relationship between TP gain, FP cost, and profit
- [ ] Lift at the 20% contact level (RF cumulative gains curve)

## Hints and Common Pitfalls

1. **Expected revenue**: Use the **median** total_spend of lapsed customers in the test set (not all customers, not mean). You need to load the raw data to get the original total_spend values (the checkpoint has scaled values).

2. **The "contact everyone" strategy loses money.** This is realistic — when the expected revenue per customer is only slightly above the campaign cost, contacting the 80% of active customers is very expensive.

3. **Threshold = 0.5 is NOT optimal.** The whole point of this MP is to find a better threshold. Plot profit vs. threshold and find the peak.

4. **Understand threshold mechanics.** The MCQs test your understanding of why the optimal threshold differs from 0.5 and the profit trade-off at threshold 0.5.

5. **Lift curve**: Sort customers by predicted probability (highest first), then plot cumulative % of lapsed customers captured vs. % of customers contacted. The closer the curve is to the upper-left corner, the better the model.

6. **Annual profit extrapolation**: The test set is ~20% of the total customer base. Scale proportionally: `annual = test_profit / test_fraction`.

7. **Negative profit is possible** and realistic. If false positives (80 PLN cost each) outnumber the true positives, the campaign loses money at that threshold.

## If Your Output Differs

If your profit calculations don't match:
1. Verify expected revenue = median of lapsed test customers' total_spend (not mean, not all customers)
2. Check your compute_profit function: TP × (revenue - cost) + FP × (-cost)
3. If still stuck, load `2. data/checkpoints/checkpoint_for_mp5.pkl` to continue to MP5

## Reproducibility

- Campaign cost: 80 PLN (fixed)
- Expected revenue: median of lapsed test customers' total_spend (deterministic)
- Thresholds: np.arange(0.05, 1.0, 0.05) — 19 steps
