# MP1: Business Context & Data Exploration

## Scenario

MajsterPlus has provided you with two datasets containing customer demographics, purchasing behavior, and transaction records. Before building any models, you need to **understand the data**: its structure, distributions, quality issues, and whether there's enough signal to predict customer lapse.

This mini-project covers the **Business Understanding** and **Data Understanding** phases of CRISP-DM.

## Learning Objectives

By the end of this mini-project, you should be able to:
- Load and verify dataset integrity using MD5 fingerprints
- Identify column types, missing value patterns, and data quality issues
- Visualize distributions and correlations to understand feature relationships
- Train a simple baseline model to assess learnability

## What You Receive

- `3. notebooks/mp1_starter.ipynb` — starter notebook with pre-filled cells and `# YOUR CODE HERE` placeholders
- `2. data/customers.csv` — 5,000 customer records (21 columns)
- `2. data/transactions.csv` — ~25,000 transaction records (8 columns)
- `2. data/data_dictionary.md` — complete column documentation

## What You Do

| Step | Task | Estimated Time |
|------|------|---------------|
| 1 | Run setup cells (reproducibility locks, imports, data loading) | 5 min |
| 2 | Examine shape, dtypes, and first rows — already pre-filled | 10 min |
| 3 | **TODO**: Analyze missing values across all columns | 15 min |
| 4 | Review target variable distribution (pre-filled) | 5 min |
| 5 | **TODO**: Create correlation heatmap of numeric features | 20 min |
| 6 | **TODO**: Create relationship plots (boxplots, histograms) | 25 min |
| 7 | Review sneak-peek baseline model (pre-filled) | 10 min |
| 8 | **TODO**: Write 3-5 key observations | 20 min |
| | **Total** | **~2 hours** |

## What You Submit

**10 MCQ answers** via Edux (48-hour window, 3 attempts).

### Before you start the test, you should understand:

- [ ] The relationship between `customers.csv` (5,000 rows) and `transactions.csv` (~25,000 rows) — what the ratio implies
- [ ] Why high accuracy can be misleading on imbalanced data (~19.5% lapse rate)
- [ ] Which columns have missing values, which has the most, and why missingness patterns matter
- [ ] What extreme outliers in `avg_basket_value` mean and how they affect modeling
- [ ] The baseline model's ROC-AUC score (~0.83) and what it tells you about learnability

## Hints and Common Pitfalls

1. **Read `2. data/data_dictionary.md` first.** It documents all 21 columns, their types, ranges, and intentional data quality issues.

2. **Don't try to fix data quality issues in MP1.** This mini-project is about *discovering* them. Fixing happens in MP2.

3. **The `total_spend` column is a string**, not a number. It contains values like "PLN 1,234.50". You'll handle this in MP2.

4. **`registration_date` uses Polish month abbreviations** (sty, lut, mar, ...). Standard `pd.to_datetime()` won't parse them.

5. **The baseline model excludes `days_since_last_purchase`** on purpose — it's quasi-deterministic with the target (lapse is defined as > 90 days since last purchase). Including it would inflate metrics.

6. **Correlation heatmap**: Look for both strong positive AND negative correlations. Several pairs have |r| > 0.5.

## Reproducibility

- Random seed: 42 (set in the first cell)
- Do NOT modify the reproducibility locks or data loading cells
- Your results should match the solution notebook exactly
