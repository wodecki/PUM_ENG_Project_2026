# MP2: Data Cleaning & Feature Engineering

## Scenario

You've explored the MajsterPlus data and identified several quality issues: Polish date formats, currency strings, missing values, outliers, and impossible values. Now you need to **clean and transform** the data into a format suitable for machine learning.

This mini-project covers the **Data Preparation** phase of CRISP-DM — the most labor-intensive phase in any data science project.

## Learning Objectives

By the end of this mini-project, you should be able to:
- Parse non-standard date formats and currency strings
- Identify and handle different types of missing data
- Detect and remove outliers using the IQR method
- Encode categorical variables (binary, ordinal, one-hot)
- Apply feature scaling (StandardScaler) with proper train/test separation

## What You Receive

- `3. notebooks/mp2_starter.ipynb` — starter notebook with the 12-step pipeline
- `2. data/customers.csv` — raw data (or use `2. data/checkpoints/checkpoint_for_mp2.csv`)
- `2. data/data_dictionary.md` — documents all 10 data quality issues

## What You Do

You must follow the **mandatory 12-step preprocessing order**. Changing the order will produce different results.

| Step | Task | Pre-filled? | Estimated Time |
|------|------|-------------|---------------|
| 1-2 | Load data & verify MD5 fingerprint | Yes | 5 min |
| 3 | Separate target, drop customer_id | Yes | 5 min |
| 4 | Parse Polish dates (helper given) | **TODO: apply** | 15 min |
| 5 | Clean total_spend (PLN string → float) | **TODO** | 10 min |
| 6 | Replace impossible satisfaction_score with NaN | **TODO** | 10 min |
| 7 | Impute missing values (median/mode) | **TODO** | 20 min |
| 8 | Remove IQR outliers on avg_basket_value | **Half pre-filled, TODO: filter** | 15 min |
| 9 | Encode categoricals (binary, ordinal, one-hot) | **TODO** | 25 min |
| 10 | Null assertion gate | Yes | 2 min |
| 11 | Train/test split (80/20, stratified) | Yes | 5 min |
| 12 | StandardScaler (fit on train only) | **TODO** | 10 min |
| Bonus | K-Means clustering (optional) | **TODO** | 15 min |
| Save | Checkpoint for MP3 | Yes | 2 min |
| | **Total** | | **~2.5 hours** |

## What You Submit

**10 MCQ answers** via LMS (48-hour window, 3 attempts).

### Before you start the test, you should understand:

- [ ] What the mean vs. median of `total_spend` tells you about the distribution shape
- [ ] Why impossible `satisfaction_score` values must be replaced with NaN *before* median imputation (pipeline ordering)
- [ ] Why filtering `y` and `gender_series` after IQR outlier removal is critical (alignment)
- [ ] The trade-off between `drop_first=False` and `drop_first=True` in one-hot encoding
- [ ] Training set dimensions (rows × features)

## Hints and Common Pitfalls

1. **Follow the step order exactly.** Step 6 (impossible values → NaN) must happen BEFORE Step 7 (imputation). Otherwise, the impossible values get imputed, not replaced.

2. **Polish date helper is provided** — you just need to apply it with `.apply(parse_polish_date)`.

3. **total_spend conversion**: Strip "PLN " first (note the space after PLN), then remove commas, then cast to float.

4. **Satisfaction score**: Valid range is [1.0, 5.0]. Values outside this range (0.0, 7.2, -1.0) are data entry errors. Replace with NaN, then impute along with other missing values.

5. **IQR outlier removal**: The IQR calculation is pre-filled. You need to apply the filter. **Don't forget** to also filter `y` (target) and `gender_series` to keep them aligned.

6. **Encoding order matters**:
   - First: `loyalty_member` → binary (Tak=1, Nie=0)
   - Then: `monthly_income_bracket` → ordinal (A=1, B=2, ..., E=5)
   - Finally: remaining categoricals → `pd.get_dummies(drop_first=False)`
   - **Sort columns alphabetically** after encoding: `df = df[sorted(df.columns)]`

7. **`drop_first=False`** — we keep all dummy columns for pedagogical clarity. This is intentional.

8. **Boolean columns**: After `pd.get_dummies()`, some columns may be boolean dtype. Convert them to int: `df[bool_cols] = df[bool_cols].astype(int)`

9. **StandardScaler**: Fit on training data ONLY, transform both train and test. Fitting on full data = data leakage.

10. **Verify your output**: Run `3. notebooks/scripts/verify_checkpoint.py` to compare against the golden baseline. It will tell you exactly what differs.

## If Your Output Differs

If your numbers don't match the expected values and you can't find the issue:
1. Run `3. notebooks/scripts/verify_checkpoint.py` for specific error messages
2. If still stuck, load the golden checkpoint `2. data/checkpoints/checkpoint_for_mp3.pkl` and move on to MP3

## Reproducibility

- Random seed: 42
- Preprocessing order: Steps 1-12 as specified
- `pd.get_dummies(drop_first=False)` — keep all dummies
- Columns sorted alphabetically after encoding
- StandardScaler fit on train only
