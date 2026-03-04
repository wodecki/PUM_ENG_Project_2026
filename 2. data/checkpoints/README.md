# Golden Baseline Checkpoints

These checkpoints let you start any mini-project without completing the previous ones.
If you fall behind or want to verify your work, load the checkpoint for your current MP.

## Available Checkpoints

| Checkpoint | Format | Start At | What It Contains |
|-----------|--------|----------|-----------------|
| `checkpoint_for_mp2.csv` | CSV | MP2 | Raw `customers.csv` — skip MP1 (EDA) |
| `checkpoint_for_mp3.csv` / `.pkl` | CSV + PKL | MP3 | Clean, encoded, scaled train/test split from MP2 |
| `checkpoint_for_mp4.pkl` | PKL | MP4 | Trained LogReg + RF models and predictions from MP3 |
| `checkpoint_for_mp5.pkl` | PKL | MP5 | All models + evaluation artifacts + business metrics from MP4 |

## How to Load

### CSV checkpoint (MP2)

```python
import pandas as pd
customers = pd.read_csv("2. data/checkpoints/checkpoint_for_mp2.csv")
```

### PKL checkpoints (MP3–MP5)

```python
import pickle
with open("2. data/checkpoints/checkpoint_for_mp3.pkl", "rb") as f:
    checkpoint = pickle.load(f)

X_train = checkpoint["X_train"]
X_test = checkpoint["X_test"]
y_train = checkpoint["y_train"]
y_test = checkpoint["y_test"]
feature_names = checkpoint["feature_names"]
```

### PKL contents (MP4)

```python
checkpoint.keys()
# ['X_train', 'X_test', 'y_train', 'y_test', 'feature_names',
#  'gender_test', 'lr_model', 'rf_model',
#  'y_pred_lr', 'y_prob_lr', 'y_pred_rf', 'y_prob_rf']
```

### PKL contents (MP5)

```python
checkpoint.keys()
# ['X_train', 'X_test', 'y_train', 'y_test', 'feature_names',
#  'gender_test', 'lr_model', 'rf_model',
#  'y_prob_lr', 'y_prob_rf',
#  'CAMPAIGN_COST', 'EXPECTED_REVENUE',
#  'optimal_threshold_rf', 'optimal_profit_rf']
```

## CSV checkpoint_for_mp3 format

The CSV version includes train and test data combined, with two extra columns:
- `__target__`: the `is_lapsed` label (0 or 1)
- `__split__`: "train" or "test"

```python
import pandas as pd
df = pd.read_csv("2. data/checkpoints/checkpoint_for_mp3.csv", index_col=0)
train = df[df["__split__"] == "train"].drop(columns=["__target__", "__split__"])
test = df[df["__split__"] == "test"].drop(columns=["__target__", "__split__"])
y_train = df[df["__split__"] == "train"]["__target__"]
y_test = df[df["__split__"] == "test"]["__target__"]
```

## Verification

Run `3. notebooks/scripts/verify_checkpoint.py` to compare your MP2 output against the golden baseline.
It provides specific, actionable error messages if your output differs.

## Metadata

See `checkpoint_summary.json` for MD5 hashes, row counts, and other metadata.

## Reproducibility

All checkpoints were generated with:
- `random_state=42` (universal seed)
- scikit-learn 1.4.x–1.5.x, pandas 2.x, numpy <2.0
- Exact 12-step preprocessing order from MP2
