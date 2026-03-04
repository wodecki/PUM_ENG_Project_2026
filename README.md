# PUM: Designing Services Using AI Methods

## What Is This?

You are a **data analyst** at **MajsterPlus**, a Polish DIY retail chain with 12 stores. About 18% of customers have lapsed (no purchase in 90+ days). Your mission: build a machine learning model to predict at-risk customers and recommend a reactivation campaign.

You'll work through **5 mini-projects (MP1-MP5)** following the CRISP-DM methodology — from raw data exploration to a final business recommendation.

## Folder Structure

```
├── 1. project/
│   ├── 1. business_scenario.md    # The MajsterPlus story
│   ├── 2. project_overview.md     # Mini-project overview and assessment model
│   ├── 3. environment_setup.md    # Colab / local setup guide
│   └── mp1_brief.md — mp5_brief.md  # Step-by-step brief for each MP
├── 2. data/
│   ├── customers.csv              # 5,000 customers, 21 columns
│   ├── transactions.csv           # ~25,000 purchase records
│   ├── data_dictionary.md         # Column descriptions and known data quality issues
│   └── checkpoints/               # Golden baselines (see below)
├── 3. notebooks/
│   ├── mp1_starter.ipynb — mp5_starter.ipynb  # Your working notebooks
│   └── scripts/
│       └── verify_checkpoint.py   # Verify your MP2 output against the golden baseline
```

## Getting Started

1. Read `1. project/1. business_scenario.md` for context
2. Set up your environment following `1. project/3. environment_setup.md` (Google Colab recommended)
3. Open `3. notebooks/mp1_starter.ipynb` and follow the instructions
4. Before each MP, read the corresponding brief (`1. project/mpN_brief.md`) and consult `2. data/data_dictionary.md`

## Mini-Project Progression

| MP | Title | What You Do |
|----|-------|-------------|
| MP1 | Business Context & Data Exploration | Load data, explore distributions, find quality issues, train a baseline |
| MP2 | Data Cleaning & Feature Engineering | Parse dates, clean strings, handle missing values, remove outliers, encode, scale |
| MP3 | Baseline Modeling & Algorithm Comparison | Train Logistic Regression and Random Forest, evaluate with ROC curves |
| MP4 | Model Evaluation & Business Impact | Build cost matrix, calculate campaign profit, optimize threshold |
| MP5 | Model Comparison & Final Recommendation | Compare models on profit, fairness, interpretability; write recommendation |

Each MP builds on the previous one. Save your checkpoint at the end of each MP — the next notebook loads it.

## Falling Behind?

Golden baseline checkpoints in `2. data/checkpoints/` let you start any MP without completing the previous ones. See `2. data/checkpoints/README.md` for loading instructions.

## Reproducibility

All students must get the **same numerical results** (MCQ answers depend on exact values):
- Use `random_state=42` everywhere
- Use scikit-learn 1.4-1.5, pandas 2.x, numpy <2.0
- Follow the exact preprocessing order in the MP2 brief

If your MP2 numbers don't match, run `3. notebooks/scripts/verify_checkpoint.py` to find out why.

## Assessment

5 mini-projects x 10 MCQ questions = **50 questions total**, delivered via Edux with a 48-hour window and 3 attempts per test. See `1. project/2. project_overview.md` for details.
