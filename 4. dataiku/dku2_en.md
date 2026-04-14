Here is the English translation of the provided instructions:

# MP2 in Dataiku: Data Cleaning and Feature Engineering

**Objective:** Processing raw data using a rigorous cleaning process (analyzing dates, currencies, removing impossible and outlier values) and transformation (encoding categorical variables, splitting into training and testing sets).

## Phase 1: Data Cleaning (Prepare Recipe)

Instead of writing 12 blocks of code in the `pandas` library, we will use the **Prepare Recipe** tool, which will build our processing pipeline step-by-step.

1. Go to the **Flow** tab. Click the `customers` dataset.
2. From the right panel, select the **Prepare** icon (the yellow broom).
3. Name the new output dataset `customers_prepared` and click **Create Recipe**.
4. You will find yourself in the script editor (left *Steps* panel). It's time to replicate the logical steps from Python:

### Step 1 & 2: Parsing Polish dates (`registration_date`)

The standard parser doesn't understand Polish abbreviations like "sty", "lut". We need to help it.

- Click the `registration_date` column header -> **Find and Replace**.
- Click **Add replacement**, and then add mappings for the months (e.g., `sty` -> `01`, `lut` -> `02`, `mar` -> `03`, `kwi` -> `04`, `maj` -> `05`, etc.). Important: check the *Match mode: substring* option.
- Once the months are numerical, click the column header again and select **Parse date**. Choose the appropriate format (e.g., `dd-MM-yyyy`).

### Step 3: Currencies as numbers (`total_spend`)

- Click the `total_spend` column -> **Find and Replace** -> Add replacement. Type `PLN ` (remember the space!). Check the *Match mode: substring* option.
- Click **Add a new step**, select the `convert number formats` processor, and configure the format accordingly so that the output is a number in the `123.45` format.

### Step 4: Impossible values (`satisfaction_score`)

The score must be in the range of 1.0 - 5.0.

- Click the `satisfaction_score` header -> **Filter with numerical range**.
- In the step configuration window on the left, restrict the range to be from `1` to `5`.

### Step 5: Missing data imputation (Fill empty cells)

- For a selected numerical column (e.g., `age`, `satisfaction_score`), check its median (`Analyze`).
- Click the **+ Add a New Step** button in the left panel, search for **Fill empty cells with fixed value**, and specify the median value.
- For categorical variables, identify the most popular value and proceed similarly.

### Step 6: Outliers - IQR Method (`avg_basket_value`)

When removing outliers in Python, you have to be careful not to mess up the indices. In Dataiku, we will remove the entire row.

Click *Analyze* on the `avg_basket_value` column > **Actions** > `Clear rows outside of 1.5 IQR`.

### Step 7: Ordinal and Binary Encoding

ML models need numbers, not text.

- For `loyalty_member`: Use **Find and Replace**. Change `Tak` to `1`, `Nie` to `0`. Then make sure the column is set to a numerical type. Note: set `Normalization mode` to *ignore accents*.
- For `monthly_income_bracket`: Use **Find and Replace**. Change `A` to `1`, `B` to `2`, ..., `E` to `5`.

### Step 8: Optional: One-Hot Encoding (Dummy Variables)

In Python, we use the `pd.get_dummies()` function for one-hot encoding. In Dataiku, however, we don't have to do this: the AutoML module will handle it automatically.

------

## Phase 2: Train/Test Split (Split Recipe)

After cleaning the data (step 10 from the notebook), we need to split it into a training and testing set (80/20, Stratified). In Dataiku, we do this with a separate recipe, which perfectly visualizes the concept.

1. In the **Flow** tab, select the created `customers_prepared` dataset.
2. Click the **Split** icon.
3. As outputs (Outputs), add two new datasets: `customers_train` and `customers_test`.
4. Click **Create recipe**.
5. Go into the Split Recipe configuration:
   - **Splitting method:** Select *Randomly Dispatch data on the output datasets*.
   - **WARNING - STRATIFICATION:** This is crucial with such unbalanced classes!
     - In `Dispatch mode`, check the *Random subsets of columns values* option and select `is_lapsed` in the column field.
   - Configure 80% to `customers_train` and 20% to `customers_test`.
   - Set the **Random seed** to `42` (a reproducibility requirement from the notebook!).
6. Click **RUN**.

------

## Phase 3: Feature Scaling (StandardScaler) - Important architectural difference!

In the Python notebook (step 12), you fit the `StandardScaler` to the training set and transform both sets. You must be very careful about so-called **Data Leakage**.

In Dataiku, **WE DO NOT DO THIS** physically on the datasets in the Flow. Feature scaling is treated as an element of the modeling pipeline.

- We leave the values in their original scales in the `customers_train` dataset.
- In the next stage (MP3), when you go to the **Lab -> Visual ML** tab, the Dataiku engine in the *Design -> Features* tab will automatically apply *Standard Rescaling* under the hood, guaranteeing zero data leakage and a perfect application to the test set.