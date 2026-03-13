# MP1 in Dataiku: Business Context and Data Exploration (No Coding)

**Goal:** Load the MajsterPlus data into Dataiku, understand its structure, identify data quality issues, visualize distributions, and create a visual baseline model.

## Step 1: Creating a project and uploading data

Instead of writing Python code to load CSV files, we will use Dataiku's visual interface.

1. Log in to Dataiku and on the homepage click **+ New Project** -> **Blank Project**. Name it something like `MajsterPlus_Churn`.
2. Open the project and go to the **Flow** tab.
3. Click **+ Import Your First Dataset** (or select `+ Dataset` -> `Upload your files` from the top menu).
4. Drag and drop the `customers.csv` file.
5. In the preview window, make sure the data displays correctly (Dataiku usually detects the `,` separator automatically). Name the dataset `customers_raw` and click **Create**.
6. Repeat steps 3-5 for the `transactions.csv` file (name it `transactions_raw`).

## Step 2: Data exploration and issue identification (Data Understanding)

Instead of using functions like `df.info()` or `df.describe()` in Python, Dataiku has a powerful **Explore** tab.

1. Open the `customers_raw` dataset. Pay attention to the quality bar below the column names (green = ok, red = invalid, grey = missing data).
2. **Analyzing missing values and types:** Click the header of any column (e.g., `monthly_income_bracket`) and select **Analyze** from the drop-down menu. A window will open where Dataiku summarizes the value distribution, the number of missing values (Empty/Null), and unique values.
3. **Identify data quality issues (Your task):** Review the columns and find the "catches" prepared in the data dictionary:
   - Look at the `total_spend` column. You'll see that Dataiku detected it as text (String) because it contains values like `"PLN 1,496.76"`.
   - Look at the dates (`registration_date`). You'll see they are treated as text due to the Polish month abbreviations (e.g., "21-kwi-2022").
   - *(Do not fix them now! Cleaning is the task for the MP2 stage, where you will use the "Prepare recipe" tool).*

## Step 3: Visualizing distributions and class balance

In this step, we will use two different Dataiku tools: quick column analysis and the Charts tab.

**1. Class balance and variable distributions (Analyze tool):**

Instead of building charts from scratch, we will use the built-in data profiling.

- While in the data table view (the **Explore** tab), find the `is_lapsed` column.
- Click on its header (or the small arrow next to the name) and select **Analyze** from the drop-down menu.
- A summary window will open. You will see an exact histogram showing how many records have the value `0` (active) and how many have `1` (churn). This perfectly illustrates the problem of imbalanced classes (~18% churn).
- Repeat the same action (click **Analyze**) for the `age` and `satisfaction_score` columns. Dataiku will automatically generate histograms showing the customers' age and the distribution of their scores.

**2. Detecting outliers (Charts tab):**

To detect anomalies in the average basket value, we need to build a boxplot.

- Go to the **Charts** tab (in the upper left corner).
- Click the chart selection icon. From the menu, go to the **Others** section (at the very bottom) and select **Boxplot**.
- Drag the `avg_basket_value` column to the **Y** field.
- The chart will generate a box with "whiskers." You will see individual dots escaping far upwards beyond the main distribution—these are the extreme outliers that you will deal with in MP2.

## Step 4: Dependencies and correlations (Statistics)

In Python, we would use a Correlation Heatmap. In Dataiku, we have a dedicated tab for this.

1. While in the `customers_raw` dataset, go to the **Statistics** tab.
2. Click **+ Create your first worksheet** -> select **Correlation Matrix**.
3. Select the numerical variables (e.g., `age`, `purchase_count`, `avg_basket_value`, `is_lapsed`, `satisfaction_score`).
4. Click **Compute**. Dataiku will generate an interactive heatmap. Investigate which variables seem to be most strongly correlated with the `is_lapsed` variable.

## Step 5: Quick Baseline Model (AutoML / Visual ML)

This is the most exciting part. We will create our first baseline model without writing any code.

1. Return to the **Explore** tab for the `customers_raw` dataset.
2. Right-click on the `is_lapsed` column header and select **Create Prediction Model** (or click the **Lab** button in the top right corner -> *AutoML Prediction* -> select `is_lapsed`).
3. Choose the **Quick Prototypes** template and click **Create**.
4. **CRITICAL BUSINESS NOTE:** Go to the **Design** tab in the model panel, and then to the **Features** section.
   - Find the `days_since_last_purchase` variable and **turn it off** (uncheck the toggle). Why? According to the instructions, this variable 100% determines customer churn (>90 days). Using it causes what is known as *data leakage*.
5. Go to the **Algorithms** section. Dataiku will select e.g., Random Forest and Logistic Regression by default. Leave them as they are.
6. Click the green **TRAIN** button (in the upper right corner).
7. After a moment, Dataiku will display the results. Click on the best model to view its details.
8. Go to the **Performance** -> **ROC curve** tab to see the area under the curve (ROC-AUC). The expected result for the baseline is around ~0.83.
9. Go to the **Explainability** -> **Variables Importance** tab to see which features (at this early stage) the model considered most important.

## Step 6: Summarizing observations (Reporting)

In Dataiku, you can create a visual Dashboard to answer the questions from the original notebook.

1. Go to the main project menu and select **Dashboards**.
2. Create a new Dashboard. You can "pin" (using the pin icon) the previously created charts from the *Charts* tab, the correlation matrix, and the ROC-AUC model results from the *Lab* section.
3. Add Text insight tiles, where you, as a team, will answer the 5 control questions from stage 1 (e.g., why a high *accuracy* is misleading with an 18% churn rate, and what data quality issues were noticed).