![Dataset Overview](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/dataset-card.jpg?raw=true)

# BigMart Sales Prediction
## Project Background

BigMart is a retail chain operating multiple outlet formats across different locations and store sizes.
Understanding the key drivers behind product-level sales performance is critical for **pricing strategy, store planning, shelf space allocation, and new product launch decisions.**

However in practice

Sales performance varies significantly across outlets

The relationship between price, visibility, product attributes, and outlet characteristics is non-linear

Traditional linear assumptions often fail to capture real consumer behavior

This project aims to bridge that gap by combining business analysis and machine learning to answer a core business question:

> “Given a product’s price, visibility, and outlet characteristics, what level of sales should we realistically expect per outlet?”

---

## Data Structure, Initial Checks, and Key Concepts

### Dataset Structure
The dataset consists of **8,523 transaction-level records** representing product sales across multiple retail outlets.  
Each row corresponds to the sales performance of a specific product within a particular outlet.

- **Target Variable**
  - `Item_Outlet_Sales`: Total sales value of an item in a given outlet

- **Numerical Features**
  - `Item_Weight`: Weight of the product
  - `Item_Visibility`: Percentage of shelf space allocated to the item
  - `Item_MRP`: Maximum Retail Price of the product
  - `Outlet_Establishment_Year`: Year the outlet was established

- **Categorical Features**
  - Product attributes: `Item_Type`, `Item_Fat_Content`
  - Store attributes: `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
 
### Data Overview

- **Total Records**: 8,523  
- **Total Features**: 13  

#### Financial Ratios

- **Total Sales**: 18,591,125.41  
- **Average Sales per Item**: 11,925.03  
- **Average Sales per Outlet**: 1,859,112.54  

#### Top 5 Sales Contribution by Item Type

- Fruits and Vegetables: 15.17%  
- Snack Foods: 14.70%  
- Household: 11.06%  
- Frozen Foods: 9.82%  
- Dairy: 8.19%  

#### Performance Ratios

#### Average Sales by Outlet Type

- Supermarket Type 3: 3,694.04  
- Supermarket Type 1: 2,316.18  
- Supermarket Type 2: 1,995.50  
- Grocery Store: 339.83  

#### Average Sales by Outlet Size

- Medium: 2,681.60  
- High: 2,299.00  
- Small: 1,867.18  

#### Marketing Ratios

- **Sales per Visibility Unit**: 30,909.74  
- **Visibility Efficiency**: 0.000032  

#### Price & Demand Ratios

- **Sales per MRP Unit**: 15.47  
- **High vs Low Price Sales Ratio**: 2.22

![Dataset Overview](https://raw.githubusercontent.com/SethSterlin/BigMart-Sales-Prediction/main/Outlet%20Performance%20Overview.png)

This project analyzes **BigMart sales performance across products and outlets** to uncover patterns that directly support a **sales prediction model**. The goal is not only to describe historical performance but to explain *why* certain variables are critical for accurate prediction.

#### Product Category Performance
**Fruits and Vegetables generate the highest total sales**, indicating a **high-frequency, necessity-driven demand**. This category acts as a stable revenue base, while categories such as **Snack Foods and Household items** provide diversified contributions.  
This justifies including **`Item_Category`** as a core predictive feature.

#### Outlet Type Analysis
**Supermarket Type 3 consistently achieves the highest average sales**, reflecting the impact of **store format, scale, and customer traffic**. In contrast, **Grocery Stores show significantly lower sales**, suggesting structural limitations.  
**`Outlet_Type`** is therefore a **strong explanatory and predictive variable**.

#### Outlet Size Effect
Contrary to intuition, **Medium-sized outlets outperform both Small and High-sized outlets** in average sales. This suggests that **operational efficiency and location density** may outweigh sheer size.  
This insight reinforces the predictive value of **`Outlet_Size`**.

#### Visibility & Marketing Impact
Although the efficiency metric appears numerically small, **Item Visibility shows a measurable positive relationship with sales**. Given the large sales base, even marginal visibility improvements can lead to meaningful revenue gains.  
**`Item_Visibility` is treated as a meaningful signal, not noise**, in the prediction model.

#### Pricing Dynamics
The **Sales per MRP Unit** and the **High vs Low Price Sales Ratio (2.22)** reveal that while lower-priced items drive volume, **higher-priced items still perform strongly**. This confirms **price sensitivity without eliminating premium demand**.  
**`Item_MRP`** is essential for capturing **non-linear pricing effects** in prediction.

#### From Insight to Prediction
These findings explain *why* features such as **`ItemCategory`, `OutletType`, `OutletSize`, `ItemVisibility`, and `ItemMRP`** are included in the **machine learning sales prediction model**. The model is therefore grounded not only in statistical performance but also in **clear business logic and real-world behavior**.
- Optimize outlet strategy and store formats  
- Improve shelf allocation and visibility planning  
- Understand price sensitivity and demand behavior

### Exploratory Data Analysis: Distribution & Outliers

Understanding the **distribution of key numerical variables** is a crucial step before building a predictive model.  
The following visualizations highlight the **spread, skewness, and presence of outliers** in the BigMart dataset, which directly influence model performance and feature engineering decisions.

#### Item MRP Distribution

| Visualization | Insight |
|--------------|--------|
| ![](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/hist%20im.png?raw=true) | The distribution of **Item_MRP (Maximum Retail Price)** shows a **multi-modal pattern**, suggesting that products are grouped into several pricing tiers rather than following a normal distribution. This indicates that pricing strategy plays a significant role in sales behavior. While extreme high-price values exist, they appear to be valid premium products rather than noise, so they were retained for modeling. |

#### Item Visibility Distribution

| Visualization | Insight |
|--------------|--------|
| ![](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/hist%20iv.png?raw=true) | **Item_Visibility** is highly **right-skewed** with several extreme values. These outliers may represent products that dominate shelf space or suffer from data recording inconsistencies. Instead of removing them outright, the feature was carefully reviewed, as visibility can be a strong sales driver. In later stages, transformation techniques can help stabilize this distribution. |

#### Item Weight Distribution

| Visualization | Insight |
|--------------|--------|
| ![](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/hist%20iw.png?raw=true) | The **Item_Weight** distribution is relatively smooth but contains a few extreme values. These outliers are likely genuine heavy or lightweight items rather than data errors. Since weight may indirectly affect logistics and consumer preference, these values were preserved to maintain real-world variability. |

#### Outlet Sales Distribution

| Visualization | Insight |
|--------------|--------|
| ![](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/hist%20os.png?raw=true) | **Outlet_Sales** shows a strong **right-skewed distribution** with notable high-value outliers. These represent top-performing outlets or products with exceptionally strong demand. Such outliers are especially important for prediction tasks, as they help the model learn patterns associated with high sales performance rather than focusing only on average cases. |


- Not all outliers are errors — many reflect **real business phenomena** such as premium pricing or high-performing outlets  
- Blindly removing outliers may distort important sales patterns  
- Instead of aggressive removal, this project emphasizes **understanding context** and applying transformations only when necessary  

These insights informed subsequent **feature engineering and model selection**, ensuring that the prediction model captures both typical and exceptional sales behaviors.

### Exploratory Data Analysis: Handling Missing Values

Before building predictive models, missing values were carefully examined and handled to ensure data quality and reliable model performance.

From the initial data inspection, only two features contained missing values:

| Feature        | Missing Values |
|---------------|----------------|
| `Item_Weight`   | 1,463          |
| `Outlet_Size`   | 2,410          |

All other variables, including the target variable **`Item_Outlet_Sales`**, had no missing values.

#### 1. Handling Missing Values in `Item_Weight`

```python
df["Item_Weight"].fillna(df["Item_Weight"].mean(), inplace=True)
```

**Approach:** Mean Imputation

**Reasoning:**
`Item_Weight` is a continuous numerical variable representing product weight.
Since its distribution is relatively stable and does not show extreme skewness, replacing missing values with the **mean weight** helps preserve the overall distribution without introducing bias.

**Business Interpretation:**
Missing product weights are assumed to be similar to the average weight of existing products, which is a reasonable assumption in retail datasets where product specifications are standardized.

#### 2. Handling Missing Values in `Outlet_Size`

```python
df["Outlet_Size"] = (
    df.groupby("Outlet_Type")["Outlet_Size"]
      .transform(lambda x: x.fillna(x.mode()[0]))
)
```

**Approach:** Group-wise Mode Imputation

**Reasoning:**
`Outlet_Size` is a categorical variable (Small, Medium, High) and is highly related to `Outlet_Type`.
Instead of using a global mode, missing values were filled using the **most frequent outlet size within each outlet type**.

**Why this matters:**
Different outlet types (e.g., Grocery Store vs Supermarket Type 3) naturally tend to have different typical sizes.
This method preserves **structural relationships** in the data and avoids unrealistic combinations.

**Business Interpretation:**
When outlet size information is missing, it is inferred based on the typical size of similar outlet formats, which aligns with how retail chains are usually standardized.

| Feature       | Missing Values | Method Used                 | Rationale |
|--------------|----------------|-----------------------------|-----------|
| `Item_Weight`  | 1,463          | Mean Imputation             | Numerical feature with a relatively stable distribution; using the mean preserves overall statistical properties without introducing strong bias |
| `Outlet_Size`  | 2,410          | Mode Imputation (by Outlet Type) | Categorical feature highly correlated with outlet format; imputing by group-level mode maintains structural and business consistency |

This targeted and context-aware approach ensures that:

No records were dropped unnecessarily

### Multicollinearity Check (VIF Analysis)

**Multicollinearity** occurs when two or more independent variables in a model are highly correlated with each other.  
In simple terms, it means **some features are telling almost the same story**.

For example:
- Product price (MRP) and product weight may be related
- Store size and store type may overlap in meaning

When multicollinearity exists:
- Linear models struggle to understand the *true impact* of each variable
- Coefficients become unstable or misleading
- Model interpretation becomes unreliable, even if accuracy looks acceptable

**Variance Inflation Factor (VIF)** is a metric used to **measure how much multicollinearity exists** in a feature.

It answers the question:

“How much does this feature overlap with other features in the model?”

General interpretation:
- **VIF = 1** → No correlation with other features
- **VIF between 1–5** → Acceptable correlation
- **VIF > 5** → High multicollinearity (potential problem)
- **VIF > 10** → Serious multicollinearity (should be fixed)

To assess multicollinearity among key numerical features, Variance Inflation Factor (VIF) was calculated for the main continuous variables used in the model.

| Feature          | VIF Value |
|------------------|-----------|
| `Item_Weight`      | 4.75      |
| `Item_Visibility`  | 2.37      |
| `Item_MRP`         | 4.38      |

#### Multicollinearity Interpretation

- All VIF values are **below the commonly accepted threshold of 5**, indicating that multicollinearity is **not a critical issue** in this dataset.
- **`Item_Visibility`** shows very low multicollinearity, suggesting it provides largely independent information.
- **`Item_Weight`** and **`Item_MRP`** exhibit moderate correlation with other features, but remain within an acceptable range.
- Downstream models (Linear Regression and XGBoost) could learn from realistic and business-consistent inputs

---

## Analysis Methodology
### Defining Features and Target Variable

In this step, the dataset is separated into **input features (X)** and the **target variable (y)** in preparation for model training.

```python
x = df_model.drop("Item_Outlet_Sales", axis=1)
y = df_model["Item_Outlet_Sales"]
```

- **X (Features):**  
  Contains all independent variables that are used to predict sales.  
  The target column `Item_Outlet_Sales` is explicitly removed to prevent data leakage.

- **y (Target Variable):**  
  Represents the dependent variable `Item_Outlet_Sales`, which is the value the model aims to predict.

This separation ensures a clear distinction between predictors and the prediction target, following standard machine learning best practices.

### Train–Test Split

The dataset is divided into **training** and **testing** sets to evaluate the model’s performance on unseen data.

- **Training Set (80%)**  
  Used to train the machine learning model and learn patterns from historical data.

- **Testing Set (20%)**  
  Held out and used to assess how well the model generalizes to new, unseen data.

- **random_state = 42**  
  Ensures reproducibility by keeping the data split consistent across runs.

This step helps prevent overfitting and provides an unbiased evaluation of the model’s predictive performance.

### Model Training – Linear Regression

In this step, a Linear Regression model is initialized and trained using the training dataset.

The model learns the relationship between the input features (`x_train`) and the target variable (`y_train`) by estimating coefficients that minimize the overall prediction error.

Once trained, the model can be used to generate sales predictions and evaluate performance on unseen test data.

This Linear Regression model serves as a **baseline model**, providing an interpretable reference point before moving to more advanced models such as XGBoost.

![Linear Regression: Actual vs Predicted Sales](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/lr%20scatter.png?raw=true)

To evaluate how well the Linear Regression model performs, two key diagnostic plots are used:  
**Actual vs Predicted Sales** and **Residual Plot**.

#### Actual vs Predicted Item Outlet Sales

This scatter plot compares the **actual sales values** with the **predicted sales values** generated by the model.

- The dashed diagonal line (y = x) represents a **perfect prediction**, where predicted values exactly match actual values.
- Most data points follow a clear upward trend, indicating that the model successfully captures the **overall direction** of sales.
- The model performs relatively well in the **low to medium sales range**, where predictions are closer to actual values.
- However, for **high sales values**, predictions tend to fall below the diagonal line, suggesting that the model **underestimates high-performing outlets or products**.

**Business Insight:**  
The model is effective for understanding general sales behavior and making baseline forecasts, but it struggles to fully capture extreme sales scenarios.

#### Residual Plot

The residual plot shows the difference between **actual sales and predicted sales** (residuals) plotted against predicted values.

- Residuals are centered around zero, which indicates that the model does not have a strong systematic bias.
- As predicted sales increase, the spread of residuals becomes wider, forming a **fan-shaped pattern**.
- This pattern indicates **heteroscedasticity**, meaning prediction errors increase for higher sales values.

**Business Interpretation:**  
Sales behavior in high-performing outlets is more complex and volatile, likely influenced by factors such as promotions, store traffic, or demand spikes that a linear model cannot fully explain.

#### Linear Regression Key Takeaways

- Linear Regression provides a **solid and interpretable baseline model**.
- The model explains general sales trends well but has **limited accuracy for high-sales cases**.
- Increasing error variance suggests the presence of **non-linear relationships** in the data.
- More advanced models (e.g., tree-based models like XGBoost) are recommended to better capture these patterns.

#### Limitations of Linear Regression

Although Linear Regression provides strong interpretability and serves as a solid baseline, it relies on several assumptions:

- **Linearity**: Assumes a linear relationship between features and sales.
- **Constant variance (Homoscedasticity)**: Error variance should be stable across all prediction levels.
- **Limited interaction handling**: Feature interactions must be manually engineered.

From the residual analysis, we observed:

- Increasing prediction error at higher sales levels.
- Fan-shaped residual patterns indicating **heteroscedasticity**.
- Systematic underestimation for high-performing outlets.

These patterns suggest that real-world sales behavior is **non-linear and interaction-driven**, which exceeds the capability of a simple linear model.

**Why XGBoost Is a Better Fit**

XGBoost (Extreme Gradient Boosting) is a **tree-based ensemble model** that overcomes many of these limitations.

**Key advantages of XGBoost:**

- **Captures non-linear relationships** automatically without manual feature transformation.
- **Models feature interactions naturally**, such as:
  - Outlet Type × Outlet Size  
  - Item MRP × Visibility  
- **Robust to heteroscedasticity**, handling varying error distributions more effectively.
- **Handles outliers better** by learning decision boundaries instead of fitting a single global line.
- Provides **feature importance scores**, allowing continued business interpretation.

### XGBoost Model Training & Evaluation

After identifying the limitations of Linear Regression, an **XGBoost Regressor** was implemented to capture non-linear relationships and feature interactions in sales data.

**Model Initialization**

The XGBoost model was configured with carefully selected hyperparameters to balance model complexity, generalization, and training stability:

- **n_estimators = 300**  
  Number of boosting trees. More trees allow the model to learn complex patterns.

- **learning_rate = 0.05**  
  Controls how much each tree contributes to the final prediction. A lower value improves generalization.

- **max_depth = 4**  
  Limits tree depth to prevent overfitting while still capturing interactions.

- **min_child_weight = 5**  
  Prevents the model from learning patterns based on very small sample splits.

- **subsample = 0.8**  
  Uses 80% of the training data for each tree, improving robustness.

- **colsample_bytree = 0.8**  
  Randomly samples 80% of features per tree to reduce feature dependency.

- **reg_alpha = 0.1 (L1 regularization)**  
  Encourages sparsity and reduces noise.

- **reg_lambda = 1.0 (L2 regularization)**  
  Penalizes large weights to improve stability.

- **random_state = 42**  
  Ensures reproducibility.

#### Model Training - XGBoost

The model was trained using the training dataset:

```python
regressor.fit(x_train, y_train)
```

Predictions were generated for both training and test sets to evaluate performance and generalization:

```python
y_hat_train = regressor.predict(x_train)
y_hat_test  = regressor.predict(x_test)
```

![Dataset Overview](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/XGB%20scatter.png)

- Each point represents a single product–outlet observation.
- The dashed diagonal line (y = x) represents a **perfect prediction**, where predicted sales equal actual sales.

**Key Observations**

- Most data points are closely clustered around the diagonal line, indicating **strong predictive alignment** between actual and predicted values.
- The model performs well across **low to medium sales ranges**, where predictions closely track actual outcomes.
- Compared to Linear Regression, XGBoost shows **reduced underprediction** in higher sales segments.
- Although some dispersion remains at very high sales levels, the overall error spread is more stable.

**Business Interpretation**

- XGBoost captures **non-linear demand patterns** that commonly occur in retail sales data.
- The model is better suited for forecasting sales in **high-performing outlets**, where Linear Regression struggled due to its linear assumptions.
- This improvement supports more reliable decision-making for:
  - Product launch forecasting
  - Inventory planning
  - Outlet performance evaluation
