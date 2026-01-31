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

**“Given a product’s price, visibility, and outlet characteristics, what level of sales should we realistically expect per outlet?”**

### Project Highlights

- Built an end-to-end retail sales prediction pipeline from EDA to machine learning  
- Used Linear Regression as a baseline and XGBoost to capture non-linear sales drivers  
- XGBoost improved prediction accuracy by ~50% RMSE over Linear Regression  
- Identified outlet characteristics as stronger sales drivers than product attributes  
- Estimated realistic expected sales of **~2,500 per outlet** for business planning

Tools: `Python`

The analysis was conducted using a Jupyter Notebook environment on Google Colab. You can view the notebook [here](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/Big_Mart_Project.ipynb).

---

## Data Structure, Initial Checks, and Key Concepts

### Dataset Used

[BigMart Sales Data](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data/data)

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

#### Model Training-XGBoost

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

This scatter plot visualizes the relationship between **actual sales** and **predicted sales** generated by the XGBoost model on the test dataset.

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
 
### Feature Importance Analysis (XGBoost)

This section explains how XGBoost identifies the key drivers of sales and how to interpret the results from a business perspective.

Steps:
- Extract feature importance from the trained model
- Map importance values to actual feature names
- Select the top 10 most important features
- Visualize them using a horizontal bar chart (barh)

![Dataset Overview](https://github.com/SethSterlin/BigMart-Sales-Prediction/blob/main/top%2010%20driver%20sales.png)

The chart is built using `feature_importances_` from the trained **XGBoost Regressor**.

Each importance score represents how much a feature contributes to reducing prediction error across all decision trees in the model.  
A higher importance value indicates a stronger influence on sales prediction.

A red color gradient is applied:
- **Darker bars = more influential features**
- This helps quickly identify the strongest drivers at a glance

#### Top 10 Sales Drivers – Interpretation

**Outlet Type (Primary Driver)**

The most influential features are related to **Outlet Type**:

- **Supermarket Type1** (highest importance)
- **Supermarket Type3**
- **Supermarket Type2**

**Business Insight:**  
Outlet type is the strongest determinant of sales performance.  
This reflects differences in customer traffic, store scale, and purchasing power.  
In practice, *where a product is sold matters more than what the product is*.

**Outlet Size & Pricing (Strong Secondary Drivers)**

- **`Outlet_Size_Medium`**
- **`Item_MRP`**

**Business Insight:**  
Medium-sized outlets consistently outperform expectations, indicating an optimal balance between foot traffic and product visibility.  
Price remains an important driver, but its impact is still secondary to store characteristics.

**Location Factors (Supporting Role)**

- **`Outlet_Location_Type_Tier 2`**
- **`Outlet_Location_Type_Tier 3`**

**Business Insight:**  
Location contributes positively to sales but acts as a supporting factor rather than a primary driver.  
Store format and size outweigh pure geographic effects.

**Product-Level Factors (Lower Relative Impact)**

- **`Item_Visibility`**
- **`Item_Type (Fruits & Vegetables)`**

**Business Insight:**  
Product attributes do influence sales; however, their impact is significantly smaller compared to outlet-related factors.  
This highlights the importance of distribution strategy over individual product features.

**Executive Takeaway**

**Sales performance is driven more by outlet characteristics than by product attributes.**  
Optimizing outlet type, store size, and placement strategy has a greater impact than fine-tuning individual product features.

This insight supports strategic decisions around store expansion, product placement, and channel optimization.

### Model Comparison: Linear Regression vs XGBoost

This project compares a traditional **Linear Regression** model with an advanced **XGBoost Regressor** to evaluate their effectiveness in predicting retail sales in a real-world business dataset.

#### Linear Regression (Baseline Model)

**Purpose:**  
Linear Regression is used as a baseline due to its simplicity and strong interpretability.

**Performance (Test Set):**
- **R²**   : -0.53  
- **MAE**  : 1,720  
- **RMSE** : 2,174  

**Observations:**
- The negative R² indicates that the model performs worse than predicting the mean sales value.
- Errors are large, especially for high-sales observations.
- Residual plots show increasing variance as predicted sales increase (heteroscedasticity).

**Limitations:**
- Assumes a **linear relationship** between features and sales.
- Cannot capture **non-linear effects**, **feature interactions**, or **demand spikes**.
- Struggles with high-sales outlets where business dynamics are more complex.

**Business Interpretation:**  
Linear Regression is useful for explaining basic relationships (e.g., outlet type impact) but is not reliable for accurate sales forecasting in real retail environments.

#### XGBoost Regressor (Advanced Model)

**Purpose:**  
XGBoost is applied to overcome the limitations of linear models by learning non-linear patterns and interactions between features.

**Performance (Test Set):**
- **R²**   : 0.58  
- **MAE**  : 798  
- **RMSE** : 1,138  

**Performance (Train Set):**
- **R²**   : 0.68  
- **RMSE** : 964  

**Observations:**
- XGBoost reduces prediction error by nearly **50%** compared to Linear Regression.
- The model captures high-sales behavior more effectively.
- Train and test performance are closely aligned, indicating good generalization.

**Strengths:**
- Captures **non-linear relationships**
- Learns **interaction effects** between product, outlet, and location features
- Robust to outliers and skewed sales distributions

**Business Interpretation:**  
XGBoost provides significantly more reliable sales forecasts, especially for high-performing outlets and complex demand patterns.

#### Side-by-Side Comparison

| Aspect | Linear Regression | XGBoost |
|------|------------------|--------|
| Model Type | Parametric | Tree-based Ensemble |
| Linearity Assumption | Required | Not Required |
| Handles Non-linearity | ❌ | ✅ |
| Handles Feature Interaction | ❌ | ✅ |
| Test R² | -0.53 | 0.58 |
| Test RMSE | 2,174 | 1,138 |
| Interpretability | High | Medium |
| Predictive Power | Low | High |

#### Final Conclusion

- **Linear Regression** is suitable as a baseline and for high-level business interpretation.
- **XGBoost** is far superior for predictive accuracy and real-world deployment.
- Retail sales behavior is inherently non-linear and driven by complex interactions, making tree-based models a better choice.

### Using the Model for Sales Prediction (New Product Scenario)

After training both **Linear Regression** and **XGBoost** models, we can use them to estimate sales for a **new product scenario** by manually defining its characteristics.

**“If we launch a new snack product priced at 150, sold in Supermarket Type 1 outlets, located in Tier 2 areas with medium-sized stores, with a product weight of 9.3 grams, approximately 4.5% shelf visibility, and positioned as a regular snack (not low-fat),

what level of sales per outlet should we reasonably expect?”**

**1.Define Business Input Features**
Example scenario:
- Product type: Snack Foods
- Price (MRP): 150
- Weight: 9.3 grams
- Shelf visibility: 4.5%
- Fat content: Regular
- Outlet: Supermarket Type 1
- Outlet size: Medium
- Location: Tier 2

These features are encoded in a dictionary that matches the model’s training features.

```python
input = {
    "Item_Weight": 9.3,
    "Item_Visibility": 0.045,
    "Item_MRP": 150,
    "Item_Fat_Content_Regular": 1,
    "Item_Type_Snack Foods": 1,
    "Outlet_Size_Medium": 1,
    "Outlet_Location_Type_Tier 2": 1,
    "Outlet_Type_Supermarket Type1": 1
}
```

**2.Convert Input to Model-Ready Format**
The input dictionary is converted into a pandas DataFrame and aligned with the model’s feature set.

```python
input = pd.DataFrame([input])
input = input.reindex(columns=x_train.columns, fill_value=0)
```

Why this step is important:

- Ensures the input has exactly the same features as the training data

- Missing features are filled with 0 (for one-hot encoded variables)

- Prevents shape mismatch errors during prediction

**3.Generate Predictions**
The prepared input is passed to both models to generate sales predictions.

```python
pred_lr = lr.predict(input)
print(f"Predicted Sales (Linear Regression): {pred_lr[0]:,.2f}")

pred_xgb = regressor.predict(input)
print(f"Predicted Sales (XGBoost): {pred_xgb[0]:,.2f}")
```

**4.Business Interpretation**
Key Results

**Predicted Sales (Linear Regression): 2,622.85**

**Predicted Sales (XGBoost): 2,526.07**

#### What level of sales should we reasonably expect?

Considering both models, a **reasonable expected sales level per outlet is approximately _2,500–2,600_ per period** (same unit as the target variable).

### How to interpret this for executives

- **Model consensus:**  
  Although Linear Regression and XGBoost use very different assumptions, their predictions are close (difference ≈ 97), which increases confidence in the estimate.

- **Why XGBoost is more reliable:**  
  XGBoost demonstrated significantly better test performance (higher R², lower MAE/RMSE). Therefore, **~2,500** should be treated as the **more conservative and realistic baseline**.

- **Business meaning:**  
  For an outlet with:
  - Medium size  
  - Tier 2 location  
  - Supermarket Type 1  
  - Snack food category  
  - Average visibility and pricing  

  The outlet is expected to generate **around 2.5K in sales**, assuming normal operating conditions.

**Recommended usage**

- Use **2,500** as a **baseline forecast** for planning and benchmarking.
- Treat values **above 2,600** as **outperformance**.
- Treat values **below 2,300–2,400** as a **potential underperformance signal** requiring investigation (pricing, visibility, assortment, or location factors).

**Bottom line:**  
**Executives should reasonably expect **~2.5K sales per outlet**, with XGBoost providing the most decision-relevant estimate.**

## Recommendation & Business Usage

Based on the sales prediction results from both models:

- **Predicted Sales (Linear Regression):** 2,622.85  
- **Predicted Sales (XGBoost):** 2,526.07  

**Recommended Expected Sales per Outlet**

A **reasonable and defensible expected sales level per outlet is approximately 2,500–2,600 per period**  
(using the same unit as `Item_Outlet_Sales`).

**Why this range is appropriate**

- Both models, despite different assumptions, produce **very similar predictions** (difference ≈ 97), increasing confidence in the estimate.
- **XGBoost is preferred for decision-making** due to:
  - Higher test R²
  - Lower MAE and RMSE
  - Better handling of non-linear demand and outlet-level interactions

Therefore, **~2,500** should be treated as the **baseline forecast**, while **~2,600** represents an optimistic but still realistic outcome.

**Practical Business Interpretation**

For a product with the following characteristics:
- Snack Foods category
- MRP ≈ 150
- Regular fat content
- Medium-sized outlet
- Supermarket Type 1
- Tier 2 location
- Average shelf visibility

The outlet should be expected to generate **around 2.5K in sales under normal operating conditions**.

**Actionable Guidance**

- **Baseline planning target:** ~2,500  
- **Outperformance benchmark:** > 2,600  
- **Underperformance signal:** < 2,300–2,400  
  - Investigate pricing, shelf visibility, assortment fit, or outlet characteristics

**Executive Takeaway**

Use **XGBoost-based predictions (~2.5K)** as the primary reference for forecasting, inventory planning, and outlet performance benchmarking.
