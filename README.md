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
