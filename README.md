# Practice Project - The Condo Services Retention Model

## Project Overview

In this practice project, I developed a comprehensive data analysis and predictive modeling solution for identifying condo buildings at risk of churning from a property services provider. Key activities included:

* **Data Generation**: Created a synthetic dataset with 1,500 building records featuring realistic churn logic based on contract type, spending patterns, service usage, and account tenure.
* **Data Cleaning**: Verified data quality with no missing values, duplicates, or significant outliers detected. Filtered dataset to focus on the North Region (~900 buildings).
* **Exploratory Data Analysis**: Conducted univariate, bivariate, and multivariate visualizations to understand churn drivers across building characteristics and service usage.
* **Feature Engineering**: Created domain-relevant features (spend_per_unit, number_of_services, individual service flags) and implemented One-Hot Encoding for categorical variables.
* **Predictive Modeling**: Built and validated a Logistic Regression classifier to predict building churn with class imbalance handling.
* **Model Interpretation**: Analyzed model coefficients to explain feature contributions and guide business recommendations.

This project showcases proficiency in Python, statistical analysis, data preprocessing, handling class imbalance, and machine learning while demonstrating the ability to build classification models following the PACE methodology.

## Project Details: The Condo Services Retention Model

This hands-on project focused on creating a predictive model for a condo services provider to identify buildings in the North Region most likely to churn, enabling targeted retention interventions by the Operations Director. The dataset was synthetically generated with class imbalance (~20% churn rate) to test real-world data handling capabilities.

### Key Components

#### 1. Data Generation
- Created **synthetic dataset** with 1,500 condo building records
- Implemented realistic churn logic based on:
  - Price Sensitivity (spend per unit threshold > $1,200)
  - Contract Type (Transactional = high risk, Annual Subscription = lower risk)
  - "Sticky" Services (Concierge service reduces churn risk)
  - Legacy Clients (account age > 10 years = lower risk)
- Introduced intentional data challenges:
  - String-based services column requiring parsing
  - Class imbalance (~20% churn rate)
  - Multi-region data requiring filtering

#### 2. Data Cleaning and Preprocessing
- Verified **no duplicate records** in the dataset
- Confirmed **no missing values** across all features
- **Filtered dataset** to North Region only (~900 buildings)
- Dropped data generation artifacts (churn_prob column)
- Validated data types and distributions

#### 3. Exploratory Data Analysis
- Performed comprehensive **univariate analysis**:
  - Count plots for categorical variables (contract_type, churn)
  - Count plots for discrete variables (account_age_years, number_of_services)
  - Histogram and KDE plots for continuous variables (annual_spend, condo_size_units)
- Conducted **bivariate analysis**:
  - Box plots and count plots comparing feature distributions by churn status
  - Identified that churned buildings have lower annual spend, smaller sizes, and shorter account ages
- Performed **multivariate analysis**:
  - KDE contour plots of annual spend vs. condo size by churn status
  - Identified distinct clustering patterns between churned and retained buildings

#### 4. Statistical Testing
- Conducted **Independent T-Tests** for numeric features vs. churn
- Performed **Chi-Squared Tests** for categorical features vs. churn
- Assessed **statistical significance** of all features (p < 0.05)
- Checked **multicollinearity** using:
  - Correlation matrix heatmaps
  - Iterative Variance Inflation Factor (VIF) analysis with protected features

#### 5. Feature Engineering
- Created **domain-relevant engineered features**:
  - `number_of_services`: Count of services used per building
  - `spend_per_unit`: Annual spend / condo size units (price sensitivity metric)
  - `has_cleaning`, `has_security`, `has_concierge`, `has_landscaping`, `has_pool_maint`: Binary flags for each service
- Applied **One-Hot Encoding** for contract_type categorical variable
- Performed **iterative VIF-based feature removal** to manage multicollinearity (with protected features)

#### 6. Predictive Modeling
- Built **Logistic Regression Classifier** using scikit-learn
- Addressed **class imbalance** using `class_weight='balanced'`
- Split data into training (80%) and test (20%) sets with stratification
- Applied **StandardScaler** for feature normalization
- Performed **5-fold Stratified Cross-Validation** for robust evaluation
- Validated **Box-Tidwell test** for linearity of logit assumption

#### 7. Model Evaluation and Interpretation
- Evaluated model performance using:
  - **AUC-ROC**: 0.8572
  - **Recall**: 0.80 (Class 1)
  - **Precision**: 0.09 (Class 1)
- Analyzed **model coefficients** for feature importance interpretation
- Validated generalization with train-test gap analysis (0.05)
- Generated confusion matrix visualization

### Technologies Used

- **Python Libraries**:
  - `numpy`: Numerical computations and array operations
  - `pandas`: Data manipulation and analysis
  - `matplotlib`: Data visualization and plotting
  - `seaborn`: Statistical data visualization
  - `scipy.stats`: Statistical testing (t-tests, chi-squared)
  - `statsmodels`: VIF calculation, Box-Tidwell test, Logit model
  - `scikit-learn`: Machine learning and model evaluation
    - `train_test_split`: Stratified data splitting
    - `LogisticRegression`: Predictive modeling
    - `StratifiedKFold`: Cross-validation strategy
    - `StandardScaler`: Feature scaling
    - Classification metrics: AUC-ROC, precision, recall, F1-score

- **Statistical Methods**:
  - Independent T-Tests (Welch's t-test)
  - Chi-Squared Tests for categorical variables
  - Variance Inflation Factor (VIF) analysis
  - Box-Tidwell test for linearity of logit

### Project Structure

The notebook follows the **PACE methodology** (Plan → Analyze → Construct → Execute):

```
The Condo Services Retention Model/
├── Data Analysis Template v7.ipynb          # Main analysis notebook
├── north_region_churn_analysis_ready.csv    # Preprocessed dataset
├── README.md                                # Project documentation
└── images/                                  # PACE methodology icons
    ├── Plan.png
    ├── Analyze.png
    ├── Construct.png
    └── Execute.png
```

### Dataset Specifications

| Feature            | Description                                      | Data Type   | Example Values                                    |
|--------------------|--------------------------------------------------|-------------|---------------------------------------------------|
| building_id        | Unique identifier for each condo building        | Integer     | 1001, 1002, ..., 2500                             |
| region             | Geographic region of the building                | Categorical | North, South, East, West                          |
| condo_size_units   | Number of condo units in the building            | Integer     | 10–500                                            |
| contract_type      | Type of service contract                         | Categorical | Annual_Subscription, Monthly_Retainer, Transactional |
| account_age_years  | Number of years the building has been a client   | Integer     | 1–15                                              |
| services_used      | Comma-separated list of services used            | String      | Cleaning,Security,Concierge                       |
| annual_spend       | Estimated annual spend on services               | Float       | ~$5,000–$300,000                                  |
| churn              | Churn label (target variable)                    | Binary      | 0 (Stayed), 1 (Churned)                           |

**Engineered Features:**

| Feature                        | Description                                  | Data Type |
|--------------------------------|----------------------------------------------|-----------|
| number_of_services             | Count of services used per building          | Integer   |
| spend_per_unit                 | Annual spend / condo size units              | Float     |
| has_cleaning                   | Building uses Cleaning service               | Binary    |
| has_security                   | Building uses Security service               | Binary    |
| has_concierge                  | Building uses Concierge service              | Binary    |
| has_landscaping                | Building uses Landscaping service            | Binary    |
| has_pool_maint                 | Building uses Pool Maintenance service       | Binary    |
| contract_type_Monthly_Retainer | One-hot encoded contract type                | Binary    |
| contract_type_Transactional    | One-hot encoded contract type                | Binary    |

**Data Quality Notes:**
- No missing values detected across all features
- No significant outliers detected via IQR method
- Class imbalance: ~20% churn rate (handled with class weighting)
- Dataset filtered to North Region (~900 buildings from 1,500 total)

### Skills Demonstrated

- Synthetic data generation with realistic business logic
- Data quality assessment (duplicates, missing values, outliers)
- Exploratory data analysis with multi-level visualizations (univariate, bivariate, multivariate)
- Statistical significance testing (t-tests, chi-squared)
- Feature engineering based on domain knowledge
- Multicollinearity assessment and iterative VIF-based feature removal
- Handling class imbalance with class weighting
- Logistic Regression classification modeling
- Cross-validation with Stratified K-Fold
- Model assumption validation (Box-Tidwell test for linearity of logit)
- PACE methodology implementation

### Key Findings

**Statistical Test Results:**
- Key features showed **statistically significant** relationships with churn (p < 0.05)
- `spend_per_unit`, `account_age_years`, and `contract_type` showed strongest associations
- `spend_per_unit` retained as protected feature despite high VIF due to business importance

**Model Performance:**

| Metric    | Class 0 (Stayed) | Class 1 (Churned) |
|-----------|------------------|-------------------|
| AUC-ROC   | —                | 0.8572            |
| Recall    | —                | 0.80              |
| Precision | —                | 0.09              |

**Feature Impact (Coefficient Analysis):**
- **spend_per_unit**: Strongest predictor; higher cost per unit increases churn risk
- **account_age_years**: Longer tenure associated with lower churn probability
- **has_concierge**: Buildings with concierge service are less likely to churn (sticky service)
- **contract_type_Transactional**: Transactional contracts show elevated churn risk
- **contract_type_Monthly_Retainer**: Monthly retainer clients have moderate churn risk

**Business Insights:**
- Buildings with high spend-per-unit (price-sensitive buildings) are at significantly higher risk
- Transactional contract buildings churn more than Annual Subscription buildings
- Legacy clients (>10 years) show lower churn risk
- Concierge service creates a "sticky" relationship that reduces churn likelihood
- Smaller buildings with disproportionately high spending need proactive outreach

### Outcome

Successfully developed a Logistic Regression classifier that identifies **80% of buildings that will churn** (recall) with an AUC-ROC of 0.8572, enabling targeted retention efforts for the North Region Operations Director. While precision (9%) indicates significant false positives, the business trade-off is acceptable given that the cost of losing a building client exceeds the cost of proactive retention outreach.

**Champion Model:** Logistic Regression with class_weight='balanced'
- **AUC-ROC**: 0.8572 (exceeds target of 0.75)
- **Recall**: 0.80 (Class 1)
- **Train-CV Gap**: 0.05 (acceptable generalization)
- **Business Value**: Enables the Operations Director to proactively identify and contact at-risk buildings

### How to Run

1. **Install Dependencies:**
   ```bash
   pip install numpy pandas matplotlib seaborn scipy statsmodels scikit-learn
   ```

2. **Open Jupyter Notebook:**
   ```bash
   jupyter notebook "Data Analysis Template v7.ipynb"
   ```

3. **Run All Cells:**
   - Execute cells sequentially following the PACE methodology
   - Dataset is generated synthetically within the notebook
   - All preprocessing, analysis, and modeling steps are automated

4. **Review Outputs:**
   - Exploratory data visualizations
   - Statistical test results and p-values
   - Correlation heatmap and VIF analysis
   - Model performance metrics (AUC-ROC, precision, recall, F1)
   - Confusion matrix and coefficient analysis

### Success Criteria

**Target Metrics:**
- AUC-ROC ≥ 0.75
- Correctly identify at least 40 of the top 50 at-risk buildings

**Achieved:**
- AUC-ROC: 0.8572 (exceeds target)
- Recall: 0.80 (strong identification of churners)

**Interpretation:**
- High recall ensures most churning buildings are identified
- Low precision is acceptable given the business context where missing a churner is costlier than a false alarm
- Model provides actionable foundation for the Operations Director's retention program

### Future Enhancements

- Implement **Random Forest** and **XGBoost** models for comparison
- Explore **SMOTE** or other resampling techniques for class imbalance
- Perform **threshold tuning** to optimize precision-recall balance
- Add **SHAP analysis** for deeper feature interpretation
- Apply **RFECV** for systematic feature selection
- Investigate the **has_concierge** feature's unexpected importance further
- Extend analysis to other regions (South, East, West) for company-wide deployment