# Data Preprocessing and Feature Engineering Report - WiDS Datathon 2025
*Generated on: 2025-03-20*

## 1. Overview

This report outlines the data preprocessing and feature engineering steps implemented for the WiDS Datathon 2025 project on women's brain health and ADHD. Based on our exploratory data analysis findings, we have developed a comprehensive pipeline to prepare the data for modeling.

## 2. Data Preprocessing

### 2.1 Handling Missing Values

**Identified Missing Data:**
- **MRI_Track_Age_at_Scan** (Quantitative): 360 missing values (29.7%)
- **PreInt_Demos_Fam_Child_Ethnicity** (Categorical): 11 missing values (0.9%)

**Implementation Strategy:**

1. **MRI_Track_Age_at_Scan**:
   - Used K-Nearest Neighbors (KNN) imputation with k=5
   - Features used for imputation: Other demographic variables including sex and race
   - Rationale: KNN captures relationships between age and other demographic factors better than mean/median imputation, while being more robust than regression imputation for this context

2. **PreInt_Demos_Fam_Child_Ethnicity**:
   - Used mode imputation (most frequent value)
   - Rationale: With only 0.9% missing, mode imputation preserves the distribution with minimal bias

### 2.2 Categorical Variable Encoding

**Implementation Strategy:**

1. **Binary Categorical Variables**:
   - Applied binary encoding (0/1) for:
     - PreInt_Demos_Child_Sex (Male=0, Female=1)

2. **Nominal Categorical Variables**:
   - Applied one-hot encoding for:
     - PreInt_Demos_Child_Race
     - PreInt_Demos_Fam_Child_Ethnicity
     - PreInt_Demos_Fam_Dwelling_Type
     - PreInt_Demos_Fam_Dwelling_Ownership
     - PreInt_Demos_Fam_Household_Structure
     - Instrument_Version
   - Rationale: One-hot encoding avoids introducing ordinal relationships where none exist

3. **Ordinal Categorical Variables**:
   - Applied ordinal encoding for:
     - PreInt_Demos_Fam_Highest_Education (mapped to education levels from 0 to 5)
     - PreInt_Demos_Fam_Household_Income (mapped to income brackets from 0 to 7)
   - Rationale: Preserves inherent ordering in these variables

### 2.3 Numerical Feature Scaling

**Implementation Strategy:**

1. **Standardization (z-score)**:
   - Applied to all behavioral assessment scores (SDQ_* and APQ_P_*)
   - Formula: z = (x - μ) / σ
   - Rationale: These features have different scales but approximately normal distributions, making standardization appropriate

2. **Min-Max Scaling**:
   - Applied to MRI_Track_Age_at_Scan
   - Formula: x_scaled = (x - min) / (max - min)
   - Rationale: Age has natural boundaries, and min-max scaling preserves the distribution shape

3. **No Scaling**:
   - ColorVision_CV_Score was left unscaled as it already has a standardized scale

## 3. Feature Engineering

### 3.1 Behavioral Feature Engineering

1. **Interaction Features**:
   - Created interaction terms between highly correlated features:
     - SDQ_Hyperactivity_X_ConductProblems = SDQ_SDQ_Hyperactivity * SDQ_SDQ_Conduct_Problems
     - SDQ_Emotional_X_Peer = SDQ_SDQ_Emotional_Problems * SDQ_SDQ_Peer_Problems
   - Rationale: EDA indicated potential interaction effects between these variables

2. **Polynomial Features**:
   - Created squared terms for top correlated features:
     - SDQ_Hyperactivity_Squared = SDQ_SDQ_Hyperactivity²
     - SDQ_Externalizing_Squared = SDQ_SDQ_Externalizing²
   - Rationale: To capture potential non-linear relationships

3. **Ratios**:
   - Created meaningful ratios:
     - Internalizing_to_Externalizing_Ratio = SDQ_SDQ_Internalizing / SDQ_SDQ_Externalizing
     - Prosocial_to_Difficulties_Ratio = SDQ_SDQ_Prosocial / SDQ_SDQ_Difficulties_Total
   - Rationale: These ratios represent balance between different behavioral patterns

### 3.2 Connectome Feature Engineering

1. **Dimensionality Reduction**:
   - Applied Principal Component Analysis (PCA) to connectome data
   - Retained 50 components, explaining approximately 85% of variance
   - Rationale: Connectome data is high-dimensional (19,901 features) with high collinearity

2. **Graph Metrics**:
   - Calculated global graph metrics for each subject:
     - Network density
     - Average clustering coefficient
     - Characteristic path length
     - Global efficiency
   - Rationale: These metrics capture brain network properties relevant to ADHD

3. **ROI-based Features**:
   - Created aggregated connectivity features for key regions implicated in ADHD:
     - Prefrontal connectivity strength
     - Striatal connectivity strength 
     - Default Mode Network (DMN) connectivity
     - Fronto-parietal connectivity
   - Rationale: Prior research implicates these specific networks in ADHD

### 3.3 Demographic Feature Engineering

1. **Age-based Features**:
   - Created age group categories (early childhood, middle childhood, adolescence)
   - Rationale: ADHD presentation varies across developmental stages

2. **SES Index**:
   - Created a composite socioeconomic status (SES) index combining:
     - PreInt_Demos_Fam_Highest_Education
     - PreInt_Demos_Fam_Household_Income
     - PreInt_Demos_Fam_Dwelling_Ownership
   - Rationale: Combined SES index may better capture environmental factors than individual variables

## 4. Feature Selection

After preprocessing and engineering, we implemented feature selection to improve model performance:

1. **Correlation-based Selection**:
   - Removed features with correlation > 0.85 to reduce multicollinearity
   - Retained the feature most correlated with the target from highly correlated pairs

2. **Feature Importance from Tree-based Models**:
   - Used Random Forest to rank feature importance
   - Selected top 100 features based on importance scores

3. **Sequential Feature Selection**:
   - Performed sequential forward selection with cross-validation
   - Selected optimal feature subset that maximizes model performance

## 5. Final Dataset

The final preprocessed dataset has the following characteristics:
- Total samples: 1,213
- Features after preprocessing and engineering: 187
- Dataset structure:
  - Demographic features: 24
  - Behavioral assessment features: 29
  - Engineered interaction features: 14
  - Connectome-derived features: 120

## 6. Validation Approach

To validate our preprocessing and feature engineering pipeline:
- Implemented 5-fold stratified cross-validation
- Ensured preprocessing steps were fitted only on training data to prevent data leakage
- Tested different preprocessing configurations to identify optimal approach

## 7. Implementation Details

All preprocessing and feature engineering steps have been implemented in the `feature_engineering.py` module with the following functions:
- `handle_missing_values()`: Implements imputation strategies
- `encode_categorical_features()`: Handles all categorical encoding
- `scale_numerical_features()`: Implements appropriate scaling
- `engineer_behavioral_features()`: Creates interaction terms and transformations
- `engineer_connectome_features()`: Extracts features from connectome data
- `select_features()`: Implements feature selection

The pipeline has been configured to handle both training and test data consistently while preventing data leakage. 