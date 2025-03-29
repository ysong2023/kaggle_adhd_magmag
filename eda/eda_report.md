# Exploratory Data Analysis Report - WiDS Datathon 2025

*Generated on 2025-03-20 03:50:52*

## 1. Dataset Overview

The dataset contains **1213** samples with:

- **18** quantitative features
- **9** categorical features
- Total of **27** features

### Data Columns Description

#### Quantitative Features:

- **MRI_Track_Age_at_Scan**: Age of the patient at the time of brain scan (years)
- **SDQ_SDQ_Emotional_Problems**: Score measuring emotional problems (anxiety, depression)
- **SDQ_SDQ_Conduct_Problems**: Score measuring behavioral issues (aggression, rule-breaking)
- **SDQ_SDQ_Hyperactivity**: Score measuring hyperactivity and attention problems
- **SDQ_SDQ_Peer_Problems**: Score measuring difficulties in peer relationships
- **SDQ_SDQ_Prosocial**: Score measuring positive social behaviors
- **SDQ_SDQ_Internalizing**: Combined score for emotional and peer problems
- **SDQ_SDQ_Externalizing**: Combined score for conduct and hyperactivity problems
- **SDQ_SDQ_Difficulties_Total**: Overall difficulties score from SDQ assessment
- **SDQ_SDQ_Generating_Impact**: Score measuring impact of difficulties on daily life
- **APQ_P_APQ_P_INV**: Parental Involvement score
- **APQ_P_APQ_P_PP**: Positive Parenting score
- **APQ_P_APQ_P_PM**: Poor Monitoring score
- **APQ_P_APQ_P_ID**: Inconsistent Discipline score
- **APQ_P_APQ_P_CP**: Corporal Punishment score
- **APQ_P_APQ_P_OPD**: Other Discipline Practices score
- **EHQ_EHQ_Total**: Environmental Health Questionnaire total score
- **ColorVision_CV_Score**: Color vision test score

#### Categorical Features:

- **PreInt_Demos_Child_Sex**: Biological sex of the child (Male/Female)
- **PreInt_Demos_Child_Race**: Racial background of the child
- **PreInt_Demos_Fam_Child_Ethnicity**: Ethnic background of the child
- **PreInt_Demos_Fam_Dwelling_Type**: Type of dwelling the family lives in
- **PreInt_Demos_Fam_Dwelling_Ownership**: Home ownership status
- **PreInt_Demos_Fam_Highest_Education**: Highest education level in family
- **PreInt_Demos_Fam_Household_Income**: Family income bracket
- **PreInt_Demos_Fam_Household_Structure**: Family structure (single parent, etc.)
- **Instrument_Version**: Version of assessment instruments used

### Missing Values

**Quantitative features with missing values:**

- MRI_Track_Age_at_Here is the initial EDA report:
- 
- [Paste your initial EDA report here]
- 
- Based on this report, please perform the following tasks:
- 
- 1. **Enhance the EDA Report:**
- * **Data Column Descriptions:**
- * We have 19 numerical columns and 10 categorical columns.
- * Enhance the `eda_report.md` by listing all data columns (numerical and categorical) and providing a reasonable guess of what each column represents.
- * **Target Variable Correction:**
- * Correct the analysis of the `ADHD_Outcome` variable. It should be treated as a categorical variable, not numerical.
- * Exclude any data that does not fit the categorical definition.
- * **Detailed Analysis:**
- * Provide a deeper analysis of the correlations between features and the target variable, including potential interactions.
- * Analyze the connectome data columns, and provide a summary of its structure.
- 2. **Data Preprocessing and Feature Engineering:**
- * Perform data preprocessing steps based on the enhanced EDA, including:
- * Handling missing values.
- * Encoding categorical variables.
- * Scaling numerical variables.
- * Perform feature engineering to create new features that may improve model performance.
- * **Generate Report:**
- * Create a `prep_feateng.md` report in the `notebooks/` directory, detailing the preprocessing and feature engineering steps performed.
- * Explain the choices made during preprocessing and feature engineering, and justify them based on the EDA findings.
- 
- Instructions:
- 
- * Update the `eda_report.md` in the `reports/` directory with the enhanced analysis.
- * Create the `prep_feateng.md` notebook in the `notebooks/` directory.
- * Ensure all code and analysis are well-documented and follow best practices.
- * Adhere to the project structure and any Kaggle competition rules.
- * Do not modify the original data files in `data/raw/`."Scan: 360 missing values (29.7%)

**Categorical features with missing values:**

- PreInt_Demos_Fam_Child_Ethnicity: 11 missing values (0.9%)

## 2. Target Variable Analysis

Target variable: **ADHD_Outcome**
Type: **Categorical (Binary)**

**Distribution:**

- 0 (No ADHD): 31.5% (382 samples)
- 1 (ADHD): 68.5% (831 samples)

Note: ADHD_Outcome was previously incorrectly treated as a numerical variable. We've corrected this to treat it as a binary categorical variable representing ADHD diagnosis status.

![Target Distribution](../reports\figures\target_distribution.png)

## 3. Feature Analysis

### 3.1 Numerical Features

![Numerical Feature Distributions](../reports\figures\numerical_distributions.png)

#### Feature Correlations with Target

**Top positively correlated features with ADHD diagnosis:**

- SDQ_SDQ_Hyperactivity: 0.5553 - Strong relationship between hyperactivity scores and ADHD diagnosis
- SDQ_SDQ_Externalizing: 0.5126 - Strong association between externalizing behaviors and ADHD
- SDQ_SDQ_Difficulties_Total: 0.4635 - Overall difficulties strongly linked to ADHD
- SDQ_SDQ_Generating_Impact: 0.4106 - Impact of difficulties associated with ADHD
- SDQ_SDQ_Conduct_Problems: 0.2770 - Moderate association between conduct issues and ADHD

**Top negatively correlated features:**

- SDQ_SDQ_Prosocial: -0.1669 - Higher prosocial scores slightly associated with lower ADHD probability
- APQ_P_APQ_P_INV: -0.0515 - Minor negative correlation with parental involvement

**Feature Interaction Analysis:**

- The combination of high SDQ_SDQ_Hyperactivity and SDQ_SDQ_Conduct_Problems appears to have a stronger relationship with ADHD than either feature alone
- SDQ_SDQ_Difficulties_Total shows high correlation with ADHD, but this is likely mediated by its component scores (especially hyperactivity)
- Parenting measures (APQ_P_*) show weaker correlations individually but may interact with other behavioral measures

![Correlation Heatmap](../reports\figures\correlation_heatmap.png)

### 3.2 Categorical Features

![Categorical Feature Distributions](../reports\figures\categorical_distributions.png)

**Categorical Feature Relationships with ADHD:**

- Gender appears to have some association with ADHD diagnosis (higher prevalence in males)
- Household structure shows some correlation with ADHD outcomes
- No strong relationships observed between ethnicity/race and ADHD diagnosis

## 4. Relationships between Features and Target

![Feature-Target Relationships](../reports\figures\feature_target_relationships.png)

## 5. Connectome Data Analysis

The connectome data contains **19901** columns representing functional brain connectivity measurements.

- Each column represents the strength of connection between two brain regions
- Column names follow the format "Region1_Region2" indicating connected brain areas
- Values represent correlation coefficients between brain regions' activity
- No missing values detected in the connectome data (0.0%)

**Structure:**

- Dense matrix representation of brain connectivity
- High dimensionality (19,901 features)
- Values typically range between -1 and 1
- Likely contains significant redundancy due to the symmetric nature of connectivity

## 6. Key Findings and Recommendations

### Key Findings:

1. **Missing Values**: Significant missing data in age variable (29.7%) and minimal missing data in ethnicity (0.9%)
2. **Strong Predictors**: Behavioral assessment scores, particularly hyperactivity and externalizing behaviors, show strong correlation with ADHD
3. **Target Imbalance**: The dataset has class imbalance with 68.5% ADHD cases and 31.5% non-ADHD cases
4. **Connectome Data**: The brain connectivity data contains 19,901 features, likely with high redundancy but potential for rich feature extraction

### Recommendations:

1. **Data Preprocessing**:
   - Implement missing value imputation strategies for age (MRI_Track_Age_at_Scan)
   - Consider stratified sampling or rebalancing techniques for the imbalanced target
   - Encode categorical variables appropriately
2. **Feature Engineering**:
   - Extract meaningful features from the connectome matrices
   - Consider creating interaction features between behavioral scores
   - Perform dimensionality reduction on connectome data
3. **Modeling Approach**:
   - Use classification models rather than regression
   - Consider ensemble methods to handle the complex relationships
   - Implement cross-validation with stratification to handle imbalance
