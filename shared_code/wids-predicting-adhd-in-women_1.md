# Improved ADHD Prediction Model

## Model Overview
This is an improved version of the ADHD prediction model for the WiDS Datathon 2025. The model focuses on predicting ADHD outcome in women by leveraging behavioral scores, demographic information, and brain connectivity data.

## Key Improvements

1. **Enhanced Feature Engineering**
   - Created squared terms for behavioral features (e.g., SDQ_SDQ_Hyperactivity_Squared)
   - Generated interaction terms between key behavioral metrics
   - Added ratios between related scores (e.g., Internalizing_to_Externalizing ratio)
   - Created composite parenting style features (Positive_Parenting, Negative_Parenting)
   - Incorporated age-related interactions with behavioral scores

2. **Advanced Modeling Approach**
   - Used LightGBM for sex prediction (instead of LogisticRegressionCV)
   - Implemented a stacking ensemble for ADHD prediction with:
     - Logistic Regression
     - Random Forest
     - Gradient Boosting
     - LightGBM as final estimator
   - Applied SMOTE for handling class imbalance
   - Optimized thresholds using cross-validation

3. **Improved Preprocessing**
   - Enhanced feature filtering to handle potential missing values
   - Added handling for feature interaction combinations
   - Applied more robust evaluation metrics

## Model Performance

### Sex Prediction
- F1 Score: 0.5496
- AUC: 0.5856
- Accuracy: 0.5505
- Precision: 0.5054
- Recall: 0.6040
- Optimal Threshold: 0.0606

### ADHD Prediction
- F1 Score: 0.8503
- AUC: 0.8043
- Accuracy: 0.7783
- Precision: 0.8481
- Recall: 0.8537
- Optimal Threshold: 0.2828

## Methodology

The approach incorporates a two-step modeling process:
1. First, predict the sex of participants using an advanced LightGBM model
2. Use this prediction to inform the ADHD prediction model through:
   - Direct inclusion of sex probability as a feature
   - Interaction terms between sex probability and key behavioral features
   - Ensemble modeling for robust predictions

This methodology acknowledges the sex-specific manifestations of ADHD and provides a more nuanced prediction approach compared to the baseline model.

## Submission

The final submission file contains predictions for 304 participants, with an estimated ADHD prevalence of 0.7895 and a female proportion of 1.0000.

Generated on: 2025-03-21 03:34:13
