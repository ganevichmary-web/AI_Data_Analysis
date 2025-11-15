
# Customer Purchase Analysis Report

## Overview
This report summarizes an analysis of factors influencing product purchase decisions using a dataset with customer demographics, reviews, and behavioral features.

## Methodology
- Cleaned the data, standardized columns, and converted target to binary.
- Engineered additional text-based features (length, word count, exclamation count, uppercase ratio).
- Built preprocessing pipelines for numeric, categorical, and text features.
- Trained a Random Forest classifier.
- Extracted feature importance using permutation importance.

## Results
Top features influencing purchase:
1. Education level (specifically *PG* category)
2. Gender (Male/Female effects)
3. Text review–related features had low predictive power
4. Age showed minor influence

## Insights
- Educational background appears most strongly associated with the likelihood of purchase.
- Gender-based differences also contribute somewhat.
- Text reviews in this dataset provide minimal predictive signal.
- To improve future modeling, consider collecting richer behavioral data, clearer text reviews, or product interaction metrics.
