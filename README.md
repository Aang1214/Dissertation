# Predicting Traffic Accident Severity in Brazil

A Multiclass Machine Learning Approach with Imbalance Handling

**MSc Data Analytics Dissertation**
Marek Martinak — D00250465 — Dundalk Institute of Technology

## Overview

This dissertation focuses on the severity of traffic accidents based on a publicly available dataset from the Federal Highway Police (PRF) for the period 2017–2023 and compares various machine learning models (logistic regression, decision tree, random forest, XGBoost, LightGBM, and CatBoost) to contribute to a reduction in the number of serious traffic accidents in Brazil.

The main difference lies in how to handle the number of classes being predicted and the amount of data in each case, using techniques to address class imbalance and applying SHAP analysis to identify important features. Previous studies focused only on binary classification. They did not address multiple classifications or the amount of data each class contains. Without considering these variables, it is possible to overlook important information that impacts the final results and potentially leads to the model making poor predictions.

The purpose of this project is to find a machine learning model capable of balancing the data and predicting these cases:

* *Without victims*

* *With injured victims*

* *With dead victims*

There is no requirement for participants to attend or help with this project.

## Repository structure

* data/              Raw PRF CSV and processed PRF CSV
* notebooks/         Jupyter notebooks for EDA and modelling phases
* scripts/           Python modules: data processing and model training
* models/            Trained models 
* outputs/           Figures, tables, dashboards
* docs/              Markdown documentation, meeting minutes, AI log

## Research Questions

**RQ1:** When it comes to predicting the multiclass severity of accidents using macro-F1 and per-class metrics, can tree-based classifiers (Decision Tree, Random Forest, XGBoost, LightGBM, and CatBoost) outperform a Logistic Regression baseline?
    
**RQ2:** How does handling class imbalance (using random oversampling, random undersampling, SMOTE, ADASYN, SMOTE+Tomek, and class weighting) affect the performance of these models, and how do the main methods of class imbalance handling compare?
    
**RQ3:** Which features are most important in predicting traffic accident severity?

## Status

**Completed**
- [x] EDA, cleaning, and feature engineering (May 2026)
- [x] Interim report submitted (7 June 2026)
- [x] Poster presentation (10–11 June 2026)

**In progress / upcoming**
- [x] Phase 1 — Finalise dataset, evaluation module (mid–late June 2026)
- [x] Phase 2 — Baseline modelling, 6 classifiers (late June 2026) → RQ1
- [x] *Three-week summer break (late June – mid July)*
- [x] Phase 3 — Imbalance grid: 7 methods × top 3 classifiers (mid–late July 2026) → RQ2
- [x] Phase 4 — Hyperparameter tuning (Optuna) on top 2–3 combinations (early August 2026) → RQ2
- [ ] Phase 5 — SHAP analysis on best tuned model (early–mid August 2026) → RQ3
- [ ] Phase 6 — Writing remaining chapters (late August – early September 2026)
- [ ] Final dissertation submission (early September 2026)

## Data source

- Kaggle dataset (used in this project): <https://www.kaggle.com/datasets/mlippo/car-accidents-in-brazil-2017-2023>

## Mahara

- Personal Mahara Portfolio: https://mahara.dkit.ie/view/view.php?t=c0f895f03955bfc95a86

## Generative AI usage

In accordance with DkIT Academic Integrity policy, generative AI was used to assist with the artefact only. The interim report text was written by the author. The full AI prompt file is available in docs/ai_prompts
