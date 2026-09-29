# Insurance Charges Prediction

An end-to-end Machine Learning regression project that predicts individual medical insurance charges from demographic and lifestyle features.

## Project Overview

This project follows a complete ML workflow:

1. Data loading and understanding
2. Data cleaning and duplicate removal
3. Exploratory Data Analysis (EDA)
4. Train/test split
5. Manual one-hot encoding
6. Baseline model experimentation
7. 5-fold cross-validation
8. Hyperparameter tuning with GridSearchCV
9. Final evaluation on an untouched test set
10. Residual/error analysis
11. Feature importance analysis
12. Streamlit deployment

## Dataset

The dataset contains 1,338 original records and these features:

- age
- sex
- bmi
- children
- smoker
- region
- charges (target)

One duplicate row was removed, leaving 1,337 records for modeling.

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Hyperparameter Tuning

The strongest tuned models were compared using 5-fold cross-validation.

| Model | Best CV R² |
|---|---:|
| XGBoost | 0.846782 |
| Gradient Boosting | 0.846479 |
| Random Forest | 0.834839 |

The final held-out test-set evaluation was then performed on the tuned models.

## Final Test Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Gradient Boosting | ₹2,449.85 | ₹4,233.48 | 0.902467 |
| XGBoost | ₹2,463.54 | ₹4,240.26 | 0.902154 |
| Random Forest | ₹2,790.21 | ₹4,691.70 | 0.880210 |

The tuned Gradient Boosting Regressor was carried forward as the final model.

### Final Model Performance

- **R²:** 0.902467
- **MAE:** ₹2,449.85
- **RMSE:** ₹4,233.48

R² is a regression metric and should not be described as classification accuracy.

## Feature Importance

The final Gradient Boosting model's impurity-based feature importance was:

| Feature | Importance |
|---|---:|
| smoker_yes | 0.690491 |
| bmi | 0.173258 |
| age | 0.122369 |
| children | 0.012150 |
| region_southwest | 0.000852 |
| sex_male | 0.000414 |
| region_southeast | 0.000292 |
| region_northwest | 0.000173 |

These values describe how the trained tree ensemble used the features for splitting; they do not establish causation.

## Error Analysis

Residual analysis showed that most predictions were relatively close to the actual charges, while some high-cost cases produced large positive residuals, indicating underprediction.

The model also showed similar mean absolute error for smokers and non-smokers in the held-out test set:

- Non-smokers: ₹2,447.91 MAE
- Smokers: ₹2,456.55 MAE

## Streamlit Application

The project includes a Streamlit interface where users can enter:

- Age
- Sex
- BMI
- Number of children
- Smoker status
- Region

The application loads the saved Gradient Boosting model and returns an estimated insurance charge.

## Project Structure

```text
Insurance-Charges-Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    ├── insurance_gradient_boosting_model.pkl
    └── insurance_feature_columns.pkl
```

## Run Locally

Create/activate your Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## Important

The `.pkl` model files are generated from the training workflow. Keep the model and feature-column files in the `models/` directory when running the Streamlit application.

## Disclaimer

This application provides an ML-based estimate from the available dataset features. It is a demonstration/portfolio project and not a medical or financial decision-making tool.
