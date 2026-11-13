# MSCS 634 – Lab 4: Regression Analysis with Regularization

## Purpose

This lab uses the **Diabetes dataset** from `sklearn.datasets` (442 patients, 10 baseline health measurements) to predict disease progression one year after baseline. It builds and compares several regression techniques:

- **Simple Linear Regression** (one feature: BMI)
- **Multiple Linear Regression** (all 10 features)
- **Polynomial Regression** (BMI with degrees 1–15, and all features with degrees 1–3)
- **Ridge (L2)** and **Lasso (L1)** regression, on both the original features and degree-2 polynomial features

Each model is evaluated on the same 80/20 train/test split using **MAE, MSE, RMSE and R²**, and the results are visualized with actual-vs-predicted plots, residual plots, coefficient paths and train-vs-test error curves.

## Key Insights

1. **Single-feature models underfit.** BMI is the strongest single predictor (r = 0.59), but alone it explains only ~23% of the variance. Using all 10 features roughly doubles R² to ~0.45.
2. **Polynomial features overfit quickly on small data.** With all features, degree 2 creates 65 features and test R² drops; degree 3 creates 285 features for only 353 training rows. Training R² rises to 0.88 while test R² falls to −13.7. On BMI alone, test error explodes above degree 10.
3. **Regularization controls overfitting.** Ridge and Lasso, with α tuned by 5-fold cross-validation on the training set, turned the overfitting degree-2 model into the **best models in the lab** (R² ≈ 0.50–0.51). Lasso + degree 2 also had the smallest train–test gap (≈ 0.01).
4. **Alpha controls the bias–variance trade-off.** Small α behaves like ordinary least squares (risk of overfitting). Large α shrinks coefficients toward zero (underfitting). Ridge shrinks all coefficients smoothly. Lasso sets them to exactly zero, so it also selects features: it kept only 13 of the 65 polynomial terms.
5. **Multicollinearity matters.** The cholesterol features `s1` and `s2` are 0.90 correlated, which gave them large opposite-sign coefficients in plain multiple regression. Lasso removed `s2`, shrank `s1`, and gave HDL (`s3`) a negative coefficient that matches its protective relationship with the target.
6. **Dataset insights.** BMI, `s5` (log serum triglycerides) and blood pressure are the most important predictors, and HDL cholesterol has a protective effect. Even the best model explains only about half of the variance, which suggests disease progression depends on factors that these 10 baseline measurements don't capture.

## Challenges and Decisions

- **Pre-scaled data:** scikit-learn's Diabetes features are already mean-centered and scaled, so raw values (e.g. `age`) aren't in their original units. No extra scaling was needed for the linear models. For polynomial features, a `StandardScaler` was added inside a pipeline so the penalty treats all terms equally.
- **Data cleaning:** the dataset had no missing values or duplicates. The IQR rule flagged a few outliers in the serum measurements. These are realistic clinical values and the dataset is small, so they were **kept** rather than removed.
- **Fair comparison:** one fixed train/test split (`random_state=42`) is used for every model. Alpha is tuned with cross-validation on the **training data only**, so the test set is never used to make choices.
- **Choice of feature for simple regression:** BMI was chosen because it has the highest correlation with the target.
- **Visualization scale:** the degree-3 polynomial model's error is so large that it is left out of the comparison bar charts (its numbers are still reported) so that the differences between the other models stay visible.
- **Small test set:** with only 89 test rows, small differences between the top models could change with a different split. 5-fold cross-validation was included as a robustness check.

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook MSCS_634_Lab_4.ipynb
```
Then run all cells (*Kernel → Restart & Run All*).
