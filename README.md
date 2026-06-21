# Telco Customer Churn Prediction

A churn prediction project on subscription data, framed as a business problem rather than an accuracy contest. The headline finding: the optimal probability threshold for flagging at-risk customers depends almost entirely on the ratio of customer lifetime value to retention outreach cost — not on what maximizes accuracy.

## Problem

A telecom company has ~7,000 customers and a 26.5% annual churn rate. Given customer demographics, account information, and service subscriptions, predict who is likely to churn so retention efforts can be targeted.

Two characteristics make this harder than it looks. First, the class imbalance: a naive "no one churns" model achieves 73.5% accuracy by doing nothing, so accuracy is misleading from the start. Second, the two types of error are not equally costly — missing a churner loses thousands of dollars in lifetime value, while a false positive costs a small retention outreach. The business problem is not "maximize accuracy"; it is "minimize total expected cost given the LTV / retention cost ratio."

## Approach

- **Exploratory data analysis** with pandas — churn rates broken down by contract type, payment method, tenure, internet service, and add-on services. Four distinct business patterns surfaced before any model was built.
- **Data quality** — discovered and fixed a hidden type issue in `TotalCharges` (11 rows stored as blank strings rather than zeros for new customers with zero tenure).
- **Preprocessing** — one-hot encoding for categorical features, standard scaling for numeric features inside a sklearn Pipeline (to prevent train/test leakage during cross-validation).
- **Model comparison** — Logistic Regression, Random Forest, and XGBoost. Class imbalance addressed with `class_weight='balanced'` for the linear and tree models and `scale_pos_weight` for XGBoost.
- **Honest evaluation** — accuracy deliberately deprioritized in favor of recall, precision, and F1 on the churn class. 5-fold cross-validation with a Pipeline to confirm stability.
- **Threshold tuning with cost analysis** — moved beyond the default 0.5 threshold to compute the net business value of each operating point under stated assumptions about LTV and retention costs.

## Results

| Model | CV Recall (Churned) | CV F1 (Churned) |
|---|---|---|
| Logistic Regression (balanced) | **0.803 ± 0.025** | **0.627 ± 0.012** |
| Random Forest (balanced) | 0.50 | 0.56 |
| XGBoost (balanced) | 0.65 | 0.59 |

Best model: **Logistic Regression with balanced class weights**, achieving 80.3% recall on churners (5-fold CV, σ = 2.5%) against a baseline of 0% (the majority-class predictor catches no churners).

At a tuned probability threshold of 0.30 (chosen for business-cost reasons explained below), recall rises to ~92% on the test set, flagging 57% of customers as at-risk.

## Key Findings (from EDA)

- **Contract type is the single strongest churn driver.** Month-to-month customers churn at 42.7%, one-year at 11.3%, two-year at 2.8% — a 15x range. Moving customers from monthly to longer contracts is the largest available retention lever.
- **Electronic check is a churn-flagging payment method.** Electronic check customers churn at 45.3%; autopay customers churn at 15–17%. Friction at each payment moment creates regular exit points.
- **The loyalty cliff is real and steep.** New customers (0–12 months) churn at 47.4%, dropping to 9.5% for customers past 4 years. Retention spend is best concentrated in the first year.
- **Fiber optic is the highest-churn internet segment, despite being the premium product.** Fiber: 41.9% churn. DSL: 19.0%. Likely combination of price sensitivity and tech-savvy customer base that comparison-shops.
- **Protective services reduce churn far more than entertainment services.** OnlineSecurity and TechSupport subscribers churn ~27 points less than non-subscribers. Streaming services barely move the needle (~3-4 point gap). Services that "solve problems" build retention; services that "add entertainment" do not.

## The Business Framing (where most churn projects stop short)

Most churn analyses end at "the model has 80% accuracy." That number alone is not actionable. The model outputs a probability for every customer; the decision of *who to actually flag* is a policy choice that depends on business economics.

Under illustrative assumptions of $2,000 customer lifetime value, $50 retention outreach cost, and 30% retention success rate, expected net value across thresholds (test cohort of 1,409 customers):

| Threshold | Recall | Precision | Flagged | Estimated Net Value |
|---|---|---|---|---|
| 0.30 | 92.5% | 42.9% | 807 | **+$111K** |
| 0.40 | 86.6% | 46.5% | 697 | +$60K |
| 0.50 (default) | 78.6% | 50.7% | 580 | −$13K |
| 0.60 | 70.6% | 54.1% | 488 | −$86K |

The default threshold of 0.5 produces *negative* expected business value under these assumptions, because the cost of missed churners (40x retention cost in dollar terms) dominates the cost of false positives. A threshold of 0.30 — flagging more aggressively, accepting lower precision — is the value-maximizing choice.

The actual optimal threshold for any business depends on its specific LTV-to-retention-cost ratio. The framework here is reusable; the numbers are inputs.

## Why Logistic Regression (and Not the Tree Models)

Random Forest and XGBoost underperformed Logistic Regression on this dataset — by 13 to 30 percentage points on Churned recall, even after balancing. This is consistent with the data structure: 26 of 31 features are binary after one-hot encoding, leaving little non-linear structure for tree-based methods to exploit. Further tuning of tree models (grid search, more aggressive class weighting, SMOTE resampling) could plausibly close part of the gap, but was not pursued for three reasons:

1. The expected marginal gain is small relative to the engineering cost.
2. Tuning against a single test set without nested cross-validation risks overfitting in the validation step itself.
3. Logistic regression coefficients are directly interpretable, which is required for the downstream business recommendations. Tree-based models would require SHAP values or partial dependence plots to deliver the same explanatory quality.

The final model's coefficients align cleanly with the EDA findings: tenure and Contract length push strongly negative (less churn), Fiber and PaymentMethod_Electronic check push positive. A small number of coefficients show sign flips relative to univariate EDA (notably MonthlyCharges flips negative) — these reflect proper multivariate adjustment for correlated features, not a model error.

## Recommendations for Telco

If the analysis here were delivered to the business, three concrete actions would follow:

1. **Front-load retention spend on tenure < 12 months.** Almost half of new customers leave in their first year. Retention offers at month 3, 6, and 9 will move the needle far more than offers to long-tenured customers.
2. **Migrate electronic check customers to autopay.** Even small incentives ($5 off for a year, free service month) are cheaper than the churn cost of leaving these customers in friction-heavy payment flows.
3. **Audit the fiber optic experience.** Fiber is the highest-revenue and highest-churn segment simultaneously. Whether the cause is price, reliability, or competitive pressure determines the intervention — and it is worth identifying before further fiber acquisition spend.

## Limitations & Next Steps

- **LTV and retention cost are placeholders.** Real implementation requires Telco's actual financial inputs. The framework holds; specific dollar figures will shift the optimal threshold.
- **No customer-level retention success rate.** The model predicts churn but does not predict who can be saved. A second model — given churn risk, predict retention response — would refine the cost calculation significantly.
- **Feature multicollinearity.** `tenure`, `MonthlyCharges`, and `TotalCharges` carry overlapping information. Dropping `TotalCharges` (which is approximately `tenure × MonthlyCharges`) is worth testing.
- **No time-based validation.** Churn data is inherently temporal; a proper validation would train on earlier months and test on later ones, not random splits. The current evaluation is honest about its assumptions but not the strongest possible.
- **No tree-model hyperparameter tuning.** Justified for this project (see above), but worth revisiting if performance demands increase.

## Tech Stack

Python, pandas, scikit-learn, XGBoost, matplotlib, seaborn. Notebook developed in Google Colab.

## Files

- `telco_churn_analysis.ipynb` — full annotated notebook with EDA, preprocessing, modeling, evaluation, threshold analysis, and feature interpretation.
- `telco_churn.csv` — IBM Telco customer churn dataset (publicly available on Kaggle).

## About

I build practical ML solutions for prediction problems faced by small and mid-sized businesses — customer churn, demand forecasting, lead scoring, classification of unstructured text. Open to project work.

**Contact:** thaixingyun790@gmail.com · [LinkedIn](https://www.linkedin.com/in/shawn-liew-2ab65a80/)
