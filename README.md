# Employee turnover — quick README

Brief: reproducible notebook that analyzes an HR dataset (1.1K rows) and evaluates classifiers for predicting employee turnover.

Notebook: `employee_turnover/ml_pipeline.ipynb`

Quick start
- Clone the repo and open the notebook (Jupyter/Colab):

Data
- Expected local path: `employee_turnover/data/turnover.csv`.

Minimal requirements
- `requirements.txt` in `employee_turnover/`, and install with `pip install -r employee_turnover/requirements.txt` (recommended packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, torch).

Key results (test set)
- Random Forest: accuracy ≈ 0.73, F1 ≈ 0.74 (best)
- Logistic Regression: accuracy ≈ 0.67, F1 ≈ 0.67
- Small neural net: overfit (train F1 ≈ 0.74, test F1 ≈ 0.64)


