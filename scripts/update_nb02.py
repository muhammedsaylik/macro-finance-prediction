"""Build a complete, runnable notebook 02 from scratch."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_feature_validation_and_modeling.ipynb"


def lines(text: str):
    return [line + "\n" for line in text.split("\n")]


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


SETUP = """%load_ext autoreload
%autoreload 2

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

is_nb = 'notebooks' in os.getcwd()
PROJECT_ROOT = os.path.abspath('..' if is_nb else '.')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'reports', 'figures')
sys.path.append(PROJECT_ROOT)
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'grid.color': '#e1e4e8',
    'grid.linestyle': '--',
    'grid.linewidth': 0.6,
    'axes.edgecolor': '#d1d5da',
    'axes.labelcolor': '#24292e',
    'xtick.color': '#586069',
    'ytick.color': '#586069',
    'text.color': '#24292e'
})

from src.features import FeaturePipeline
from src.feature_eng import FeatureEngineer, REDUNDANT_MACRO_BASES
from src.models import MacroFinanceModels
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

print('Notebook environment configured.')"""


def main():
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": [
            md(
                "# 02 Feature Validation and Modeling\n"
                "This notebook validates engineered features from the pipeline, performs multicollinearity and "
                "target-correlation diagnostics, trains forecasting models, and produces explainability outputs."
            ),
            code(SETUP),
            md("## 2.1 Pipeline Execution and Engineered Feature Set\nRun ingestion and feature engineering, then inspect the processed dataset."),
            code(
                """pipeline = FeaturePipeline()
df_processed = pipeline.run_pipeline(start_date='2000-01-01')

print('Processed data shape:', df_processed.shape)
display(df_processed.head())
print('Processed dataset path:', pipeline.processed_file_path())"""
            ),
            md("## 2.2 Multicollinearity Diagnostics\nEvaluate feature multicollinearity with VIF and identify redundant predictors."),
            code(
                """excluded_cols = ['FEDFUNDS', 'CPIAUCSL', 'T10Y2Y', 'UNRATE', 'SP500', 'SP500_log', 'target_next_ret']
feature_cols = [col for col in df_processed.columns if col not in excluded_cols]
X_vif = df_processed[feature_cols].copy()
X_vif = add_constant(X_vif)

vif_data = pd.DataFrame({
    'Feature': X_vif.columns,
    'VIF': [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
})
vif_results = vif_data[vif_data['Feature'] != 'const'].sort_values(by='VIF', ascending=False)

print('Top 10 VIF values:')
display(vif_results.head(10))
print('Safe predictors (VIF < 5):')
display(vif_results[vif_results['VIF'] < 5.0])"""
            ),
            md("## 2.3 Target Correlation and Predictor Ranking\nRank engineered predictors by absolute correlation with the one-period-ahead return target."),
            code(
                """correlations = df_processed[feature_cols].apply(lambda x: x.corr(df_processed['target_next_ret']))
corr_df = pd.DataFrame({
    'Feature': correlations.index,
    'Correlation': correlations.values,
})
corr_df['AbsCorrelation'] = corr_df['Correlation'].abs()
corr_df = corr_df.sort_values(by='AbsCorrelation', ascending=False).reset_index(drop=True)

print('Top 15 predictors by absolute correlation with target:')
display(corr_df.head(15))"""
            ),
            md("## 2.4 Refined Feature Set for Modeling\nRemove redundant features flagged by VIF and domain logic, then re-evaluate the candidate set."),
            code(
                """redundant_cols = []
for base in REDUNDANT_MACRO_BASES:
    redundant_cols.append(base)
    for lag in [1, 2, 3]:
        redundant_cols.append(f'{base}_lag{lag}')

refined_features = [col for col in feature_cols if col not in redundant_cols]
X_refined = df_processed[refined_features].copy()
X_refined = add_constant(X_refined)

vif_refined = pd.DataFrame({
    'Feature': X_refined.columns,
    'VIF': [variance_inflation_factor(X_refined.values, i) for i in range(X_refined.shape[1])]
})
vif_refined = vif_refined[vif_refined['Feature'] != 'const'].sort_values(by='VIF', ascending=False)

print('Refined feature set VIF analysis:')
display(vif_refined)"""
            ),
            md("## 2.5 Modeling Preparation\nPrepare time-ordered training and test splits from the engineered dataset, preserving temporal integrity."),
            code(
                """engineer = FeatureEngineer()
X_train, X_test, y_train, y_test, feature_cols, scaler = engineer.prepare_modeling_data(test_size=0.2)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)
print('Feature count:', len(feature_cols))
print('Top features for modeling:')
display(pd.DataFrame({'Feature': feature_cols}).head(20))"""
            ),
            md("## 2.6 Model Training and Evaluation\nTrain baseline and regularized regressors with time-series cross-validation."),
            code(
                """model_pipeline = MacroFinanceModels(random_state=42)
df_metrics, df_predictions = model_pipeline.fit_and_evaluate(X_train, X_test, y_train, y_test)

print('=== Predictive Performance Matrix ===')
display(df_metrics.round(5))
print('Prediction columns:', list(df_predictions.columns))"""
            ),
            md("## 2.7 Out-of-Sample Forecast Comparison\nVisualize one-month-ahead return predictions against realized values."),
            code(
                """plt.figure(figsize=(14, 6), dpi=300)

plt.plot(df_predictions.index, df_predictions['Actual'], label='Actual S&P 500 Return', color='#24292e', linewidth=1.5, alpha=0.7)
plt.plot(df_predictions.index, df_predictions['OLS_Pred'], label='OLS Prediction', color='#0366d6', linestyle='--', linewidth=1.2)
plt.plot(df_predictions.index, df_predictions['RidgeCV_Pred'], label='RidgeCV Prediction', color='#2b7a78', linestyle=':', linewidth=1.2)
plt.plot(df_predictions.index, df_predictions['MLP_Pred'], label='MLP (Neural Net) Prediction', color='#d9534f', linewidth=1.5)

plt.title('Out-of-Sample Prediction Comparison: One-Month Ahead S&P 500 Log Returns', loc='left', fontsize=12, fontweight='bold', color='#24292e', pad=12)
plt.xlabel('Timeline', fontsize=10, labelpad=8)
plt.ylabel('Logarithmic Return', fontsize=10, labelpad=8)
plt.legend(loc='upper left', frameon=True, facecolor='#fafbfc', edgecolor='#e1e4e8')
plt.grid(True, color='#e1e4e8', linestyle='--', linewidth=0.6)

os.makedirs(FIGURES_DIR, exist_ok=True)
plt.savefig(os.path.join(FIGURES_DIR, 'model_predictions_comparison.png'), bbox_inches='tight', dpi=300, facecolor='#ffffff')
plt.show()"""
            ),
            md("## 2.8 Residual Diagnostics\nInspect forecast errors for linear models to detect systematic bias."),
            code(
                """fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
fig.patch.set_facecolor('#ffffff')

for ax, model_name, color in zip(axes, ['RidgeCV', 'OLS'], ['#2b7a78', '#0366d6']):
    pred_col = f'{model_name}_Pred'
    residuals = df_predictions['Actual'] - df_predictions[pred_col]
    ax.scatter(df_predictions.index, residuals, color=color, alpha=0.75, s=28, edgecolors='none')
    ax.axhline(0, color='#24292e', linewidth=1.0, linestyle='--', alpha=0.6)
    ax.set_title(f'{model_name} Residuals (Out-of-Sample)', loc='left', fontsize=11, fontweight='semibold', color='#24292e')
    ax.set_xlabel('Timeline', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.grid(True, color='#e1e4e8', linestyle='--', linewidth=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'model_forecast_residuals.png'), bbox_inches='tight', dpi=300, facecolor='#ffffff')
plt.show()"""
            ),
            md("## 2.9 Explainable AI (XAI)\nRank macro-finance drivers via permutation importance on the MLP model."),
            code(
                """from sklearn.inspection import permutation_importance

mlp_model = model_pipeline.trained_models['MLP']
result = permutation_importance(
    mlp_model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1,
)

df_xai = pd.DataFrame({
    'Feature': feature_cols,
    'Importance_Mean': result.importances_mean,
    'Importance_Std': result.importances_std,
}).sort_values(by='Importance_Mean', ascending=False).reset_index(drop=True)

print('=== XAI: MLP Feature Importance (Top 10 Drivers) ===')
display(df_xai.head(10).round(5))"""
            ),
            code(
                """plt.figure(figsize=(10, 6), dpi=300)

top_xai = df_xai.head(10).sort_values(by='Importance_Mean', ascending=True)

plt.barh(top_xai['Feature'], top_xai['Importance_Mean'],
         xerr=top_xai['Importance_Std'],
         color='#0f4c81', alpha=0.85, edgecolor='#e1e4e8', height=0.6,
         error_kw={'ecolor': '#586069', 'linewidth': 1, 'capsize': 3})

plt.title('XAI Diagnostics: MLP Feature Importance via Permutation', loc='left', fontsize=12, fontweight='bold', color='#24292e', pad=15)
plt.xlabel('Drop in Predictive Accuracy (MSE Delta)', fontsize=10, labelpad=8)
plt.ylabel('Engineered Macro-Finance Features', fontsize=10, labelpad=8)
plt.grid(True, axis='x', color='#e1e4e8', linestyle='--', linewidth=0.6)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#d1d5da')
ax.spines['bottom'].set_color('#d1d5da')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'xai_feature_importance.png'), bbox_inches='tight', dpi=300, facecolor='#ffffff')
plt.show()"""
            ),
        ],
    }

    with NB_PATH.open("w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)

    print(f"Wrote complete notebook: {len(nb['cells'])} cells")


if __name__ == "__main__":
    main()
