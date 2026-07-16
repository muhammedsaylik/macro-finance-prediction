<div align="center">

<!-- Institutional / Domain Indicators -->
<img src="https://img.shields.io/badge/Type-Independent%20Research-0d233a?style=flat-square"/>
<img src="https://img.shields.io/badge/JEL%20Classification-C53%2C%20G17%2C%20E44-475569?style=flat-square"/>
<img src="https://img.shields.io/badge/Status-Completed%20Research-238636?style=flat-square"/>

<br><br>

<h1 style="margin-bottom:12px; font-size:34px; font-weight:700; letter-spacing:-0.8px; color:#0d233a; font-family: 'Georgia', serif;">
Macroeconomic Information Content in Equity Return Forecasting
</h1>

<p style="font-size:15px; color:#475569; max-width:760px; line-height:1.65; margin:auto; font-family: 'Georgia', serif; font-style: italic;">
An Empirical Comparison of Financial Econometric and Machine Learning Models for One-Month-Ahead S&amp;P 500 Log Return Forecasting
</p>

<br>

<!-- Minimalist Tech Stack Badges -->
<img src="https://img.shields.io/badge/Python-3.13-24292e?style=flat-square&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Libraries-Pandas%20%7C%20NumPy-24292e?style=flat-square"/>
<img src="https://img.shields.io/badge/Econometrics-Statsmodels-24292e?style=flat-square"/>
<img src="https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost%20%7C%20LightGBM-24292e?style=flat-square"/>
<img src="https://img.shields.io/badge/XAI-SHAP%20%7C%20Permutation%20Importance-24292e?style=flat-square"/>
<img src="https://img.shields.io/badge/Visualization-Matplotlib%20%7C%20Seaborn-24292e?style=flat-square"/>

</div>

<br><br>

---

<p align="center" style="max-width:700px; margin:20px auto; font-family:'Georgia',serif; font-size:13.5px; font-style:italic; color:#475569; line-height:1.7;">
This study examines whether a small set of low-frequency macroeconomic indicators carries meaningful predictive information for next-month equity returns, benchmarking classical econometric estimators against machine learning regressors under a strict anti-leakage, out-of-sample evaluation protocol.
</p>

<!-- Academic/Journal Style Framework Matrix -->
<table align="center" style="border-collapse: collapse; border: none; width: 100%; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">
<!-- Top Thick Rule -->
<tr style="border-top: 2px solid #0d233a; border-bottom: 1px solid #e1e4e8; background: transparent;">
  <th width="25%" align="center" style="border: none; padding: 12px 10px; font-size: 12px; font-weight: 700; color: #0d233a; text-transform: uppercase; letter-spacing: 0.8px;">Research Objective</th>
  <th width="25%" align="center" style="border: none; padding: 12px 10px; font-size: 12px; font-weight: 700; color: #0d233a; text-transform: uppercase; letter-spacing: 0.8px;">Forecasting Target</th>
  <th width="25%" align="center" style="border: none; padding: 12px 10px; font-size: 12px; font-weight: 700; color: #0d233a; text-transform: uppercase; letter-spacing: 0.8px;">Methodology</th>
  <th width="25%" align="center" style="border: none; padding: 12px 10px; font-size: 12px; font-weight: 700; color: #0d233a; text-transform: uppercase; letter-spacing: 0.8px;">Data Ingestion</th>
</tr>

<!-- Content Row -->
<tr style="border: none; background: transparent;">

<td align="center" style="border: none; padding: 14px 10px; vertical-align: top;">
<p style="margin: 0; font-size: 13px; color: #24292e; line-height: 1.55; font-weight: 500;">
Evaluating how much predictive information a small set of macroeconomic indicators contains for one-month-ahead S&amp;P 500 log return forecasting.
</p>
</td>

<td align="center" style="border: none; padding: 14px 10px; vertical-align: top;">
<p style="margin: 0; font-size: 13px; color: #24292e; line-height: 1.55;">
One-month-ahead S&amp;P 500 (<code style="font-family: monospace; font-size: 11.5px; background: #f6f8fa; padding: 2px 4px; border-radius: 4px;">^GSPC</code>) logarithmic returns (<code style="font-family: monospace; font-size: 11.5px; background: #f6f8fa; padding: 2px 4px; border-radius: 4px;">target_next_ret</code>), carefully aligned to avoid look-ahead bias.
</p>
</td>

<td align="center" style="border: none; padding: 14px 10px; vertical-align: top;">
<p style="margin: 0; font-size: 13px; color: #24292e; line-height: 1.55;">
Out-of-sample forward validation benchmarking a naive baseline against linear regularized models (Ridge, ElasticNet), a neural network regressor (MLP), and gradient boosting models (XGBoost, LightGBM).
</p>
</td>

<td align="center" style="border: none; padding: 14px 10px; vertical-align: top;">
<p style="margin: 0; font-size: 13px; color: #24292e; line-height: 1.55;">
Monthly macroeconomic indicators retrieved from FRED (<code style="font-family: monospace; font-size: 11.5px;">FEDFUNDS</code>, <code style="font-family: monospace; font-size: 11.5px;">CPIAUCSL</code>, <code style="font-family: monospace; font-size: 11.5px;">UNRATE</code>, <code style="font-family: monospace; font-size: 11.5px;">T10Y2Y</code>) and synchronized with S&amp;P 500 market data obtained from Yahoo Finance.
</p>
</td>

</tr>
<!-- Bottom Thick Rule -->
<tr style="border-bottom: 2px solid #0d233a; background: transparent;"><td colspan="4" style="border: none; padding: 0;"></td></tr>
</table>

<br>

<!-- Journal-style Figure: framed research pipeline with Roman-numeral steps and academic caption -->
<div align="center" style="border-top: 1px solid #0d233a; border-bottom: 1px solid #0d233a; padding: 22px 18px 16px; font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif;">

<table align="center" style="border-collapse: collapse; border: none; width: 100%; max-width: 820px;">
<tr>

<td width="20%" align="center" style="border: none; padding: 0 10px; vertical-align: top;">
<p style="margin:0; font-family:'Georgia',serif; font-size:20px; font-weight:400; color:#0d233a;">I.</p>
<p style="margin:8px 0 0; font-size:11.5px; font-weight:600; color:#0d233a; letter-spacing:0.3px;">Data Ingestion</p>
<p style="margin:5px 0 0; font-size:11.5px; color:#64748b; line-height:1.5; font-style:italic;">FRED macro series &amp; S&amp;P 500 quotes, unified and time-aligned</p>
</td>

<td width="2%" style="border: none; border-left: 1px solid #cbd5e1; padding: 0;"></td>

<td width="20%" align="center" style="border: none; padding: 0 10px; vertical-align: top;">
<p style="margin:0; font-family:'Georgia',serif; font-size:20px; font-weight:400; color:#0d233a;">II.</p>
<p style="margin:8px 0 0; font-size:11.5px; font-weight:600; color:#0d233a; letter-spacing:0.3px;">Feature Engineering</p>
<p style="margin:5px 0 0; font-size:11.5px; color:#64748b; line-height:1.5; font-style:italic;">Log-returns, realized volatility, and momentum construction</p>
</td>

<td width="2%" style="border: none; border-left: 1px solid #cbd5e1; padding: 0;"></td>

<td width="20%" align="center" style="border: none; padding: 0 10px; vertical-align: top;">
<p style="margin:0; font-family:'Georgia',serif; font-size:20px; font-weight:400; color:#0d233a;">III.</p>
<p style="margin:8px 0 0; font-size:11.5px; font-weight:600; color:#0d233a; letter-spacing:0.3px;">Statistical Diagnostics</p>
<p style="margin:5px 0 0; font-size:11.5px; color:#64748b; line-height:1.5; font-style:italic;">Correlation, stationarity and multicollinearity diagnostics</p>
</td>

<td width="2%" style="border: none; border-left: 1px solid #cbd5e1; padding: 0;"></td>

<td width="20%" align="center" style="border: none; padding: 0 10px; vertical-align: top;">
<p style="margin:0; font-family:'Georgia',serif; font-size:20px; font-weight:400; color:#0d233a;">IV.</p>
<p style="margin:8px 0 0; font-size:11.5px; font-weight:600; color:#0d233a; letter-spacing:0.3px;">Estimation</p>
<p style="margin:5px 0 0; font-size:11.5px; color:#64748b; line-height:1.5; font-style:italic;">Naive baseline, OLS/Ridge/ElasticNet, MLP, and tree-ensemble regressors</p>
</td>

<td width="2%" style="border: none; border-left: 1px solid #cbd5e1; padding: 0;"></td>

<td width="20%" align="center" style="border: none; padding: 0 10px; vertical-align: top;">
<p style="margin:0; font-family:'Georgia',serif; font-size:20px; font-weight:400; color:#0d233a;">V.</p>
<p style="margin:8px 0 0; font-size:11.5px; font-weight:600; color:#0d233a; letter-spacing:0.3px;">Evaluation &amp; Interpretation</p>
<p style="margin:5px 0 0; font-size:11.5px; color:#64748b; line-height:1.5; font-style:italic;">Out-of-sample evaluation, feature importance and SHAP analysis</p>
</td>

</tr>
</table>

<p style="margin: 18px 0 0; font-family:'Georgia',serif; font-size:12.5px; font-style:italic; color:#475569;">
Figure 1. Overview of the research workflow, from macroeconomic data acquisition to out-of-sample model evaluation and interpretation.
</p>

</div>

<br>