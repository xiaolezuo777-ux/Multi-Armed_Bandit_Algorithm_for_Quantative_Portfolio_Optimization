# Multi-Armed_Bandit_Algorithm_for_Quantative_Portfolio_Optimization
Portfolio optimization using multi-armed bandit algorithms for adaptive asset allocation and risk-adjusted investment strategies.
Multi-Armed Bandits for Quantitative Portfolio Optimization

An adaptive portfolio optimization framework combining Fuzzy C-Means (FCM), Upper Confidence Bound (UCB) multi-armed bandits, and Genetic Algorithms (GA) for dynamic asset selection and portfolio weight optimization.

📄 Published Paper: Application of Multi-Armed Bandit Algorithm in Quantitative Finance
ITM Web of Conferences, Vol. 73, 01011 (2025)
URL: https://doi.org/10.1051/itmconf/20257301011

Overview

Traditional portfolio optimization methods often rely on static assumptions about asset returns and risk, which may be less effective under changing market conditions.

This project explores an adaptive portfolio construction framework based on multi-armed bandit algorithms. The framework first uses Fuzzy C-Means clustering to group assets according to their return and volatility characteristics, then applies UCB to dynamically select promising asset clusters.

To further optimize portfolio allocation, two hybrid approaches are investigated:

* UCB-GA — integrates UCB-based information into the Genetic Algorithm’s portfolio weight optimization process.
* GA-UCB — uses a Genetic Algorithm to optimize asset-specific UCB exploration parameters.

The goal is to dynamically adapt asset selection and portfolio weights while improving risk-adjusted performance.

Methodology

The overall pipeline consists of three main stages:

1. Asset Clustering — Fuzzy C-Means

Assets are clustered using average daily return and volatility as features.

The elbow method is used to determine the number of clusters, resulting in six asset clusters with different risk-return characteristics.

2. Dynamic Cluster Selection — Multi-Armed Bandit

The Upper Confidence Bound (UCB) algorithm is used to dynamically evaluate the asset clusters.

Instead of selecting a portfolio solely from historical average return and volatility, UCB balances:

* Exploitation — selecting clusters with strong observed performance
* Exploration — continuing to evaluate potentially better alternatives

The experiments show that UCB selects Cluster 6, while the static FCM analysis favors Cluster 3.

3. Portfolio Weight Optimization

Three portfolio allocation approaches are compared:

UCB

UCB rewards are normalized to generate portfolio weights dynamically.

UCB-GA

A Genetic Algorithm searches for portfolio weight vectors, while UCB-based information is incorporated into the optimization and selection process.

GA-UCB

Instead of directly optimizing portfolio weights, the Genetic Algorithm optimizes asset-specific UCB exploration coefficients.

These coefficients allow different assets to receive different exploration levels based on their observed performance. The resulting UCB rewards are then converted into portfolio weights.

Data

The experiments use historical financial data from Yahoo Finance covering 120 assets, including:

* Stocks
* Funds
* Bonds

Portfolio performance is evaluated using historical returns and risk metrics.

Evaluation Metrics

The strategies are evaluated using:

* Cumulative Return
* Sortino Ratio
* Maximum Drawdown
* Portfolio Volatility

These metrics measure both portfolio growth and downside risk.

Results

UCB vs. FCM Selection

UCB dynamically selected a different asset cluster from the one suggested by static FCM analysis.

Selection Method	Sortino Ratio	Maximum Drawdown
FCM-selected Cluster	-0.24	-22%
UCB-selected Cluster	1.60	-30%

The UCB-selected cluster achieved substantially higher cumulative returns and improved the Sortino Ratio by 1.18 compared with the cluster selected from FCM characteristics alone.

Portfolio Optimization

Algorithm	Sortino Ratio	Maximum Drawdown
UCB	1.60	-30%
UCB-GA	1.70	-27%
GA-UCB	3.23	-26%

Among the three strategies, GA-UCB achieved the strongest overall performance.

The GA-UCB portfolio reached approximately 250% cumulative return in the experiment and achieved a Sortino Ratio of 3.23, compared with 1.70 for UCB-GA and 1.60 for standard UCB.

GA-UCB also reduced maximum drawdown to approximately 26%, compared with 27% for UCB-GA and 30% for UCB.

Key Takeaways

The experiments suggest that:

* Multi-armed bandits can provide a dynamic alternative to static portfolio selection.
* UCB can adapt asset selection according to observed market performance rather than relying only on historical cluster characteristics.
* Genetic Algorithms can improve UCB-based portfolio construction by optimizing asset-specific exploration parameters.
* The proposed GA-UCB approach achieved the strongest risk-adjusted performance among the tested strategies.


Technologies

Python · NumPy · Pandas · Scikit-learn · Fuzzy C-Means · Multi-Armed Bandits · Genetic Algorithms · Quantitative Finance

Publication

Application of Multi-Armed Bandit Algorithm in Quantitative Finance

Chengxun Chen, Xuanyuan Liu, Yanyan Ma, and Xiaole Zuo

ITM Web of Conferences, Volume 73, 01011, 2025.

DOI: 10.1051/itmconf/20257301011

Limitations

The current backtest does not incorporate transaction costs. In addition, dividing the dataset into multiple formation and testing periods reduces the amount of data available within each period, which may affect generalization across different market environments.

Future work could incorporate transaction costs, larger datasets, and additional adaptive parameters to improve robustness across different market regimes.

Disclaimer

This repository is intended for research and educational purposes only and does not constitute financial advice.
