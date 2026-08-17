# Schonfeld Case-Study

This repository contains the code and resources for the **Schonfeld Case-Study**, developed by **Luis Felipe Lins**. 

The goal of this project is to build a complete research pipeline to process SEC 13F filings, construct positioning factors, and backtest trading strategies based on those factors.

## Table of Contents
- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
  - [1. Data Gathering and Engineering](#1-data-gathering-and-engineering)
  - [2. Universe Construction](#2-universe-construction)
  - [3. Factor Creation](#3-factor-creation)
  - [4. Backtesting Framework](#4-backtesting-framework)
- [Key Factors Analyzed](#key-factors-analyzed)
- [Installation and Requirements](#installation-and-requirements)
- [Usage](#usage)
- [Output](#output)

---

## Overview
The project is centered around the creation and evaluation of a new institutional positioning factor: **Centrality**. 

Centrality measures how similarly institutional managers weight a specific asset within their portfolios. A high centrality score implies significant deviation from the consensus (indicating a strong idiosyncratic bet), while a low centrality score implies universal agreement on the asset's weight. 

We benchmark this centrality measure against three standard positioning factors:
*   Net Buying
*   Breadth Change
*   Conviction

All strategies are tested on the universe of S&P 500 stocks across various configurations (rebalance frequency, transaction costs, and long-only vs. long-short implementations).

---

## Pipeline Architecture
The pipeline is designed to be fully reproducible from a cold start and is broken down into modular steps:

### 1. Data Gathering and Engineering (`get_data.py`)
*   **Download:** Automatically fetches bulk 13F zip files from the SEC EDGAR database.
*   **Parsing:** Extracts `SUBMISSION.TSV`, `COVERPAGE.TSV`, and `INFOTABLE.TSV`, retaining only relevant long equity positions.
*   **Unit Cleaning (`clean_units`):** Addresses the widespread issue of filers misreporting units (e.g., reporting in thousands instead of single dollars) by comparing implicit prices against an expanding median of peers. It also filters out bonds misreported as equities.
*   **Point-in-Time (PIT) Resolution:** Accurately applies `NEW HOLDINGS` and `RESTATEMENT` amendments to rebuild the historic reality exactly as it was known on any given vintage date, avoiding look-ahead bias.
*   **S&P 500 Mapping (`get_spx_composition`):** Reconstructs the historical daily membership of the S&P 500 and uses SEC Fails-to-Deliver data to map tickers to CUSIPs.

### 2. Universe Construction (`get_data.py`)
To isolate the signal from noise, the pipeline allows evaluating factors across four distinct institutional manager universes:
1.  **AUM:** Top 5% of filers by size.
2.  **CV_High:** Managers with the highest variance in position sizes (proxy for active/idiosyncratic betting).
3.  **CV_Low:** Managers with the lowest variance.
4.  **CONC:** Managers with the most concentrated portfolios (measured via the effective number of positions, based on HHI).

### 3. Factor Creation (`factors.py`)
*   Calculates `net_buying`, `breadth_change`, and `conviction`.
*   **`build_centrality`:** Computes the Centrality factor using Singular Value Decomposition (SVD) on a centered matrix of portfolio weights (Filers $\times$ CUSIPs). The leading singular vector represents the axis of greatest deviation from the consensus.

*   The whole process up to this point might take some 2-3 hours running. For that reason, I leave available the resulting data for download at [Google Drive](https://drive.google.com/drive/folders/16gCtgLBQNonoC3usDN6SiV92xpf0rzeB?usp=sharing).

### 4. Backtesting Framework (`trading_strategy.py`)
*   **`build_strategy`:** Constructs daily target portfolio weights based on the cross-sectional $Z$-score of the chosen factor. Incorporates strict Point-In-Time discipline by lagging fundamental data according to statutory 13F deadlines (45 days).
*   **`run_backtest`:** Simulates the portfolio's daily returns, accurately reflecting drift between rebalance dates and applying transaction costs to the traded notional.
*   **Grid Search:** Evaluates 384 combinations (4 universes $\times$ 4 factors $\times$ 3 frequencies $\times$ 2 sides $\times$ 4 cost tiers) to ensure robustness.

---

## Key Factors Analyzed
*   **Centrality:** The magnitude of an asset's loading on the leading principal component of the manager co-holding matrix.
*   **Net Buying:** The normalized change in shares held by the manager group over consecutive quarters.
*   **Breadth Change:** The change in the number of unique funds holding an asset.
*   **Conviction:** The average weight of an asset inside the portfolios of the managers that own it.

---

## Installation and Requirements
The pipeline requires Python 3.10+ and the following core dependencies (also listed in `requirements.txt`):
*   `pandas`
*   `numpy`
*   `scipy`
*   `requests`
*   `beautifulsoup4`
*   `pyarrow`
*   `yfinance`
*   `matplotlib`

To install the requirements, run:
```bash
pip install -r requirements.txt
```

## Usage

1. **Configuration:** Ensure `config.py` is properly set up to define your data directories (e.g., `DATA_ZIP`, `DATA_RAW`, `DATA_INT`, `DATA_FINAL`, `DATA_FACTORS`).
2. **Execution:** The entire pipeline can be executed by running the Jupyter Notebook provided in the repository. The notebook systematically calls the functions from `get_data.py`, `factors.py`, and `trading_strategy.py`.

*Note: The initial data download and unit-cleaning steps are computationally intensive and may take ~40-50 minutes on a cold start. Subsequent runs will bypass completed steps if the intermediate Parquet files exist.*

## Output

The pipeline generates several outputs in the `data/final/` directory:
*   `concentration.png`: A chart showing the concentration of 13F assets among the largest filers.
*   `coverage.png`: Diagnostics on CUSIP-to-ticker mapping coverage.
*   `backtest_grid.xlsx`: A comprehensive spreadsheet containing the performance metrics (Sharpe, Volatility, Max Drawdown, Turnover, etc.) for all 384 strategy configurations.
*   Factor analysis charts (e.g., `factor_deciles.png`, `risk_return.png`, `cumulative_returns.png`, `factor_correlation.png`).
