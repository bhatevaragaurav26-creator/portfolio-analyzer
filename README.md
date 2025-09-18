# Portfolio Analyzer

Analyze any stock or portfolio:

* Returns \& risk (CAGR, Vol, Sharpe, Beta)
* Sector allocation
* Monte Carlo simulation
* Multi-portfolio CSV (amounts + counts)
* Currency display (USD/GBP/EUR, etc.)

## Run locally

python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py



\# Portfolio Analyzer



Streamlit app for analyzing any stock/portfolio:

\- Returns \& Risk (CAGR, Volatility, Sharpe, Beta vs benchmark)

\- Sector allocation

\- Per-ticker currency conversion to chosen display currency (USD/GBP/EUR/INR, etc.)

\- Monte Carlo simulation (GBM)

\- CSV/Excel downloads



\## Quick start (local)



```bash

python -m venv venv

\# Windows PowerShell:

\# .\\venv\\Scripts\\Activate.ps1  (or .\\venv\\Scripts\\activate.bat if policy blocks)

pip install -r requirements.txt

python -m streamlit run app.py



