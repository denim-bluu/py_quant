import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from statsmodels.tsa.stattools import adfuller

from py_quant import coint

ige = yf.download("IGE", start="2006-04-26", end="2012-04-09", progress=False)[
    "Adj Close"
].to_numpy()
ewa = yf.download("EWA", start="2006-04-26", end="2012-04-09", progress=False)[
    "Adj Close"
].to_numpy()
ewc = yf.download("EWC", start="2006-04-26", end="2012-04-09", progress=False)[
    "Adj Close"
].to_numpy()

price_matrix = np.vstack((ewa, ewc, ige)).T

port = coint.construct_cointegrating_portfolio(price_matrix)

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(port)), y=port, mode="lines", name="lines"))

adfuller(port, regression='n')