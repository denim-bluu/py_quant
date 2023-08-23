import numpy as np
import yfinance as yf

from py_quant import hurst_exponent as he

if __name__ == "__main__":
    np.random.seed(42)
    random_changes = 1. + np.random.randn(99999) / 1000.
    series = np.cumprod(random_changes)  # create a random walk from random changes
    price = yf.download("CAD=X", start="2007-07-24", end="2012-03-28", progress=False)[
        "Adj Close"
    ].to_numpy()

    hurst_exponent1 = he.get_hurst_exponent_rs(series)
    print(f"Hurst Exponent with R/S analysis: {hurst_exponent1}")
    hurst_exponent2 = he.get_hurst_exponent(series, 20)
    print(f"Hurst Exponent: {hurst_exponent2}")