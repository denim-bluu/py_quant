import numpy as np
import yfinance as yf

from py_quant.mean_reversion import halflife as hl

if __name__ == "__main__":
    price = yf.download("CAD=X", start="2007-07-24", end="2012-03-28", progress=False)[
        "Adj Close"
    ].to_numpy()
    
    half_life = hl.calculate_half_life_mean_revert(price)
    print(f"Half-life of mean reversion: {half_life}")