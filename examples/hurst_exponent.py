import yfinance as yf

from py_quant.mean_reversion import hurst_exponent as he

if __name__ == "__main__":
    price = yf.download("CAD=X", start="2007-07-24", end="2012-03-28", progress=False)[
        "Adj Close"
    ].to_numpy()

    hurst_exponent1 = he.get_hurst_exponent_rs(price)
    print(f"Hurst Exponent with R/S analysis: {hurst_exponent1}")
    hurst_exponent2 = he.get_hurst_exponent(price, 20)
    print(f"Hurst Exponent: {hurst_exponent2}")
