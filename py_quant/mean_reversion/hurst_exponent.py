import numpy as np
from numpy import typing as npt
from util.array import calculate_daily_return


def get_hurst_exponent(price: npt.NDArray[np.float_], max_lag=20) -> float:
    """
    The function calculates the Hurst exponent of a time series
    method.

    Args:
      price (npt.NDArray[np.float_]): A time series data as a 1-dimensional numpy array of
    float values.
      max_lag: The `max_lag` parameter is the maximum lag value to consider when calculating
    the Hurst exponent. It determines the range of lags to use in the calculation. Defaults
    to 20

    Returns:
      the Hurst exponent, which is a measure of the long-term memory of a time series.
    """

    lags = range(2, max_lag)

    # variances of the lagged differences
    tau = [np.std(np.subtract(price[lag:], price[:-lag])) for lag in lags]

    # calculate the slope of the log plot -> the Hurst Exponent
    reg = np.polyfit(np.log(lags), np.log(tau), 1)

    return reg[0]


def get_hurst_exponent_rs(price: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    The function calculates the Hurst exponent using the R/S method for a given price
    series.

    Args:
      price (npt.NDArray[np.float_]): The parameter "price" is expected to be a
    1-dimensional numpy array of float values representing the price data.

    Returns:
      the Hurst exponent, which is calculated using the R/S method.
    """
    ret = calculate_daily_return(price)[1:]
    n = np.round(len(ret) / 2 ** np.arange(0, 10)).astype(int)

    y = []
    x = []

    for i in n:
        rs = []
        for t in range(0, round(len(ret) / i)):
            sub_ts = ret[t * i : (t + 1) * i]
            mean_adjusted = sub_ts - np.mean(sub_ts)
            cumul_mean_adjusted = np.cumsum(mean_adjusted)
            r = np.max(cumul_mean_adjusted) - np.min(cumul_mean_adjusted)
            s = np.std(sub_ts)
            rs.append(r / s)
        y.append(np.mean(rs))
        x.append(i)

    return np.polyfit(np.log(x), np.log(y), 1)[0]
