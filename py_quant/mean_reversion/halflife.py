import numpy as np
from numpy import typing as npt


def calculate_half_life_mean_revert(
    ts: npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """
    The function calculates the half-life of mean reversion for a given time series.

    Args:
      ts (npt.NDArray[np.float_]): A numpy array containing the time series data.

    Returns:
      the half-life of mean reversion for a given time series.
    """
    delta = np.diff(ts)
    x = np.vstack((np.ones_like(ts[:-1]), ts[:-1])).T
    coef = np.linalg.lstsq(x, delta, rcond=None)[0][-1]
    return -np.log(2) / coef