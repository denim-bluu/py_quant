import numpy as np

from util.array import shift


def calculate_std(x: np.ndarray, dof: int = 1) -> np.ndarray:
    return np.sqrt(np.sum((x - np.mean(x)) ** 2) / (len(x) - dof))


def calculate_daily_return(ts: np.ndarray) -> np.ndarray:
    """
    The function calculates the daily return of a time series.

    Args:
      ts (npt.NDArray[np.float_]): A numpy array containing the time series data.

    Returns:
      the daily return of a time series.
    """
    return ts / shift(ts, 1) - 1


def calculate_rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    roll_mean = np.empty(len(x))
    for i, _ in enumerate(x):
        if i < window:
            roll_mean[i] = np.mean(x[: i + 1])
        else:
            roll_mean[i] = np.mean(x[i - window : i])
    return roll_mean


def calculate_rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    roll_std = np.empty(len(x))
    for i, _ in enumerate(x):
        if i == 0:
            roll_std[i] = np.nan
        elif i < window:
            roll_std[i] = calculate_std(x[: i + 1])
        else:
            roll_std[i] = calculate_std(x[i - window : i])
    return roll_std
