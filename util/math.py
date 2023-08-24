import numpy as np

from util.array import shift


def calculate_std(x: np.ndarray, dof: int = 1) -> np.ndarray:
    """
    The function calculates the standard deviation of an array, with an optional parameter
    for degrees of freedom.
    
    Args:
      x (np.ndarray): An array of values for which you want to calculate the standard
    deviation.
      dof (int): The parameter "dof" stands for "degrees of freedom". It is an optional
    parameter with a default value of 1. Degrees of freedom is a concept used in statistics
    to determine the number of values in a calculation that are free to vary. In this
    context, it is used to adjust the. Defaults to 1
    
    Returns:
      the standard deviation of the input array `x`.
    """
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
    """
    The function calculates the rolling mean of an array using a specified window size.
    
    Args:
      x (np.ndarray): An array of numbers for which we want to calculate the rolling mean.
      window (int): The `window` parameter represents the size of the rolling window. It
    determines the number of elements to include in each rolling mean calculation.
    
    Returns:
      an array of rolling means calculated from the input array x.
    """
    roll_mean = np.empty(len(x))
    for i, _ in enumerate(x):
        if i < window:
            roll_mean[i] = np.mean(x[: i + 1])
        else:
            roll_mean[i] = np.mean(x[i - window : i])
    return roll_mean


def calculate_rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """
    The function calculates the rolling standard deviation of an array using a specified
    window size.
    
    Args:
      x (np.ndarray): An array of values for which we want to calculate the rolling standard
    deviation.
      window (int): The "window" parameter represents the size of the rolling window. It
    determines the number of elements to consider when calculating the rolling standard
    deviation.
    
    Returns:
      an array of rolling standard deviations.
    """
    roll_std = np.empty(len(x))
    for i, _ in enumerate(x):
        if i == 0:
            roll_std[i] = np.nan
        elif i < window:
            roll_std[i] = calculate_std(x[: i + 1])
        else:
            roll_std[i] = calculate_std(x[i - window : i])
    return roll_std
