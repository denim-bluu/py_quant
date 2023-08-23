import numpy as np
from numpy import typing as npt


def shift(arr: np.ndarray, num: int, fill_value=np.nan) -> np.ndarray:
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def calculate_daily_return(ts: npt.NDArray[np.float_]) -> npt.NDArray[np.float_]:
    """
    The function calculates the daily return of a time series.

    Args:
      ts (npt.NDArray[np.float_]): A numpy array containing the time series data.

    Returns:
      the daily return of a time series.
    """
    return ts / shift(ts, 1) - 1
