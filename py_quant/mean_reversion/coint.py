import numpy as np
from numpy import typing as npt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd


def johansen_test(
    price_matrix: npt.NDArray[np.float_], det_order=0, k_ar_diff=1, verbose=True
):
    """
    The function `johansen_test` performs the Johansen cointegration test on a price matrix
    and returns the eigenvectors.

    Args:
        price_matrix (npt.NDArray[np.float_]): A numpy array containing the price data of
    multiple assets. Each row represents the prices of one asset over time.
        det_order: The `det_order` parameter specifies the order of deterministic terms to
    include in the Johansen test. It determines whether to include a constant term (0), a
    trend term (1), or both (2) in the test. Defaults to 0
        k_ar_diff: The parameter "k_ar_diff" represents the number of times the price series
    should be differenced to make it stationary. It is used in the Johansen test for
    cointegration. Defaults to 1
        verbose: The `verbose` parameter is a boolean flag that determines whether or not to
    print the output of the Johansen test. If `verbose` is set to `True`, the function will
    print the output, including the maximum eigenvalue statistic, trace statistic, and
    critical values. If `verbose` is. Defaults to True

    Returns:
      The function `johansen_test` returns the eigenvectors of the cointegration matrix.
    """
    res = coint_johansen(price_matrix, det_order=det_order, k_ar_diff=k_ar_diff)
    output = pd.DataFrame([res.lr2, res.lr1], index=["max_eig_stat", "trace_stat"])
    if verbose:
        print(output.T, "\n")
        print("Critical values(90%, 95%, 99%) of max_eig_stat\n", res.cvm, "\n")
        print("Critical values(90%, 95%, 99%) of trace_stat\n", res.cvt, "\n")
    return res.evec


def construct_cointegrating_portfolio(
    price_matrix: npt.NDArray[np.float_], det_order=0, k_ar_diff=1
):
    """
    The function constructs a cointegrating portfolio using the Johansen test and price
    matrix.

    Args:
        price_matrix (npt.NDArray[np.float_]): A numpy array of shape (n, m) where n is the
    number of observations and m is the number of assets. Each row represents the prices of
    the assets at a specific time.
        det_order: The det_order parameter specifies the order of differencing to be applied
    to the price matrix before conducting the Johansen test. It determines the number of
    times the price series needs to be differenced to make it stationary. The default value
    is 0, which means no differencing is applied. Defaults to 0
        k_ar_diff: The parameter `k_ar_diff` represents the number of lagged differences to be
    included in the model. It is used in the Johansen test for cointegration to determine
    the order of integration of the time series data. A higher value of `k_ar_diff`
    indicates a higher order of differ. Defaults to 1

    Returns:
        a flattened array that is the result of multiplying the price_matrix with the normalised
    eigenvector.
    """
    evec = johansen_test(
        price_matrix, det_order=det_order, k_ar_diff=k_ar_diff, verbose=False
    )
    return np.dot(price_matrix, np.atleast_2d((evec / evec[0])[:, 0]).T).flatten()
