import numpy as np
from numpy import typing as npt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import pandas as pd


def johansen_test(
    price_matrix: npt.NDArray[np.float_], det_order=0, k_ar_diff=1, verbose=True
):
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
    evec = johansen_test(
        price_matrix, det_order=det_order, k_ar_diff=k_ar_diff, verbose=False
    )
    return np.dot(price_matrix, np.atleast_2d((evec / evec[0])[:, 0]).T).flatten()
