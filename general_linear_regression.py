from typing import NamedTuple

import numpy as np
import pandas as pd

""" [GeneralLinearRegressionResults] represents the results of performing
    general linear regression on a design matrix [X], where each column
    represents an explanatory variable and each row represents a measurement
    of each of the explanatory variables, and a target matrix [y]. These
    results are the least squares solution to the equation [y = Xb], where
    [coefficients] represents [b], [coefficients_cov] represents the
    variance-covariance matrix of [b], [error] represents the error in
    prediction of [b], and [error_cov] is the variance-covariance matrix
    of [error]
"""


class GeneralLinearRegressionResults(NamedTuple):
    coefficients: pd.DataFrame
    coefficients_cov: pd.DataFrame
    error: pd.DataFrame
    error_cov: pd.DataFrame


""" [general_linear_regression(X_df, y_df, df_correction_error_cov)] represents
    the results of performing general linear regression on a design matrix
    [X_df], where each column represents an explanatory variable and each row
    represents a measurement of each of the explanatory variables, and a target
    matrix [y_df]. These results are the least squares solution to the equation
    [y_df = X_df * b], where [coefficients] represents [b], [coefficients_cov]
    represents the variance-covariance matrix of [b], [error] represents the
    error in prediction of [b], and [error_cov] is the variance-covariance 
    matrix of [error]
"""


def general_linear_regression(
    X_df: pd.DataFrame, y_df: pd.DataFrame, df_correction_error_cov=False
) -> GeneralLinearRegressionResults:
    # The matrix of prediction observations used for MLS estimation
    X: np.ndarray = X_df.to_numpy(copy=False)
    # The matrix of target observations
    y: np.ndarray = y_df.to_numpy(copy=False)

    coefficients_array: np.ndarray = np.linalg.lstsq(X, y, rcond=None)[0]
    # coefficients_array: np.ndarray = np.linalg.solve(X, X.T @ y)
    coefficients_table: pd.DataFrame = pd.DataFrame(
        data=coefficients_array,
        index=X_df.columns,
        columns=y_df.columns,
        copy=False,
    )

    error_array: np.ndarray = y - X @ coefficients_array
    error_table: pd.DataFrame = pd.DataFrame(
        data=error_array, columns=y_df.columns, copy=False
    )

    error_cov_coefficient_denominator: int = len(X_df.index) - len(
        coefficients_table.index
    )
    if df_correction_error_cov and error_cov_coefficient_denominator > 0:
        error_cov_coefficient: float = 1 / (error_cov_coefficient_denominator)
    else:
        error_cov_coefficient: float = 1 / (len(X_df.index))

    error_cov_array: np.ndarray = error_cov_coefficient * (
        np.transpose(error_array) @ error_array
    )
    error_cov_table = pd.DataFrame(
        data=error_cov_array.reshape((len(y_df.columns), len(y_df.columns))),
        index=y_df.columns,
        columns=y_df.columns,
        copy=False,
    )

    xtx_inv: np.ndarray = np.linalg.inv(X.T @ X)

    params_indices: list[str] = []
    for feat in y_df.columns.array:
        params_indices += [(str(feat) + "_" + str(s)) for s in X_df.columns]

    coefficients_cov_array: np.ndarray = np.kron(error_cov_array, xtx_inv)
    coefficients_cov_table: pd.DataFrame = pd.DataFrame(
        data=coefficients_cov_array,
        index=params_indices,
        columns=params_indices,
        copy=False,
    )

    return GeneralLinearRegressionResults(
        coefficients=coefficients_table,
        coefficients_cov=coefficients_cov_table,
        error=error_table,
        error_cov=error_cov_table,
    )
