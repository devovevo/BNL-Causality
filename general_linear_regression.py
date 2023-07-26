from typing import NamedTuple

from pandas import DataFrame
from numpy.linalg import lstsq, pinv
from numpy import transpose, kron, ndarray

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
    coefficients: DataFrame
    coefficients_cov: DataFrame | None
    error: DataFrame
    error_cov: DataFrame | None


""" [general_linear_regression(X_df, y_df, df_correction_error_cov, compute_cov)] represents
    the results of performing general linear regression on a design matrix
    [X_df], where each column represents an explanatory variable and each row
    represents a measurement of each of the explanatory variables, and a target
    matrix [y_df]. These results are the least squares solution to the equation
    [y_df = X_df * b], where [coefficients] represents [b], [coefficients_cov]
    represents the variance-covariance matrix of [b], [error] represents the
    error in prediction of [b], and [error_cov] is the variance-covariance 
    matrix of [error]. The optional [df_correction_error_cov] gives an adjustment
    to the error covariance matrix calculation, where it divides the matrix by
    [1 / (N - k)], where [N] is the number of rows in [X_df] and [k] is the number
    of columns of [X_df]. The [compute_cov] option describes whether to compute
    the error and coefficient covariance matrices.
"""


def general_linear_regression(
    X_df: DataFrame,
    y_df: DataFrame,
    df_correction_error_cov=False,
    compute_cov=True,
) -> GeneralLinearRegressionResults:
    # The matrix of prediction observations used for LS estimation
    X: ndarray = X_df.to_numpy(copy=False)
    # The matrix of target observations
    y: ndarray = y_df.to_numpy(copy=False)

    coefficients_array: ndarray = lstsq(X, y, rcond=None)[0]
    coefficients_table: DataFrame = DataFrame(
        data=coefficients_array,
        index=X_df.columns,
        columns=y_df.columns,
        copy=False,
    )

    error_array: ndarray = y - X @ coefficients_array
    error_table: DataFrame = DataFrame(
        data=error_array, columns=y_df.columns, copy=False
    )

    coefficients_cov_table: DataFrame | None = None
    error_cov_table: DataFrame | None = None

    if compute_cov:
        error_cov_coefficient_denominator: int = (
            X.shape[0] - coefficients_array.shape[0]
        )

        if df_correction_error_cov and error_cov_coefficient_denominator > 0:
            error_cov_coefficient: float = 1 / (
                error_cov_coefficient_denominator
            )
        else:
            error_cov_coefficient: float = 1 / X.shape[0]

        error_cov_array = error_cov_coefficient * (
            transpose(error_array) @ error_array
        )
        error_cov_table = DataFrame(
            data=error_cov_array,
            copy=False,
            index=y_df.columns,
            columns=y_df.columns,
        )

        xtx_inv: ndarray = pinv(X.T @ X)

        params_indices = []
        for feat in y_df.columns.values:
            params_indices.extend(
                [(str(feat) + "_" + str(s)) for s in X_df.columns.values]
            )

        coefficients_cov_array = kron(error_cov_array, xtx_inv)
        coefficients_cov_table = DataFrame(
            data=coefficients_cov_array,
            copy=False,
            index=params_indices,
            columns=params_indices,
        )

    return GeneralLinearRegressionResults(
        coefficients=coefficients_table,
        coefficients_cov=coefficients_cov_table,
        error=error_table,
        error_cov=error_cov_table,
    )
