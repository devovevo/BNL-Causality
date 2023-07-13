from typing import NamedTuple, Callable

import numpy as np
import pandas as pd
from general_linear_regression import (
    GeneralLinearRegressionResults,
    general_linear_regression
)
from sklearn.metrics.pairwise import euclidean_distances
from time_series import TimeSeries, laggedTimeSeries, dfTimeSeries

''' Represents the results of performing linear time series regression on
    a given [TimeSeries] object. In essence, creates a design matrix [X] with the
    columns being the desired number of lags for each variable (except for the
    first, which is a column of 1s to represent an intercept term), and each row
    representing the lags before a given measurement. It also creates a target 
    matrix [y] with the next measurement in the time series. It then calculates
    the least squares solution to [y = Xb], with [coefficients] representing [b]
    in this case. The [coefficients_cov] represents the variance-covariance
    matrix for all the coefficients, and [error] and [error_cov] represent
    the errors in prediction for each measurement and the covariance of these
    errors, respectively.
'''
class LinearTimeSeriesRegressionResults(NamedTuple):
    coefficients: pd.DataFrame
    coefficients_cov: pd.DataFrame
    error: pd.DataFrame
    error_cov: pd.DataFrame


''' Represents the results of performing General Radial Basis Function (GRBF)
    time series regression on a given [TimeSeries] object. In essence, creates
    a design matrix [X] with the columns being the desired number of lags for
    each variable, and each row representing the lags before a given measurement.
    It then normalizes the columns of [X], before calculating a distance matrix
    [D] from [X], where the [ij]-th entry of [D] is the distance between the
    [i]-th and [j]-th rows of [X]. Finally, it normalizes [D] by dividing by the
    L2 norm of the matrix, before applying the GRBF to every entry of [D],
    resulting in the final [Phi] matrix, which has a column of 1s inserted into
    it to represent intercept terms. It also creates a target matrix [y] 
    with the next measurement in the time series.  It then calculates
    the least squares solution to [y = Phi * b], with [coefficients] representing
    [b] in this case. The [coefficients_cov] represents the variance-covariance
    matrix for all the coefficients, and [error] and [error_cov] represent
    the errors in prediction for each measurement and the covariance of these
    errors, respectively. The [feature_norms] array represents the norms
    of each of the columns of the original [X] design matrix, and [matrix_norm]
    represents the norm of the distance matrix [D].
'''
class GRBFTimeSeriesRegressionResults(NamedTuple):
    coefficients: pd.DataFrame
    coefficients_cov: pd.DataFrame
    error: pd.DataFrame
    error_cov: pd.DataFrame
    feature_norms: pd.DataFrame
    matrix_norm: float

''' [linear_time_series_regression(time_series, lags)] is the
    [LinearTimeSeriesRegressionResults] that come from taking the time series
    data in [time_series] and generating a design matrix [X] where each row
    contains [lags] lags of each variable in the time series given by
    [time_series], and each column represents one specific lag of one of the
    variables. It then creates a target matrix [y] with the corresponding
    next observation of each row in the design matrix [X]. It then returns
    the [LinearTimeSeriesRegressionResults] that results from solving the system
    [y = Xb] for [b], where [coefficients] represents [b], [coefficients_cov]
    is the covariance matrix of [b], [error] is the error in prediction for each
    measurement, and [error_cov] is the covariance matrix of all the errors
    in prediction
'''
def linear_time_series_regression(
    time_series: TimeSeries, lags: int
) -> LinearTimeSeriesRegressionResults:
    lagged_time_series = laggedTimeSeries(time_series=time_series, lags=lags)

    intercept_array = np.ones(shape=(len(lagged_time_series.df.index), 1))
    intercept_dataframe = pd.DataFrame(
        data=intercept_array, columns=["Intercept"]
    )
    lagged_time_series_w_intercept = pd.concat(
        objs=[intercept_dataframe, lagged_time_series.df], axis=1, copy=False
    )

    time_series_regression = general_linear_regression(
        lagged_time_series_w_intercept, time_series.df[lags:]
    )

    return LinearTimeSeriesRegressionResults(
        coefficients=time_series_regression.coefficients,
        coefficients_cov=time_series_regression.coefficients_cov,
        error=time_series_regression.error,
        error_cov=time_series_regression.error_cov,
    )

''' [grbf_time_series_regression(time_series,lags,radial_basis_function)] is the
    [GRBFTimeSeriesRegressionResults] that results from fitting a GRBF VAR
    model to the given [time_series]. First, it creates
    a design matrix [X] with the columns being the desired number of lags for
    each variable, and each row representing the lags before a given measurement.
    It then normalizes the columns of [X], before calculating a distance matrix
    [D] from [X], where the [ij]-th entry of [D] is the distance between the
    [i]-th and [j]-th rows of [X]. Finally, it normalizes [D] by dividing by the
    L2 norm of the matrix, before applying the GRBF to every entry of [D],
    resulting in the final [Phi] matrix, which has a column of 1s inserted into
    it to represent intercept terms. It also creates a target matrix [y] 
    with the next measurement in the time series.  It then calculates
    the least squares solution to [y = Phi * b], with [coefficients] representing
    [b] in this case. The [coefficients_cov] represents the variance-covariance
    matrix for all the coefficients, and [error] and [error_cov] represent
    the errors in prediction for each measurement and the covariance of these
    errors, respectively. The [feature_norms] array represents the norms
    of each of the columns of the original [X] design matrix, and [matrix_norm]
    represents the norm of the distance matrix [D].
'''
def grbf_time_series_regression(
    time_series: TimeSeries,
    lags: int,
    radial_basis_function: Callable[[float], float]
):
    lagged_time_series: TimeSeries = laggedTimeSeries(
        time_series=time_series, lags=lags
    )

    feature_norms_array: np.ndarray = np.linalg.norm(
        lagged_time_series.df, axis=0
    ).reshape((1, len(lagged_time_series.df.columns)))
    feature_norms_dataframe: pd.DataFrame = pd.DataFrame(
        data=feature_norms_array,
        index=["Norms"],
        columns=lagged_time_series.df.columns,
        copy=False,
    )

    normalized_lagged_time_series: pd.DataFrame = lagged_time_series.df.divide(
        feature_norms_dataframe.values, axis="columns"
    )

    distance_matrix = euclidean_distances(normalized_lagged_time_series, normalized_lagged_time_series)
    distance_matrix_norm = np.linalg.norm(distance_matrix).astype(float)
    normalized_distance_matrix = distance_matrix / distance_matrix_norm
    phi_vectorize: Callable[[np.ndarray], np.ndarray] = np.vectorize(
        radial_basis_function
    )

    phi_matrix: np.ndarray = phi_vectorize(normalized_distance_matrix)
    normalized_phi_matrix: np.ndarray = phi_matrix
    normalized_phi_dataframe: pd.DataFrame = pd.DataFrame(
        data=normalized_phi_matrix,
        copy=False,
    )

    intercept_array: np.ndarray = np.ones(shape=(len(normalized_phi_dataframe.index), 1))
    intercept_dataframe: pd.DataFrame = pd.DataFrame(
        data=intercept_array, columns=["Intercept"], copy=False
    )

    phi_intercept_dataframe: pd.DataFrame = pd.concat(
        objs=[intercept_dataframe, normalized_phi_dataframe], axis=1, copy=False
    )
    grbf_linear_regression: GeneralLinearRegressionResults = (
        general_linear_regression(
            phi_intercept_dataframe, time_series.df[lags:]
        )
    )
    
    return GRBFTimeSeriesRegressionResults(
        coefficients=grbf_linear_regression.coefficients,
        coefficients_cov=grbf_linear_regression.coefficients_cov,
        error=grbf_linear_regression.error,
        error_cov=grbf_linear_regression.error_cov,
        feature_norms=feature_norms_dataframe,
        matrix_norm=distance_matrix_norm
    )