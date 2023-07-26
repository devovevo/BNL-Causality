from typing import NamedTuple, Callable

from pandas import DataFrame, concat
from numpy import ndarray, ones, vectorize
from numpy.linalg import norm
from numba import jit
from general_linear_regression import (
    GeneralLinearRegressionResults,
    general_linear_regression,
)
from time_series import (
    TimeSeries,
    laggedTimeSeries,
    dfTimeSeries,
    laggedTimeSeries,
)
from fastdist import fastdist


class VARDesignMatrix(NamedTuple):
    lagged_time_series_w_intercept: DataFrame


class VARRegressionResults(NamedTuple):
    var_design_matrix: VARDesignMatrix
    target_time_series: TimeSeries
    gls: GeneralLinearRegressionResults


class GRBFDesignMatrix(NamedTuple):
    normalized_lagged_time_series: DataFrame
    distance_dataframe: DataFrame
    distance_dataframe_norm: float
    feature_norms: DataFrame
    phi_matrix_norms: DataFrame
    normalized_phi_dataframe: DataFrame


class GRBFRegressionResults(NamedTuple):
    grbf_design_matrix: GRBFDesignMatrix
    target_time_series: TimeSeries
    gls: GeneralLinearRegressionResults


def var_time_series_regression_design_matrix(
    time_series: TimeSeries, lags: list[int]
) -> VARDesignMatrix:
    lagged_time_series = laggedTimeSeries(time_series=time_series, lags=lags)

    intercept_array = ones(shape=(len(lagged_time_series.df.index), 1))
    intercept_dataframe = DataFrame(data=intercept_array, columns=["Intercept"])
    lagged_time_series_w_intercept = concat(
        objs=[intercept_dataframe, lagged_time_series.df], axis=1, copy=False
    )

    return VARDesignMatrix(lagged_time_series_w_intercept)


""" [linear_time_series_regression(time_series, lags)] is the
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
"""


def var_time_series_regression(
    time_series: TimeSeries, lags: list[int]
) -> VARRegressionResults:
    var_design = var_time_series_regression_design_matrix(time_series, lags)
    target_ts = dfTimeSeries(time_series.df[max(lags) :])
    gls = general_linear_regression(
        var_design.lagged_time_series_w_intercept, target_ts.df
    )

    return VARRegressionResults(var_design, target_ts, gls)


def grbf_time_series_design_matrix(
    time_series: TimeSeries,
    lags: list[int],
    radial_basis_function: Callable[[float], float],
    exclude_explanatory_variables: list[str] = [],
) -> GRBFDesignMatrix:
    excluded_time_series: TimeSeries = dfTimeSeries(
        time_series.df.drop(exclude_explanatory_variables, axis=1)
    )
    lagged_time_series: TimeSeries = laggedTimeSeries(
        time_series=excluded_time_series, lags=lags
    )

    feature_norms_array: ndarray = norm(
        lagged_time_series.df, axis=0, keepdims=True
    )
    feature_norms_dataframe: DataFrame = DataFrame(
        data=feature_norms_array,
        index=["Norms"],
        columns=lagged_time_series.df.columns,
        copy=False,
    )

    normalized_lagged_time_series: DataFrame = lagged_time_series.df.div(
        feature_norms_dataframe.values, axis="columns"
    )

    distance_matrix_df = DataFrame(
        data=fastdist.matrix_pairwise_distance(
            normalized_lagged_time_series.to_numpy(copy=False),
            metric=fastdist.euclidean,
            metric_name="euclidian",
            return_matrix=True,
        )
    )

    distance_matrix_norm = norm(distance_matrix_df).astype(float)
    normalized_distance_matrix = distance_matrix_df / distance_matrix_norm

    phi_matrix: ndarray = vectorize(radial_basis_function)(
        normalized_distance_matrix
    )
    phi_df: DataFrame = DataFrame(data=phi_matrix, copy=False)
    phi_matrix_norms: DataFrame = DataFrame(
        data=phi_matrix.sum(axis=1, keepdims=True)
    )
    normalized_phi_dataframe = phi_df.div(phi_matrix_norms.values)

    return GRBFDesignMatrix(
        normalized_lagged_time_series,
        distance_matrix_df,
        distance_matrix_norm,
        feature_norms_dataframe,
        phi_matrix_norms,
        normalized_phi_dataframe,
    )


""" [grbf_time_series_regression(time_series,lags,radial_basis_function)] is the
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
"""


def grbf_time_series_regression(
    time_series: TimeSeries,
    lags: list[int],
    radial_basis_function: Callable[[float], float],
    exclude_explanatory_variables: list[str] = [],
) -> GRBFRegressionResults:
    grbf_design = grbf_time_series_design_matrix(
        time_series, lags, radial_basis_function, exclude_explanatory_variables
    )
    target_ts = dfTimeSeries(time_series.df[max(lags) :])
    gls = general_linear_regression(
        grbf_design.normalized_phi_dataframe, target_ts.df
    )

    return GRBFRegressionResults(grbf_design, target_ts, gls)
