from typing import NamedTuple, Callable

from pandas import DataFrame
from numpy import ndarray, zeros
from numpy.linalg import pinv
from scipy.stats import levene

from time_series import TimeSeries
from time_series_regression import (
    var_time_series_regression,
    grbf_time_series_regression,
    VARRegressionResults,
    GRBFRegressionResults,
)
from f_test import FTestResults

""" Represents a vector autogressive (VAR) model of a given time series
- [lags] is the number of lags upon which the VAR model was estimated,
    i.e. a lag of 3 means the past 3 values of each variable are used for
    prediction
- [glr_results] is the [LinearTimeSeriesRegressionResults] that have resulted
    from fitting this VAR model to the given time series
"""


class VAR(NamedTuple):
    lags: list[int]
    regression: VARRegressionResults


""" Represents a vector autogressive (VAR) model of a given time series that
    has been trained using a given Generalized Radial Basis Function (GRBF)
- [lags] is the number of lags upon which the GRBF VAR model was estimated,
    i.e. a lag of 3 means the past 3 values of each variable are used for
    prediction
- [regression] is the [GRBFRegressionResults] that have 
    resulted from fitting this GRBF VAR model to the given time series
- [rbf] is the RBF of this model, which is required to take in a float
    representing a distance and output the result of applying the function
    to that distance
"""


class GRBF(NamedTuple):
    lags: list[int]
    regression: GRBFRegressionResults
    rbf: Callable[[float], float]


""" Represents the results of performing a Granger-causality (GC) test on
    a VAR model
- [f_statistics] is a [DataFrame] with the [i,j]-th entry being the F-statistic
    for GC of the [j]-th variable on the [i]-th variable
- [dfn] is the first degrees of freedom for the given [f_statistics]
- [rbf] is the second degrees of freedom for the given [f_statistics]
"""


class GCTestResults(NamedTuple):
    f_statistics: DataFrame
    dfn: int
    dfd: int


"""[fit_var(time_series, lags)] is a [VAR] model fitted to the given time series
    with the specific number of lags
"""


def fit_var(time_series: TimeSeries, lags: list[int]) -> VAR:
    return VAR(
        lags=lags,
        regression=var_time_series_regression(
            time_series=time_series, lags=lags
        ),
    )


""" [var_gc_test_wald(var, causes,responses)] is the [FTestResults] of applying a
    Wald-statistic based Granger-causality test on [var] that the variables in
    [causes] Granger-cause the variable given by [response]. Since this statistic
    follows an F-distribution, the results are given for the corresponding
    F-statistic.
"""


def var_gc_test_wald(
    var: VAR, causes: list[str], response: str
) -> FTestResults:
    assert len(causes) >= 1
    assert [cause in var.regression.gls.coefficients.index for cause in causes]
    assert response in var.regression.gls.coefficients.columns
    assert var.regression.gls.coefficients_cov is not None

    resp_ind: int = var.regression.gls.error.columns.get_loc(response)
    cause_ind: list[int] = [
        var.regression.gls.error.columns.get_loc(c) for c in causes
    ]

    num_restr: int = len(causes) * len(var.lags)
    num_vars: int = len(var.regression.gls.error.columns)

    num_params: int = var.regression.gls.coefficients.size
    num_obs: int = len(var.regression.gls.error.index)

    C: ndarray = zeros(shape=(num_restr, num_params))
    b_vec: ndarray = var.regression.gls.coefficients.to_numpy().ravel("F")

    row: int = 0
    for c_ind in cause_ind:
        for lag_index in range(len(var.lags)):
            resp_offset: int = (1 + num_vars * len(var.lags)) * resp_ind
            cause_offset: int = (
                resp_offset + 1 + c_ind * len(var.lags) + lag_index
            )

            C[row, cause_offset] = 1
            row += 1

    Cb: ndarray = C @ b_vec
    middle: ndarray = pinv(
        C @ var.regression.gls.coefficients_cov.to_numpy(copy=False) @ C.T
    )
    wald_statistic: float = float(Cb.T @ middle @ Cb)

    f_statistic: float = wald_statistic / num_restr
    dfn: int = num_restr
    dfd: int = num_obs - (num_vars * len(var.lags) + 1)

    return FTestResults(dfn=dfn, dfd=dfd, f_statistic=f_statistic)


""" [var_gc_test(time_series, lags)] is the [GCTestResults] that results from
    testing for GC on the variables of the given [time_series] using a VAR model
    at the lags specified by [lags]
"""


def var_gc_test(time_series: TimeSeries, lags: list[int]) -> GCTestResults:
    var = fit_var(time_series, lags)
    gc_test_df = DataFrame(
        data=zeros(
            shape=(len(time_series.df.columns), len(time_series.df.columns))
        ),
        columns=time_series.df.columns,
        index=time_series.df.columns,
    )

    dfn: int = 0
    dfd: int = 0

    for cause in time_series.df.columns.array:
        for response in time_series.df.columns.array:
            gc_result: FTestResults = var_gc_test_wald(var, [cause], response)
            gc_test_df[response][cause] = gc_result.f_statistic

            dfn = gc_result.dfn
            dfd = gc_result.dfd

    return GCTestResults(f_statistics=gc_test_df, dfn=dfn, dfd=dfd)


""" [fit_grbf(time_series,lags,rbf)] is a [GRBF] model fitted to the given
    [time_series] using the information given by [lags] and the RBF given by [rbf]
"""


def fit_grbf(
    time_series: TimeSeries,
    lags: list[int],
    rbf: Callable[[float], float],
    exclude_explanatory_variables: list[str] = [],
) -> GRBF:
    return GRBF(
        lags=lags,
        regression=grbf_time_series_regression(
            time_series=time_series,
            lags=lags,
            radial_basis_function=rbf,
            exclude_explanatory_variables=exclude_explanatory_variables,
        ),
        rbf=rbf,
    )


""" [grbf_gc_test(time_series,lags,rbf)] is a [GCTestResults] that results from
    testing for Granger-causality using a [GRBF] model fitted to the given
    [time_series] with lags given by [lags] and radial basis function [rbf]
"""


def grbf_gc_test(
    time_series: TimeSeries, lags: list[int], rbf: Callable[[float], float]
) -> GCTestResults:
    grbf_var_u = fit_grbf(time_series=time_series, lags=lags, rbf=rbf)

    results_df = DataFrame(
        data=zeros(
            shape=(len(time_series.df.columns), len(time_series.df.columns))
        ),
        index=time_series.df.columns,
        columns=time_series.df.columns,
    )

    dfn: int = 0
    dfd: int = 0

    for cause in time_series.df.columns:
        grbf_var_r = fit_grbf(
            time_series=time_series,
            lags=lags,
            rbf=rbf,
            exclude_explanatory_variables=[cause],
        )

        for response in time_series.df.columns:
            results_gc = levene(
                grbf_var_u.regression.gls.error[response],
                grbf_var_r.regression.gls.error[response],
            )

            results_df[response][cause] = results_gc[0]
            dfn = 2 - 1
            dfd = len(grbf_var_u.regression.gls.error) - 2

    return GCTestResults(f_statistics=results_df, dfn=dfn, dfd=dfd)
