from typing import NamedTuple, Callable

# import sys
# sys.path.append("C:\\Users\\evanv\\Desktop\\Granger Causality\\Large-scale-nonlinear-causality")

# from utils import lsNGC

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import levene, bartlett

from time_series import TimeSeries, dfTimeSeries
from time_series_regression import (
    linear_time_series_regression,
    LinearTimeSeriesRegressionResults,
    grbf_time_series_regression,
    GRBFTimeSeriesRegressionResults,
)
from f_test import FTestResults, nestedRegressionFTest, equalityVarianceFTest

""" Represents a vector autogressive (VAR) model of a given time series
- [lags] is the number of lags upon which the VAR model was estimated,
    i.e. a lag of 3 means the past 3 values of each variable are used for
    prediction
- [lr_results] is the [LinearTimeSeriesRegressionResults] that have resulted
    from fitting this VAR model to the given time series
"""


class VAR(NamedTuple):
    lags: int
    lr_results: LinearTimeSeriesRegressionResults


""" Represents a vector autogressive (VAR) model of a given time series that
    has been trained using a given Generalized Radial Basis Function (GRBF)
- [lags] is the number of lags upon which the GRBF VAR model was estimated,
    i.e. a lag of 3 means the past 3 values of each variable are used for
    prediction
- [grbf_regression_results] is the [GRBFTimeSeriesRegressionResults] that have 
    resulted from fitting this GRBF VAR model to the given time series
- [rbf] is the GRBF of this model, which is required to take in a float
    representing a distance and output the result of applying the function
    to that distance
"""


class GRBFVAR(NamedTuple):
    lags: int
    grbf_regression_results: GRBFTimeSeriesRegressionResults
    rbf: Callable[[float], float]


"""[fit(time_series, lags)] is a [VAR] model fitted to the given time series
    with the specific number of lags
"""


def fitVAR(time_series: TimeSeries, lags: int) -> VAR:
    return VAR(
        lags=lags,
        lr_results=linear_time_series_regression(
            time_series=time_series, lags=lags
        ),
    )


""" [varGCWald(var, causes,responses)] is the [FTestResults] of applying a
    Wald-statistic based Granger-causality test on [var] that the variables in
    [causes] Granger-cause the variables in [responses]. Since this statistic
    follows an F-distribution, the results are given for the corresponding
    F-statistic.
"""


def varGCTestWald(
    var: VAR, causes: list[str], responses: list[str]
) -> FTestResults:
    assert len(responses) >= 1
    assert len(causes) >= 1
    assert [cause in var.lr_results.coefficients.index for cause in causes]
    assert [
        response in var.lr_results.coefficients.columns
        for response in responses
    ]

    resp_ind: list[int] = [
        var.lr_results.error.columns.get_loc(r) for r in responses
    ]
    cause_ind: list[int] = [
        var.lr_results.error.columns.get_loc(c) for c in causes
    ]

    num_restr: int = len(causes) * len(responses) * var.lags
    num_vars: int = len(var.lr_results.error.columns)

    num_params: int = var.lr_results.coefficients.size
    num_obs: int = len(var.lr_results.error.index)

    C: np.ndarray = np.zeros(shape=(num_restr, num_params))
    b_vec: np.ndarray = var.lr_results.coefficients.to_numpy().ravel("F")

    row: int = 0
    for r_ind in resp_ind:
        for c_ind in cause_ind:
            for lag in range(var.lags):
                resp_offset: int = (1 + num_vars * var.lags) * r_ind
                cause_offset: int = resp_offset + 1 + c_ind * var.lags + lag

                C[row, cause_offset] = 1
                row += 1

    Cb: np.ndarray = np.dot(C, b_vec)
    middle: np.ndarray = np.linalg.inv(
        C @ var.lr_results.coefficients_cov.to_numpy() @ C.T
    )
    wald_statistic: float = np.dot(np.dot(Cb.T, middle), Cb)

    f_statistic: float = wald_statistic / num_restr
    dfn: int = num_restr
    dfd: int = num_obs - (num_vars * var.lags + 1)

    return FTestResults(dfn=dfn, dfd=dfd, f_statistic=f_statistic)


""" [fitGRBFVar(time_series,lags,rbf)] is a [GRBFVAR] model fitted to the given
    [time_series] with [lags] number of lags and the GRBF given by [rbf]
"""


def fitGRBFVar(
    time_series: TimeSeries, lags: int, rbf: Callable[[float], float]
) -> GRBFVAR:
    return GRBFVAR(
        lags=lags,
        grbf_regression_results=grbf_time_series_regression(
            time_series=time_series, lags=lags, radial_basis_function=rbf
        ),
        rbf=rbf,
    )


""" [fitGRBFVar(time_series,lags,rbf)] is a [GRBFVAR] model fitted to the given
    [time_series] with [lags] number of lags and the GRBF given by [rbf],
    except it uses an experimental version of [grbf_time_series_regression],
    called [grbf_time_series_regression_experimental]
"""


def fitGRBFVar_experimental(
    time_series: TimeSeries,
    lags: int,
    rbf: Callable[[float], float],
    num_clusters: int,
) -> GRBFVAR:
    return GRBFVAR(
        lags=lags,
        grbf_regression_results=grbf_time_series_regression(
            time_series=time_series, lags=lags, radial_basis_function=rbf
        ),
        rbf=rbf,
    )


""" [varFTest(unrestricted_var,restricted_var,response)] is the [FTestResults]
    that results from applying a F-test for nested model regression to the
    VAR models given by [unrestricted_var] and [restricted_var]
"""


def varFTest(
    unrestricted_var: VAR, restricted_var: VAR, response: str
) -> FTestResults:
    assert response in unrestricted_var.lr_results.error.columns
    assert response in restricted_var.lr_results.error.columns

    unrestricted_response_error: pd.DataFrame = pd.DataFrame(
        data=unrestricted_var.lr_results.error[response], columns=[response]
    )
    restricted_response_error: pd.DataFrame = pd.DataFrame(
        data=restricted_var.lr_results.error[response], columns=[response]
    )

    rss_unrestricted: float = (
        np.linalg.norm(unrestricted_response_error).astype(dtype=float) ** 2
    )
    rss_restricted: float = (
        np.linalg.norm(restricted_response_error).astype(dtype=float) ** 2
    )

    num_obs: int = len(unrestricted_var.lr_results.error.index)
    num_unrestricted_params: int = len(
        unrestricted_var.lr_results.coefficients.index
    )
    num_restricted_params: int = len(
        restricted_var.lr_results.coefficients.index
    )

    return nestedRegressionFTest(
        rss_unrestricted=rss_unrestricted,
        num_params_unrestricted=num_unrestricted_params,
        rss_restricted=rss_restricted,
        num_params_restricted=num_restricted_params,
        num_obs=num_obs,
    )


""" [grbfVARFTest(unrestricted_grbf_var,restricted_grbf_var,response)] is the
    [FTestResults] that results from applying a F-test for equality of variances
    to the error terms of the GRBF VAR models given by [unrestricted_grbf_var]
    and [restricted_grbf_var]. This test is extremely sensitive to non-normality,
    so unless you are sure the error terms are approximately normally
    distributed it is reccomended to use [grbfVARLevenTest] instead.
"""


def grbfVARFTest(
    unrestricted_grbf_var: GRBFVAR,
    restricted_grbf_var: GRBFVAR,
    response: str,
) -> FTestResults:
    assert (
        response in unrestricted_grbf_var.grbf_regression_results.error.columns
    )
    assert response in restricted_grbf_var.grbf_regression_results.error.columns

    unrestricted_error_covariance = np.linalg.norm(
        pd.DataFrame(
            unrestricted_grbf_var.grbf_regression_results.error_cov[response][
                response
            ]
        )
    ).astype(dtype=float)
    restricted_error_covariance = np.linalg.norm(
        pd.DataFrame(
            restricted_grbf_var.grbf_regression_results.error_cov[response][
                response
            ]
        )
    ).astype(dtype=float)

    num_obs: int = len(
        unrestricted_grbf_var.grbf_regression_results.error.index
    )

    return equalityVarianceFTest(
        variance_unrestricted=unrestricted_error_covariance,
        unrestricted_sample_size=num_obs,
        variance_restricted=restricted_error_covariance,
        restricted_sample_size=num_obs,
    )


""" [grbfVARFTest(unrestricted_grbf_var,restricted_grbf_var,response)] is the
    [FTestResults] that results from applying a Levene test for equality of
    variances to the error terms of the GRBF VAR models given by 
    [unrestricted_grbf_var] and [restricted_grbf_var].
"""


def grbfVARLeveneTest(
    unrestricted_grbf_var: GRBFVAR,
    restricted_grbf_var: GRBFVAR,
    response: str,
) -> FTestResults:
    unrestricted_error = unrestricted_grbf_var.grbf_regression_results.error[
        response
    ]
    restricted_error = restricted_grbf_var.grbf_regression_results.error[
        response
    ]

    assert unrestricted_error.shape == restricted_error.shape

    levene_test_results = levene(unrestricted_error, restricted_error)
    return FTestResults(
        dfn=2 - 1,
        dfd=unrestricted_error.shape[0] - 2,
        f_statistic=levene_test_results[0],
    )


def grbfVARGCTestF(
    time_series: TimeSeries, lags: int, rbf: Callable[[float], float]
) -> pd.DataFrame:
    grbf_var_u = fitGRBFVar(time_series=time_series, lags=lags, rbf=rbf)
    results_df = pd.DataFrame(
        data=np.zeros(
            shape=(len(time_series.df.columns), len(time_series.df.columns))
        ),
        index=time_series_df.columns,
        columns=time_series_df.columns,
    )

    for cause in time_series.df.columns.array:
        time_series_r = dfTimeSeries(
            time_series.df.drop(cause, axis=1, inplace=False)
        )
        grbf_var_r = fitGRBFVar(time_series=time_series_r, lags=lags, rbf=rbf)

        for response in filter(
            lambda resp: resp != cause, time_series.df.columns.array
        ):
            # grbf_var_u.grbf_regression_results.error[response].plot.hist()
            # plt.show()

            # grbf_var_r.grbf_regression_results.error[response].plot.hist()
            # plt.show()

            results_df[response][cause] = grbfVARLeveneTest(
                grbf_var_u, grbf_var_r, response
            ).f_statistic

    return results_df


# response = input("Response: ").split()
# cause = input("Cause: ").split()


# lutkepohl = pd.read_fwf("./test_data/lutkepohl/e1.dat").head(76)
# lutkepohl_diff = lutkepohl.apply(np.log).diff().dropna()
# lutkepohl_diff_c_i = lutkepohl_diff.drop(cause, axis=1)


def gaussian(r: float) -> float:
    return np.exp(-(r**2))


def spline(r: float) -> float:
    return r**2


size = int(input("Time Series Size? "))
num_tests = int(input("Num tests? "))
cutoff = float(input("Cutoff? "))

data = ["S", "T", "U", "V", "W", "X", "Y", "Z"]

for i in range(num_tests):
    s = np.random.random((size, 1)) * 8
    t = np.random.gamma(2, 5, (size, 1)) * 20
    u = np.random.chisquare(10, (size, 1)) * 10
    v = np.random.beta(4, 4, (size, 1)) * 8
    w = np.random.poisson(15, (size, 1))
    x = np.random.normal(2, 10, (size, 1))
    y = np.random.normal(6, 5, (size, 1))
    z = np.random.normal(10, 12, (size, 1))

    for i in range(3, size):
        z[i] = np.cos(i) * np.sin(0.15 * i) + np.random.normal(4, 4)
        w[i] = (
            2 * z[i - 9] ** 3
            - 5 * z[i - 9] ** 2
            + 0.3 * z[i - 9]
            + 2
            + np.random.normal(20, 3)
        )
        v[i] = w[i - 1] * y[i - 7]

    time_series_data = np.concatenate([s, t, u, v, w, x, y, z], axis=1)
    time_series_df = pd.DataFrame(data=time_series_data, columns=data)
    time_series = dfTimeSeries(time_series_df)

    print(grbfVARGCTestF(time_series, 10, gaussian))
    print(grbfVARGCTestF(time_series, 10, gaussian) > cutoff)

    # results_df = pd.DataFrame(data=lsNGC(time_series_data.T,ar_order=10)[1],index=data,columns=data)
    # print(results_df)

# lutkepohl.plot.line()
# plt.show()

# lutkepohl_diff.plot.line()
# plt.show()

# var = fitVAR(dfTimeSeries(lutkepohl_diff), 2)
# var_c_i = fitVAR(dfTimeSeries(lutkepohl_diff_c_i), 2)


# grbf_var = fitGRBFVar_experimental(dfTimeSeries(lutkepohl_diff), 2, gaussian, 3)
# grbf_var_c_i = fitGRBFVar_experimental(
#     dfTimeSeries(lutkepohl_diff_c_i), 2, gaussian, 3)

# grbf_var_r = fitGRBFVar_experimental(dfTimeSeries(lutkepohl), 2, gaussian, 3)
# grbf_var_u = fitGRBFVar_experimental(dfTimeSeries(lutkepohl.drop(cause, axis=1)), 2, gaussian, 3)

# gc = varGCTestWald(var, cause, response)
# print(gc)

# gc = varFTest(var, var_c_i, response)
# print(gc)

# grbf_var.grbf_regression_results.error[response].plot.hist()
# plt.show()

# grbf_var_c_i.grbf_regression_results.error[response].plot.hist()
# plt.show()

# gc = grbfVARFTest(grbf_var, grbf_var_c_i, response)
# print(gc)
