from typing import NamedTuple

""" [FTestResults] represents the results of performing a F-statistic based
    test with
- [dfn], the first degrees of freedom of the given F-statistic
- [dfd], the second degrees of freedom of the given F-statistic
- [f_statistic], the F-statistic with ([dfn], [dfd]) degrees of freedom
    representing the results of the given test
"""


class FTestResults(NamedTuple):
    dfn: int
    dfd: int
    f_statistic: float


""" [nested_regression_f_test(rss_u,num_params_u,rss_r,num_params_r)] is
    the [FTestResults] that comes from performing a nested regression F-test on
    an unrestricted model with residual sum of squares [rss_u] and number of
    params [num_params_u] and a restricted model with residual sum of squares
    [rss_r] and number of parameters [num_params_r]."""


def nested_regression_f_test(
    rss_unrestricted: float,
    num_params_unrestricted: int,
    rss_restricted: float,
    num_params_restricted: int,
    num_obs: int,
) -> FTestResults:
    assert num_params_unrestricted < num_params_restricted

    dfn: int = num_params_unrestricted - num_params_restricted
    dfd: int = num_obs - num_params_unrestricted

    f_statistic_numerator: float = (rss_restricted - rss_unrestricted) / (dfn)
    f_statistic_denominator: float = rss_unrestricted / dfd

    f_statistic: float = f_statistic_numerator / f_statistic_denominator
    return FTestResults(dfn=dfn, dfd=dfd, f_statistic=f_statistic)


""" [equality_variance_f_test(variance_u,u_sample_size,variance_r,r_sample_size)]
    is the [FTestResults] that results from performing an F-test for equality
    of variance on two samples, one with variance [variance_u] and sample size
    [u_sample_size], and the other with variance [variance_r] and sample size
    [r_sample_size]
"""


def equality_variance_f_test(
    variance_unrestricted: float,
    unrestricted_sample_size: int,
    variance_restricted: float,
    restricted_sample_size: int,
) -> FTestResults:
    return FTestResults(
        dfn=restricted_sample_size - 1,
        dfd=unrestricted_sample_size - 1,
        f_statistic=variance_restricted / variance_unrestricted,
    )
