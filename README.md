# BNL-Causality
## Project Overvieq
- [f_test.py] is a file containing utilities for performing F-tests
    to test for equality of variance as well as nested regression
    goodness of fit
- [general_linear_regression.py] is a file containing utilities for
    performing general least squares linear regression on data
    with potentially many explanatory and response variables
- [time_series.py] contains utilities for creating and packaging
    time series data into a format to be used with the rest of this
    project
- [time_series_regression.py] is the meat (so far) of this project,
    and includes the logic for performing regression on a time series
    for a specified number of lags
- [var.py] contains Vector Autoregression (VAR) models for representing
    time series regression, as well as utilites for assesing
    Granger-causality for different time series variables