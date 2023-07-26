from typing import NamedTuple

from numpy import ndarray, ones
from pandas import DataFrame, concat, Index
from numba import jit


""" Represents a Time Series with data given by [df], which is to be of the
    following format:
        Var1 Var2 Var3 ... VarN
    0     #    .   .   .    #
    1         .
    2     .      .          .
    3     .         #       .
    4     .            .    .
    5                    .
    6     #    .   .   .    #

    In other words, the data should have the columns be the variables tracked
    over time, and the rows show be the indexes of the time-stamps
"""


class TimeSeries(NamedTuple):
    df: DataFrame


""" [dfTimeSeries(df)] is the Time Series with data given by the [DataFrame]
    [df], which MUST be in the following format:
        Var1 Var2 Var3 ... VarN
    0     #    .   .   .    #
    1         .
    2     .      .          .
    3     .         #       .
    4     .            .    .
    5                    .
    6     #    .   .   .    #

    In other words, the data should have the columns be the variables tracked
    over time, and the rows show be the indexes of the time-stamps
"""


def dfTimeSeries(df: DataFrame):
    return TimeSeries(df=df)


""" [lagged_time_series(time_series, lags)] is a [LaggedTimeSeries] containing a table
    of the [lags] of the given [time_series] as columns. If the time series is 
    organized as Var1 Var2 Var3 ... VarJ, then this table will be organized as 
    Var1_Lag_1 Var1_Lag_2 ... Var1_Lag_n Var2_Lag_1 Var2_Lag_2 ... Var2_Lag_n
    Var3_Lag_1 Var3_Lag_2 ... Var3_Lag_n ... VarJ_Lag_n for a times series with 
    J variables and n lags.
"""


def lagged_time_series(time_series: TimeSeries, lags: list[int]) -> TimeSeries:
    assert time_series.df.size >= 1
    assert [lag >= 1 for lag in lags]
    assert len(set(lags)) == len(lags)

    numobs: int = len(time_series.df.index)
    features: Index = time_series.df.columns

    lag_array_nan_t: ndarray = ones(shape=(len(features) * len(lags), numobs))
    lag_columns: list[str] = []

    for feature_index in range(len(features)):
        feature: str = features.array[feature_index]

        for lag in lags:
            lag_array_nan_t[feature_index * len(lags) + lags.index(lag)] = (
                time_series.df[feature].shift(lag).to_numpy(copy=False)
            )
            lag_columns.append(f"{feature}_Lag_{lag}")

    lag_array: ndarray = lag_array_nan_t.T[max(lags) :]
    lag_table: DataFrame = DataFrame(
        data=lag_array, columns=lag_columns, copy=False
    )

    return TimeSeries(df=lag_table)
