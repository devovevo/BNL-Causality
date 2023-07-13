from typing import NamedTuple

import numpy as np
import pandas as pd


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
    df: pd.DataFrame

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
def dfTimeSeries(df: pd.DataFrame):
    return TimeSeries(df=df)


""" [laggedTimeSeries(time_series, lags)] is a [LaggedTimeSeries] containing a table
    of the lags of the given [time_series] as columns. If the time series is 
    organized as Var1 Var2 Var3 ... VarJ, then this table will be organized as 
    Var1_Lag_1 Var1_Lag_2 ... Var1_Lag_n Var2_Lag_1 Var2_Lag_2 ... Var2_Lag_n
    Var3_Lag_1 Var3_Lag_2 ... Var3_Lag_n ... VarJ_Lag_n for a times series with 
    J variables and n lags.
"""
def laggedTimeSeries(time_series: TimeSeries, lags: int) -> TimeSeries:
    assert time_series.df.size >= 1
    assert lags >= 1

    numobs: int = len(time_series.df.index)
    features: pd.Index = time_series.df.columns

    lag_array_nan_t: np.ndarray = np.ones(shape=(len(features) * lags, numobs))
    lag_columns: list[str] = []

    for feature_index in range(len(features)):
        feature: str = features.array[feature_index]

        for lag in range(1, lags + 1):
            lag_array_nan_t[feature_index * lags + (lag - 1)] = (
                time_series.df[feature].shift(lag).to_numpy(copy=False)
            )
            lag_columns.append(f"{feature}_Lag_{lag}")

    lag_array: np.ndarray = lag_array_nan_t.T[lags:]
    lag_table: pd.DataFrame = pd.DataFrame(
        data=lag_array, columns=lag_columns, copy=False
    )

    return TimeSeries(df=lag_table)
