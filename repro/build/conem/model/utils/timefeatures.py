# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:13:12 2022

@author: nhngu
"""

from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"

class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5

# Temporal Embedding for Daily -------------------------------------------------
class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class DayOfSafetyNet(TimeFeature):
    """Safety Net day as last date of year encoded as value between [-5, 0]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return pd.Float64Index((index.day==31)&(index.month==12))*(-5)
    
    
# Temporal Embedding for Monthly -------------------------------------------------
class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


# Temporal Embedding for Weekly -------------------------------------------------
class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return pd.Float64Index((((index.isocalendar().week- 1) / 52.0 - 0.5)*1).reset_index(name='dates').set_index('dates').index)
    
    
class WeekContainsSafetyNet(TimeFeature):
    """Safety Net week encoded as value between [-5,0]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return pd.Float64Index(((index.day+6)>=31)&(index.month==12))*(-5) # --------- HARD CODED
        return pd.Float64Index((index.month == 12)*1*index.day)/31-0.5

# Temporal functions for Freq forecasting purpse
def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    Returns a list of time features that will be appropriate for the given frequency string.
    Parameters
    ----------
    freq_str
        Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.
    """

    features_by_offsets = {
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [MonthOfYear,WeekOfYear,WeekContainsSafetyNet], #WeekOfYear,MonthOfYear,WeekContainsSafetyNet
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear,MonthOfYear,DayOfSafetyNet],
        offsets.Hour: [HourOfDay,DayOfWeek, DayOfMonth, DayOfYear,MonthOfYear],
        offsets.Minute: [MinuteOfHour,HourOfDay,DayOfWeek, DayOfMonth, DayOfYear,MonthOfYear]
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
    """
    raise RuntimeError(supported_freq_msg)

# Time features function
def time_features(dates, freq='w'):
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])




## Encoding Temporal features as One-hot-encoding
# Temporal Embedding for Daily -------------------------------------------------
class DayOfWeek_onehot(TimeFeature):
    """Hour of day encoded as value between [1,7]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class DayOfSafetyNet(TimeFeature):
    """Safety Net day as last date of year encoded as value between [-5, 0]"""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return pd.Float64Index((index.day==31)&(index.month==12))*(-5)











