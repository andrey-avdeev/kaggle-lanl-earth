import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import hilbert
from scipy.signal import convolve
from itertools import product

from scipy.signal.windows import hann
from sklearn.linear_model import LinearRegression


def classic_sta_lta(x, length_sta, length_lta):
    sta = np.cumsum(x ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
    sta /= length_sta
    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
    lta /= length_lta

    # Pad zeros
    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]


def calc_change_rate(x):
    change = (np.diff(x) / x[:-1]).values
    change = change[np.nonzero(change)[0]]
    change = change[~np.isnan(change)]
    change = change[change != -np.inf]
    change = change[change != np.inf]

    return np.mean(change)


class Artgor:

    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        feature_dict = pd.DataFrame(dtype=np.float64)
        seg_id = 1

        # lists with parameters to iterate over them
        percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        hann_windows = [50, 150, 1500, 15000]
        spans = [300, 3000, 30000, 50000]
        windows = [10, 50, 100, 500, 1000, 10000]

        # basic stats
        feature_dict.loc[seg_id, 'mean'] = x.mean()
        feature_dict.loc[seg_id, 'std'] = x.std()
        feature_dict.loc[seg_id, 'max'] = x.max()
        feature_dict.loc[seg_id, 'min'] = x.min()

        # basic stats on absolute values
        feature_dict.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))
        feature_dict.loc[seg_id, 'abs_max'] = np.abs(x).max()
        feature_dict.loc[seg_id, 'abs_mean'] = np.abs(x).mean()
        feature_dict.loc[seg_id, 'abs_std'] = np.abs(x).std()

        # geometric and harminic means
        feature_dict.loc[seg_id, 'hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
        feature_dict.loc[seg_id, 'gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))

        # k-statistic and moments
        for i in range(1, 5):
            feature_dict.loc[seg_id, f'kstat_{i}'] = stats.kstat(x, i)
            feature_dict.loc[seg_id, f'moment_{i}'] = stats.moment(x, i)

        for i in [1, 2]:
            feature_dict.loc[seg_id, f'kstatvar_{i}'] = stats.kstatvar(x, i)

        # aggregations on various slices of data
        for agg_type, slice_length, direction in product(['std', 'min', 'max', 'mean'], [1000, 10000, 50000],
                                                         ['first', 'last']):
            if direction == 'first':
                feature_dict.loc[seg_id, f'{agg_type}_{direction}_{slice_length}'] = x[:slice_length].agg(agg_type)
            elif direction == 'last':
                feature_dict.loc[seg_id, f'{agg_type}_{direction}_{slice_length}'] = x[-slice_length:].agg(agg_type)

        feature_dict.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())
        feature_dict.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_dict.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])
        feature_dict.loc[seg_id, 'sum'] = x.sum()

        feature_dict.loc[seg_id, 'mean_change_rate'] = calc_change_rate(x)
        # calc_change_rate on slices of data
        for slice_length, direction in product([1000, 10000, 50000], ['first', 'last']):
            if direction == 'first':
                feature_dict.loc[seg_id, f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(
                    x[:slice_length])
            elif direction == 'last':
                feature_dict.loc[seg_id, f'mean_change_rate_{direction}_{slice_length}'] = calc_change_rate(
                    x[-slice_length:])

        # percentiles on original and absolute values
        for p in percentiles:
            feature_dict.loc[seg_id, f'percentile_{p}'] = np.percentile(x, p)
            feature_dict.loc[seg_id, f'abs_percentile_{p}'] = np.percentile(np.abs(x), p)

        feature_dict.loc[seg_id, 'trend'] = add_trend_feature(x)
        feature_dict.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)

        feature_dict.loc[seg_id, 'mad'] = x.mad()
        feature_dict.loc[seg_id, 'kurt'] = x.kurtosis()
        feature_dict.loc[seg_id, 'skew'] = x.skew()
        feature_dict.loc[seg_id, 'med'] = x.median()

        feature_dict.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()

        for hw in hann_windows:
            feature_dict.loc[seg_id, f'Hann_window_mean_{hw}'] = (
                    convolve(x, hann(hw), mode='same') / sum(hann(hw))).mean()

        feature_dict.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()
        feature_dict.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

        # exponential rolling statistics
        ewma = pd.Series.ewm
        for s in spans:
            feature_dict.loc[seg_id, f'exp_Moving_average_{s}_mean'] = (ewma(x, span=s).mean(skipna=True)).mean(
                skipna=True)
            feature_dict.loc[seg_id, f'exp_Moving_average_{s}_std'] = (ewma(x, span=s).mean(skipna=True)).std(
                skipna=True)
            feature_dict.loc[seg_id, f'exp_Moving_std_{s}_mean'] = (ewma(x, span=s).std(skipna=True)).mean(skipna=True)
            feature_dict.loc[seg_id, f'exp_Moving_std_{s}_std'] = (ewma(x, span=s).std(skipna=True)).std(skipna=True)

        feature_dict.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict.loc[seg_id, 'iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
        feature_dict.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)

        for slice_length, threshold in product([50000, 100000, 150000],
                                               [5, 10, 20, 50, 100]):
            feature_dict.loc[seg_id, f'count_big_{slice_length}_threshold_{threshold}'] = (
                    np.abs(x[-slice_length:]) > threshold).sum()
            feature_dict.loc[seg_id, f'count_big_{slice_length}_less_threshold_{threshold}'] = (
                    np.abs(x[-slice_length:]) < threshold).sum()

            # statistics on rolling windows of various sizes
        for w in windows:
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values

            feature_dict.loc[seg_id, f'ave_roll_std_{w}'] = x_roll_std.mean()
            feature_dict.loc[seg_id, f'std_roll_std_{w}'] = x_roll_std.std()
            feature_dict.loc[seg_id, f'max_roll_std_{w}'] = x_roll_std.max()
            feature_dict.loc[seg_id, f'min_roll_std_{w}'] = x_roll_std.min()

            for p in percentiles:
                feature_dict.loc[seg_id, f'percentile_roll_std_{p}_window_{w}'] = np.percentile(x_roll_std, p)

            feature_dict.loc[seg_id, f'av_change_abs_roll_std_{w}'] = np.mean(np.diff(x_roll_std))
            feature_dict.loc[seg_id, f'av_change_rate_roll_std_{w}'] = np.mean(
                np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            feature_dict.loc[seg_id, f'abs_max_roll_std_{w}'] = np.abs(x_roll_std).max()

            feature_dict.loc[seg_id, f'ave_roll_mean_{w}'] = x_roll_mean.mean()
            feature_dict.loc[seg_id, f'std_roll_mean_{w}'] = x_roll_mean.std()
            feature_dict.loc[seg_id, f'max_roll_mean_{w}'] = x_roll_mean.max()
            feature_dict.loc[seg_id, f'min_roll_mean_{w}'] = x_roll_mean.min()

            for p in percentiles:
                feature_dict.loc[seg_id, f'percentile_roll_mean_{p}_window_{w}'] = np.percentile(x_roll_mean, p)

            feature_dict.loc[seg_id, f'av_change_abs_roll_mean_{w}'] = np.mean(np.diff(x_roll_mean))
            feature_dict.loc[seg_id, f'av_change_rate_roll_mean_{w}'] = np.mean(
                np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            feature_dict.loc[seg_id, f'abs_max_roll_mean_{w}'] = np.abs(x_roll_mean).max()

        return feature_dict
