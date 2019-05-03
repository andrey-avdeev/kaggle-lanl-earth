import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class Base:

    @staticmethod
    def _add_trend_feature(arr, abs_values: bool = False) -> float:
        idx = np.array(range(len(arr)))

        if abs_values:
            arr = np.abs(arr)

        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)

        return lr.coef_[0]

    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame(dtype=np.float64)

        df.loc[1, 'ave'] = x.values.mean()
        df.loc[1, 'std'] = x.values.std()
        df.loc[1, 'max'] = x.values.max()
        df.loc[1, 'min'] = x.values.min()

        df.loc[1, 'q90'] = np.quantile(x.values, 0.90)
        df.loc[1, 'q95'] = np.quantile(x.values, 0.95)
        df.loc[1, 'q99'] = np.quantile(x.values, 0.99)
        df.loc[1, 'q05'] = np.quantile(x.values, 0.05)
        df.loc[1, 'q10'] = np.quantile(x.values, 0.10)
        df.loc[1, 'q01'] = np.quantile(x.values, 0.01)

        df.loc[1, 'abs_max'] = np.abs(x.values).max()
        df.loc[1, 'abs_mean'] = np.abs(x.values).mean()
        df.loc[1, 'abs_std'] = np.abs(x.values).std()
        df.loc[1, 'trend'] = Base._add_trend_feature(x.values)
        df.loc[1, 'abs_trend'] = Base._add_trend_feature(x.values, abs_values=True)

        # New features - rolling features
        for w in [10, 50, 100, 1000]:
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values
            x_roll_abs_mean = x.abs().rolling(w).mean().dropna().values

            df.loc[1, 'ave_roll_std_' + str(w)] = x_roll_std.mean()
            df.loc[1, 'std_roll_std_' + str(w)] = x_roll_std.std()
            df.loc[1, 'max_roll_std_' + str(w)] = x_roll_std.max()
            df.loc[1, 'min_roll_std_' + str(w)] = x_roll_std.min()

            df.loc[1, 'q01_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.01)
            df.loc[1, 'q05_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.05)
            df.loc[1, 'q10_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.10)
            df.loc[1, 'q95_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.95)
            df.loc[1, 'q99_roll_std_' + str(w)] = np.quantile(x_roll_std, 0.99)

            df.loc[1, 'ave_roll_mean_' + str(w)] = x_roll_mean.mean()
            df.loc[1, 'std_roll_mean_' + str(w)] = x_roll_mean.std()
            df.loc[1, 'max_roll_mean_' + str(w)] = x_roll_mean.max()
            df.loc[1, 'min_roll_mean_' + str(w)] = x_roll_mean.min()
            df.loc[1, 'q01_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.01)
            df.loc[1, 'q05_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.05)
            df.loc[1, 'q95_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.95)
            df.loc[1, 'q99_roll_mean_' + str(w)] = np.quantile(x_roll_mean, 0.99)
            df.loc[1, 'ave_roll_abs_mean_' + str(w)] = x_roll_abs_mean.mean()
            df.loc[1, 'std_roll_abs_mean_' + str(w)] = x_roll_abs_mean.std()
            df.loc[1, 'max_roll_abs_mean_' + str(w)] = x_roll_abs_mean.max()
            df.loc[1, 'min_roll_abs_mean_' + str(w)] = x_roll_abs_mean.min()
            df.loc[1, 'q01_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.01)
            df.loc[1, 'q05_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.05)
            df.loc[1, 'q95_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.95)
            df.loc[1, 'q99_roll_abs_mean_' + str(w)] = np.quantile(x_roll_abs_mean, 0.99)

        return df
