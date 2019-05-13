import pandas as pd
from src.data.map_reduce import map_series
from src.config.data import SERIES_FOLDS
from src.features.base_denoise import BaseDenoise


class FoldsDenoise:
    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame()

        for fold_id, fold in map_series(x, SERIES_FOLDS):
            fold_features = BaseDenoise.features(fold)
            fold_features = fold_features.add_prefix(f'f_{fold_id}_')

            df = pd.concat([df, fold_features], ignore_index=False, sort=False, axis=1)

        return df
