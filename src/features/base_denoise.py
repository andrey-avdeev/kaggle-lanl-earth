import pandas as pd
from src.utils.denoise import wavelet_filter
from src.features.base import Base


class BaseDenoise:

    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        x_array = wavelet_filter(x.values, wavelet='haar', level=1)
        x = pd.Series(x_array)

        return Base.features(x)
