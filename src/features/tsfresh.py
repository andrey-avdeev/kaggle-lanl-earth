import numpy as np
import pandas as pd
from tsfresh.feature_extraction import extract_features, EfficientFCParameters


class Tsfresh:

    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        data = pd.DataFrame(dtype=np.float64)

        data['x'] = x
        data['id'] = 1

        df = extract_features(data, column_id='id', default_fc_parameters=EfficientFCParameters())

        return df
