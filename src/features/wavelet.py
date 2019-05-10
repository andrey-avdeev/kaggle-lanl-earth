import numpy as np
import pandas as pd
from src.utils.denoise import des_bw_filter_lp
import scipy.signal as sg


class Wavelet:

    @staticmethod
    def features(x: pd.Series) -> pd.DataFrame:
        X = pd.DataFrame(dtype=np.float64)
        seg_id = 1

        sig = x
        b, a = des_bw_filter_lp(cutoff=18000)
        sig = sg.lfilter(b, a, sig)

        peakind = []
        noise_pct = .001
        count = 0

        while len(peakind) < 12 and count < 24:
            peakind = sg.find_peaks_cwt(sig, np.arange(1, 16), noise_perc=noise_pct, min_snr=4.0)
            noise_pct *= 2.0
            count += 1

        if len(peakind) < 12:
            print('Warning: Failed to find 12 peaks for %d' % seg_id)

        while len(peakind) < 12:
            peakind.append(149999)

        df_pk = pd.DataFrame(data={'pk': sig[peakind], 'idx': peakind}, columns=['pk', 'idx'])
        df_pk.sort_values(by='pk', ascending=False, inplace=True)

        for i in range(0, 12):
            X.loc[seg_id, 'pk_idx_%d' % i] = df_pk['idx'].iloc[i]
            X.loc[seg_id, 'pk_val_%d' % i] = df_pk['pk'].iloc[i]

        return X
