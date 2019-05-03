import numpy as np
import pywt
from numpy.core.multiarray import ndarray
from scipy.signal import butter, sosfilt


def maddest(d, axis=None) -> float:
    """
    Mean Absolute Deviation
    """

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def high_pass_filter(x: ndarray, n: int = 10, low_cutoff: int = 1000, sample_rate: int = 4000) -> ndarray:
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist

    sos = butter(n, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = sosfilt(sos, x)

    return filtered_sig


def wavelet_filter(x: ndarray, wavelet: str = 'db4', level: int = 1) -> ndarray:
    coeff = pywt.wavedec(x, wavelet, mode="per")

    sigma = (1 / 0.6745) * maddest(coeff[-level])

    # Calculate the univeral threshold
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')
