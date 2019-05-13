import numpy as np
import pywt
from numpy.core.multiarray import ndarray
import scipy.signal as sg
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


def classic_sta_lta(a: ndarray, nsta: int, nlta: int) -> float:
    """
    Computes the standard STA/LTA from a given input array a. The length of
    the STA is given by nsta in samples, respectively is the length of the
    LTA given by nlta in samples. Written in Python.
    .. note::
        There exists a faster version of this trigger wrapped in C
        called :func:`~obspy.signal.trigger.classic_sta_lta` in this module!
    :type a: NumPy :class:`~numpy.ndarray`
    :param a: Seismic Trace
    :type nsta: int
    :param nsta: Length of short time average window in samples
    :type nlta: int
    :param nlta: Length of long time average window in samples
    :rtype: NumPy :class:`~numpy.ndarray`
    :return: Characteristic function of classic STA/LTA
    """
    # The cumulative sum can be exploited to calculate a moving average (the
    # cumsum function is quite efficient)
    sta = np.cumsum(a ** 2)

    # Convert to float
    sta = np.require(sta, dtype=np.float)

    # Copy for LTA
    lta = sta.copy()

    # Compute the STA and the LTA
    sta[nsta:] = sta[nsta:] - sta[:-nsta]
    sta /= nsta
    lta[nlta:] = lta[nlta:] - lta[:-nlta]
    lta /= nlta

    # Pad zeros
    sta[:nlta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float
    dtiny = np.finfo(0.0).tiny
    idx = lta < dtiny
    lta[idx] = dtiny

    return sta / lta


NY_FREQ_IDX = 75000
CUTOFF = 18000


def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter
    b, a = sg.butter(4, Wn=cutoff / NY_FREQ_IDX)
    return b, a


def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter
    b, a = sg.butter(4, Wn=cutoff / NY_FREQ_IDX, btype='highpass')
    return b, a


def des_bw_filter_bp(low, high):  # band pass filter
    b, a = sg.butter(4, Wn=(low / NY_FREQ_IDX, high / NY_FREQ_IDX), btype='bandpass')
    return b, a
