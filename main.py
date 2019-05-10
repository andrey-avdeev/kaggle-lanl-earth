import logging
from src.config.data import FEATURES_BASE_DIR, FEATURES_BASE_DENOISE_DIR, FEATURES_TSFRESH_DIR, FEATURES_BASE_FILENAME, \
    FEATURES_BASE_DENOISE_FILENAME, FEATURES_TSFRESH_FILENAME, FEATURES_FOLDS_DENOISE_DIR, \
    FEATURES_FOLDS_DENOISE_FILENAME, FEATURES_SIGNAL_DIR, FEATURES_SIGNAL_FILENAME, FEATURES_WAVELET_DIR, \
    FEATURES_WAVELET_FILENAME
from src.data.map_reduce import extract_features, reduce_features, split_raw_train, prepare_raw_test
from src.features.base import Base
from src.features.base_denoise import BaseDenoise
from src.features.tsfresh import Tsfresh
from src.features.folds_denoise import FoldsDenoise
from src.features.signal import Signal
from src.features.wavelet import Wavelet

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def build_features(is_test: bool) -> None:
    print('build_features')
    # extract_features(FEATURES_BASE_DIR, Base.features, is_test)
    # reduce_features(FEATURES_BASE_DIR, FEATURES_BASE_FILENAME, is_test)

    # extract_features(FEATURES_BASE_DENOISE_DIR, BaseDenoise.features, is_test)
    # reduce_features(FEATURES_BASE_DENOISE_DIR, FEATURES_BASE_DENOISE_FILENAME, is_test)

    # extract_features(FEATURES_TSFRESH_DIR, Tsfresh.features, is_test)
    # reduce_features(FEATURES_TSFRESH_DIR, FEATURES_TSFRESH_FILENAME, is_test)

    # extract_features(FEATURES_FOLDS_DENOISE_DIR, FoldsDenoise.features, is_test)
    # reduce_features(FEATURES_FOLDS_DENOISE_DIR, FEATURES_FOLDS_DENOISE_FILENAME, is_test)

    # extract_features(FEATURES_SIGNAL_DIR, Signal.features, is_test)
    # reduce_features(FEATURES_SIGNAL_DIR, FEATURES_SIGNAL_FILENAME, is_test)

    extract_features(FEATURES_WAVELET_DIR, Wavelet.features, is_test)
    reduce_features(FEATURES_WAVELET_DIR, FEATURES_WAVELET_FILENAME, is_test)


def main():
    print('Starting')
    # split_raw_train()
    # prepare_raw_test()

    # TRAIN
    is_test = False
    build_features(is_test)

    # TEST
    is_test = True
    build_features(is_test)


if __name__ == "__main__":
    main()
