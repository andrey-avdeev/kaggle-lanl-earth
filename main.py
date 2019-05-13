import logging
from src.config.data import FEATURES_BASE_DIR, FEATURES_BASE_DENOISE_DIR, FEATURES_TSFRESH_DIR, FEATURES_BASE_FILENAME, \
    FEATURES_BASE_DENOISE_FILENAME, FEATURES_TSFRESH_FILENAME, FEATURES_FOLDS_DENOISE_DIR, \
    FEATURES_FOLDS_DENOISE_FILENAME, FEATURES_SIGNAL_DIR, FEATURES_SIGNAL_FILENAME, FEATURES_WAVELET_DIR, \
    FEATURES_WAVELET_FILENAME, SMALL_ROWS_PER_SEGMENT, SMALL_ROWS_SEGMENT_ID_PREFIX, SMALL_ROWS_COUNT, ROWS_PER_SEGMENT, \
    STANDARD_ROWS_COUNT, STANDARD_ROWS_SEGMENT_ID_PREFIX
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
    print('build_features.base')
    extract_features(FEATURES_BASE_DIR, Base.features, is_test)

    print('build_features.base_denoise')
    extract_features(FEATURES_BASE_DENOISE_DIR, BaseDenoise.features, is_test)

    print('build_features.folds_denoise')
    extract_features(FEATURES_FOLDS_DENOISE_DIR, FoldsDenoise.features, is_test)

    print('build_features.signal')
    extract_features(FEATURES_SIGNAL_DIR, Signal.features, is_test)

    print('build_features.wavelet')
    extract_features(FEATURES_WAVELET_DIR, Wavelet.features, is_test)

    print('build_features.tsfresh')
    extract_features(FEATURES_TSFRESH_DIR, Tsfresh.features, is_test)

    print('reduce_features')
    reduce_features(FEATURES_BASE_DIR, FEATURES_BASE_FILENAME, is_test)
    reduce_features(FEATURES_BASE_DENOISE_DIR, FEATURES_BASE_DENOISE_FILENAME, is_test)
    reduce_features(FEATURES_FOLDS_DENOISE_DIR, FEATURES_FOLDS_DENOISE_FILENAME, is_test)
    reduce_features(FEATURES_SIGNAL_DIR, FEATURES_SIGNAL_FILENAME, is_test)
    reduce_features(FEATURES_WAVELET_DIR, FEATURES_WAVELET_FILENAME, is_test)
    reduce_features(FEATURES_TSFRESH_DIR, FEATURES_TSFRESH_FILENAME, is_test)


def main():
    print('Starting')
    # split_raw_train()
    # prepare_raw_test()

    # TRAIN
    # is_test = False
    # build_features(is_test)

    # TEST
    # is_test = True
    # build_features(is_test)

    # smaller train chunks size
    # split_raw_train(segment_size=SMALL_ROWS_PER_SEGMENT,
    #                 segment_id_prefix=SMALL_ROWS_SEGMENT_ID_PREFIX,
    #                 segments_count=SMALL_ROWS_COUNT)

    # TRAIN from small segments
    # is_test = False
    # build_features(is_test)

    # smaller train chunks size
    split_raw_train(segment_size=ROWS_PER_SEGMENT,
                    segment_id_prefix=STANDARD_ROWS_SEGMENT_ID_PREFIX,
                    segments_count=STANDARD_ROWS_COUNT)

    # TRAIN from standard random segments
    is_test = False
    build_features(is_test)


if __name__ == "__main__":
    main()
