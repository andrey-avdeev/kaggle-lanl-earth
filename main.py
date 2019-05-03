import logging
from src.config.data import FEATURES_BASE_DIR, FEATURES_BASE_DENOISE_DIR, FEATURES_TSFRESH_DIR, FEATURES_BASE_FILENAME, \
    FEATURES_BASE_DENOISE_FILENAME, FEATURES_TSFRESH_FILENAME
from src.data.map_reduce import extract_features, reduce_features, split_raw_train, prepare_raw_test
from src.features.base import Base
from src.features.base_denoise import BaseDenoise
from src.features.tsfresh import Tsfresh

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)


def build_features(is_test: bool) -> None:
    # extract_features(FEATURES_BASE_DIR, Base.features, is_test)
    # reduce_features(FEATURES_BASE_DIR, FEATURES_BASE_FILENAME, is_test)

    extract_features(FEATURES_BASE_DENOISE_DIR, BaseDenoise.features, is_test)
    # reduce_features(FEATURES_BASE_DENOISE_DIR, FEATURES_BASE_DENOISE_FILENAME, is_test)

    # extract_features(FEATURES_TSFRESH_DIR, Tsfresh.features, is_test)
    # reduce_features(FEATURES_TSFRESH_DIR, FEATURES_TSFRESH_FILENAME, is_test)


def main():
    print('Starting')
    # split_raw_train()
    # prepare_raw_test()

    # TRAIN
    # is_test = False
    # build_features(is_test)

    # TEST
    is_test = True
    build_features(is_test)


if __name__ == "__main__":
    main()
