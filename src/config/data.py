import numpy as np

## COMMON
DATA_DIR = 'data/'
TRAIN_DIR = 'train/'
TEST_DIR = 'test/'

## RAW DATA
RAW_DIR = 'raw/'

RAW_TRAIN_TYPES = {
    'acoustic_data': np.int16,
    'time_to_failure': np.float64,
}

RAW_TEST_TYPES = {
    'acoustic_data': np.int16,
}

## INTERIM DATA
INTERIM_DIR = 'interim/'

FEATURES_BASE_DIR = 'features_base/'
FEATURES_BASE_DENOISE_DIR = 'features_base_denoise/'
FEATURES_TSFRESH_DIR = 'features_tsfresh/'
FEATURES_FOLDS_DENOISE_DIR = 'features_folds_denoise/'
FEATURES_SIGNAL_DIR = 'features_signal/'

SEGMENTS_DIR = 'segments/'

SEGMENT_FILENAME = 'seg_{id}.csv'
FEATURE_FILENAME = 'feature_{id}.csv'

FEATURES_BASE_FILENAME = 'features_base.csv'
FEATURES_BASE_DENOISE_FILENAME = 'features_base_denoise.csv'
FEATURES_TSFRESH_FILENAME = 'features_tsfresh.csv'
FEATURES_FOLDS_DENOISE_FILENAME = 'features_folds_denoise.csv'
FEATURES_SIGNAL_FILENAME = 'features_signal.csv'


ROWS_PER_SEGMENT = 150000

INTERIM_TRAIN_TYPES = {
    'x': np.int16,
    'y': np.float64,
}

INTERIM_TEST_TYPES = {
    'x': np.int16,
}

# FOLDS_DENOISE features
SERIES_FOLDS = 10

## PROCESSED DATA
PROCESSED_DIR = 'processed/'
