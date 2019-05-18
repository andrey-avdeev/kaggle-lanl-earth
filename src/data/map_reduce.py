import logging
import os
import time
from typing import Iterator, Tuple, Callable, Optional
from functools import partial
import numpy as np
import pandas as pd

from src.config.data import ROWS_PER_SEGMENT, DATA_DIR, RAW_DIR, TRAIN_DIR, RAW_TRAIN_TYPES, INTERIM_DIR, SEGMENTS_DIR, \
    SEGMENT_FILENAME, FEATURE_FILENAME, INTERIM_TRAIN_TYPES, PROCESSED_DIR, TEST_DIR
from src.utils.file import file_names
from src.utils.parallel import get_pool

log = logging.getLogger(__name__)
segments_left = None


def map_df(df: pd.DataFrame,
           segment_size: int = ROWS_PER_SEGMENT,
           prefix: str = '',
           segments_count: int = None) -> Iterator[Tuple[int, pd.DataFrame]]:
    if segments_count and segments_count > 0:
        # random sampling
        min_start_index = 0
        max_start_index = df.shape[0] - segment_size - 1

        start_indices = np.random.randint(min_start_index, max_start_index, size=segments_count, dtype=np.int32)

        for start_index in start_indices:
            segment = df.iloc[start_index:start_index + segment_size]
            yield prefix + str(start_index), segment
    else:
        total_segments = int(np.floor(df.shape[0] / segment_size))

        # iterative sampling
        for segment_id in range(total_segments):
            segment = df.iloc[segment_id * segment_size:segment_id * segment_size + segment_size]
            yield prefix + str(segment_id), segment


def map_series(x: pd.Series, folds: int = 3) -> Iterator[Tuple[int, pd.Series]]:
    rows_per_slice = int(np.floor(x.size / folds))

    for slice_id in range(folds):
        slice = x[slice_id * rows_per_slice:slice_id * rows_per_slice + rows_per_slice]
        yield slice_id, slice


def split_raw_train(segment_size: Optional[int] = None,
                    segment_id_prefix: Optional[str] = None,
                    segments_count: Optional[int] = None) -> None:
    log.info(f'train_df.loading')

    raw_train_path = f'{DATA_DIR}{RAW_DIR}{TRAIN_DIR}train.csv'

    train_df = pd.read_csv(raw_train_path, dtype=RAW_TRAIN_TYPES)
    log.info(f'train_df.loaded')

    train_df.columns = ['x', 'y']

    for segment_id, segment_df in map_df(train_df, segment_size, segment_id_prefix, segments_count):
        seg_path = f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{SEGMENTS_DIR}{SEGMENT_FILENAME}'.format(id=segment_id)
        segment_df.to_csv(seg_path, index=False)
        log.info(f'{segment_id}.saved')


def prepare_raw_test() -> None:
    global segments_left

    segment_names = file_names(f'{DATA_DIR}{RAW_DIR}{TEST_DIR}')
    segments_left = len(segment_names)

    for file_name in segment_names:
        df = pd.read_csv(f'{DATA_DIR}{RAW_DIR}{TEST_DIR}{file_name}')
        df.columns = ['x']
        df.to_csv(f'{DATA_DIR}{INTERIM_DIR}{TEST_DIR}{SEGMENTS_DIR}{file_name}', index=False)

        segments_left -= 1
        log.info(f'{file_name}.saved.segments_left={segments_left}')


def process_feature(id: str,
                    get_features: Callable[[pd.Series], pd.DataFrame],
                    env_dir: str,
                    feature_dir: str,
                    is_test: bool) -> None:
    global segments_left
    feature_path = f'{DATA_DIR}{INTERIM_DIR}{env_dir}{feature_dir}{FEATURE_FILENAME}'.format(id=id)

    if os.path.isfile(feature_path):
        segments_left -= 1
        log.info(f'skip.segments_left={segments_left}')
    else:
        segment_path = f'{DATA_DIR}{INTERIM_DIR}{env_dir}{SEGMENTS_DIR}{SEGMENT_FILENAME}'.format(id=id)
        try:
            start = time.time()
            segment = pd.read_csv(segment_path, dtype=INTERIM_TRAIN_TYPES)

            features = get_features(segment['x'])

            if not is_test:
                features['y'] = segment['y'].values[-1]

            features.to_csv(feature_path, index=False)

            end = time.time()

            segments_left -= 1
            log.info(f'extracted.segments_left={segments_left},minutes={(end - start) / 60}')
        except Exception as e:
            log.exception('extract_feature.error')
            log.error(e)


def extract_features(feature_dir: str,
                     get_features: Callable[[pd.Series], pd.DataFrame],
                     is_test: bool = False) -> None:
    global segments_left
    pool = get_pool()

    env_dir = TEST_DIR if is_test else TRAIN_DIR

    segments_file_names = file_names(f'{DATA_DIR}{INTERIM_DIR}{env_dir}{SEGMENTS_DIR}')
    segments_left = len(segments_file_names)

    segment_ids = [file_name.replace("seg_", "").replace(".csv", "") for file_name in segments_file_names]

    process_feature_partial = partial(process_feature,
                                      get_features=get_features,
                                      env_dir=env_dir,
                                      feature_dir=feature_dir,
                                      is_test=is_test)

    pool.map(process_feature_partial, [segment_id for segment_id in segment_ids])
    log.info('paralle_processing.done')


def reduce_features(features_dir: str,
                    features_filename: str,
                    is_test: bool = False) -> None:
    global segments_left

    env_dir = TEST_DIR if is_test else TRAIN_DIR

    feature_file_names = file_names(f'{DATA_DIR}{INTERIM_DIR}{env_dir}{features_dir}')
    segments_left = len(feature_file_names)

    df = pd.DataFrame()

    for file_name in feature_file_names:
        feature_df = pd.read_csv(f'{DATA_DIR}{INTERIM_DIR}{env_dir}{features_dir}{file_name}')
        feature_df['id'] = file_name.replace("feature_", "").replace(".csv", "")
        df = df.append(feature_df)
        segments_left -= 1
        log.info(f'{features_dir},features_left={segments_left}')

    df.index = df['id'].values
    df.drop('id', axis=1, inplace=True)
    df = df.sort_index()

    df.to_csv(f'{DATA_DIR}{PROCESSED_DIR}{env_dir}{features_filename}', index=True, index_label='id')
