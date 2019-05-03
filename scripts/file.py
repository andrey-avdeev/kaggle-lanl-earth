import os
from os import listdir
from os.path import isfile, join
import logging
from typing import Optional, List

import pandas as pd
import pickle

from src.settings import DATA_DIR, TRAIN_FILE, TRAIN_ROWS, SEGMENT_ROWS, TRAIN_SEGMENT_FILE_PATH_TEMPLATE, \
    TRAIN_SEGMENTS_COUNT_FILE_PATH, TRAIN_FEATURES_FILE_PATH_TEMPLATE, TEST_SEGMENTS_COUNT_FILE_PATH, \
    TEST_SEGMENT_FILE_PATH_TEMPLATE, TEST_FEATURES_FILE_PATH_TEMPLATE, CACHE_DIR, TRAIN_DIR, FEATURES_DIR, TEST_DIR
from src.data import DATA_TYPES, CACHE_DATA_TYPES, TEST_DATA_TYPES
from src.utils.common import get_segments_ids, get_segments_count

log = logging.getLogger(__name__)


class FileUtils:
    # @staticmethod
    # def train_data(nrows: int = None) -> pd.DataFrame:
    #     if not nrows:
    #         nrows = TRAIN_ROWS
    #
    #     train_path = f'{DATA_DIR}{TRAIN_FILE}'
    #
    #     log.info(f'file.train_data.nrows={nrows},file_path={train_path},data_type={DATA_TYPES}')
    #
    #     return pd.read_csv(train_path,
    #                        nrows=nrows,
    #                        dtype=DATA_TYPES)

    # @staticmethod
    # def split_and_cache(df: pd.DataFrame, train: bool = True):
    #     segments_ids = get_segments_ids(df)
    #     segments_count = get_segments_count(df)
    #     FileUtils.set_cache_segments_count(segments_count, train)
    #
    #     for segment_id in segments_ids:
    #         segment = FileUtils.get_segment_from_df(df, segment_id)
    #         if train:
    #             segment_path = TRAIN_SEGMENT_FILE_PATH_TEMPLATE.format(segment_id=segment_id)
    #         else:
    #             segment_path = TEST_SEGMENT_FILE_PATH_TEMPLATE.format(segment_id=segment_id)
    #
    #         FileUtils.set_cache(segment, segment_path)

    @staticmethod
    def get_segment_from_df(df: pd.DataFrame, segment_id: int) -> pd.DataFrame:
        return df.iloc[segment_id * SEGMENT_ROWS:segment_id * SEGMENT_ROWS + SEGMENT_ROWS]

    # @staticmethod
    # def get_segment_from_cache(segment_id: str, train: bool = True) -> Optional[pd.DataFrame]:
    #     if train:
    #         segment_path = TRAIN_SEGMENT_FILE_PATH_TEMPLATE.format(segment_id=segment_id)
    #         return FileUtils.get_cache(segment_path)
    #     else:
    #         segment_path = TEST_SEGMENT_FILE_PATH_TEMPLATE.format(segment_id=segment_id)
    #         return FileUtils.get_cache_test(segment_path)

    @staticmethod
    def get_features_from_cache(segment_id: int, train: bool = True) -> Optional[pd.DataFrame]:
        if train:
            features_path = TRAIN_FEATURES_FILE_PATH_TEMPLATE.format(segment_id=segment_id)
        else:
            features_path = TEST_FEATURES_FILE_PATH_TEMPLATE.format(segment_id=segment_id)

        return FileUtils.get_cache(features_path)


    @staticmethod
    def get_cache(file_path: str) -> Optional[pd.DataFrame]:
        if os.path.isfile(file_path):
            return pd.read_csv(file_path, dtype=CACHE_DATA_TYPES)
        else:
            return None

    @staticmethod
    def get_cache_test(file_path: str) -> Optional[pd.DataFrame]:
        if os.path.isfile(file_path):
            return pd.read_csv(file_path, dtype=TEST_DATA_TYPES)
        else:
            return None

    @staticmethod
    def set_cache_segments_count(segments_count: int, train: bool = True):
        if train:
            file_path = TRAIN_SEGMENTS_COUNT_FILE_PATH
        else:
            file_path = TEST_SEGMENTS_COUNT_FILE_PATH
        with open(file_path, 'wb') as fobj:
            pickle.dump(segments_count, fobj)

    @staticmethod
    def get_cache_segments_count(train: bool = True) -> int:
        if train:
            file_path = TRAIN_SEGMENTS_COUNT_FILE_PATH
        else:
            file_path = TEST_SEGMENTS_COUNT_FILE_PATH

        with open(file_path, 'rb') as fobj:
            return pickle.load(fobj)


    @staticmethod
    def get_feature_file_names(folder_path: str, train: bool = True) -> List[str]:

        file_names = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

        return file_names

    @staticmethod
    def get_feature_dataframe(train: bool = True) -> pd.DataFrame:
        log.debug(f'file.get_feature_dataframe.train={train}')

        if train:
            folder = CACHE_DIR + TRAIN_DIR + FEATURES_DIR
        else:
            folder = CACHE_DIR + TEST_DIR + FEATURES_DIR
        log.info(f'file.get_feature_dataframe.folder={folder}')

        file_names = FileUtils.get_feature_file_names(folder, train)

        file_names_count = len(file_names)

        log.info(f'file.get_feature_dataframe.total_files={file_names_count}')

        df = pd.DataFrame()
        for file_name in file_names:
            log.debug(f'file.get_feature_dataframe.read.file_name={file_name}.call')
            df_seg = pd.read_csv(folder + file_name)
            df_seg['id'] = file_name.replace("feature_", "").replace(".csv", "")
            df = df.append(df_seg)
            log.debug(f'file.get_feature_dataframe.read.file_name={file_name}.success')
            file_names_count -= 1
            log.info(f'file.get_feature_dataframe.total_files={file_names_count}')

        return df
