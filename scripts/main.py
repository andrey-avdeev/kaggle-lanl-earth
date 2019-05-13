# import os
# import time
# from typing import Callable
#
# import pandas as pd
# from src.config.data import DATA_DIR, RAW_DIR, TRAIN_DIR, RAW_TRAIN_TYPES, INTERIM_DIR, SEGMENTS_DIR, SEGMENT_FILENAME, \
#     FEATURES_BASE_DIR, FEATURE_FILENAME, INTERIM_TRAIN_TYPES, FEATURES_BASE_DENOISE_DIR, FEATURES_TSFRESH_DIR, \
#     PROCESSED_DIR, FEATURES_BASE_FILENAME, FEATURES_BASE_DENOISE_FILENAME, FEATURES_TSFRESH_FILENAME
# from src.data.map_reduce import map_df
# from src.utils.parallel import get_pool
# from src.utils.file import file_names
# from src.features.base import Base
# from src.features.base_denoise import BaseDenoise
# from src.features.tsfresh import Tsfresh
#
#
# def split_raw_train() -> None:
#     raw_train_path = f'{DATA_DIR}{RAW_DIR}{TRAIN_DIR}train.csv'
#
#     train_df = pd.read_csv(raw_train_path, dtype=RAW_TRAIN_TYPES)
#     train_df.columns = ['x', 'y']
#     train_df['id'] = 1
#
#     for segment_id, segment_df in map_df(train_df):
#         seg_path = f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{SEGMENTS_DIR}{SEGMENT_FILENAME}'.format(id=segment_id)
#         segment_df.to_csv(seg_path)
#
#
# segments_left = None
#
#
# def train_features(feature_dir: str, extract_features: Callable[[pd.Series], pd.DataFrame]) -> None:
#     global segments_left
#
#     pool = get_pool()
#
#     segments_file_names = file_names(f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{SEGMENTS_DIR}')
#     segments_left = len(segments_file_names)
#
#     segment_ids = [file_name.replace("seg_", "").replace(".csv", "") for file_name in segments_file_names]
#
#     def process_feature(id: str):
#         feature_path = f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{feature_dir}{FEATURE_FILENAME}'.format(id=id)
#
#         if os.path.isfile(feature_path):
#             segments_left -= 1
#             log.info(f'skip.segments_left={segments_left}')
#         else:
#             segment_path = f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{SEGMENTS_DIR}{SEGMENT_FILENAME}'.format(id=id)
#             features = None
#             try:
#                 start = time.time()
#                 segment = pd.read_csv(segment_path, dtype=INTERIM_TRAIN_TYPES)
#
#                 features = extract_features(segment['x'])
#                 features['y'] = segment['y'].values[-1]
#
#                 features.to_csv(feature_path)
#
#                 end = time.time()
#
#                 segments_left -= 1
#                 log.info(f'extracted.segments_left={segments_left},minutes={(end - start) / 60}')
#             except Exception as e:
#                 log.exception('extract_feature.error')
#                 log.error(e)
#
#     pool.map(process_feature, [segment_id for segment_id in segment_ids])
#
#
# def train_base_features() -> None:
#     train_features(FEATURES_BASE_DIR, Base.features)
#
#
# def train_base_denoise_features() -> None:
#     train_features(FEATURES_BASE_DENOISE_DIR, BaseDenoise.features)
#
#
# def train_tsfresh_features() -> None:
#     train_features(FEATURES_TSFRESH_DIR, Tsfresh.features)
#
#
# # def reduce_train_features(features_dir: str, features_filename: str) -> None:
# #     feature_file_names = file_names(f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{features_dir}')
# #
# #     df = pd.DataFrame()
# #
# #     for file_name in feature_file_names:
# #         feature_df = pd.read_csv(f'{DATA_DIR}{INTERIM_DIR}{TRAIN_DIR}{features_dir}{file_name}')
# #         feature_df['id'] = file_name.replace("feature_", "").replace(".csv", "")
# #         df = df.append(feature_df)
# #
# #     df.index = df['id'].values
# #     df = df.sort_index()
# #
# #     df.to_csv(f'{DATA_DIR}{PROCESSED_DIR}{TRAIN_DIR}{features_filename}', index=True, index_label='id')
#
#
# def reduce_train_base_features():
#     reduce_train_features(FEATURES_BASE_DIR, FEATURES_BASE_FILENAME)
#
#
# def reduce_train_base_denoise_features():
#     reduce_train_features(FEATURES_BASE_DENOISE_DIR, FEATURES_BASE_DENOISE_FILENAME)
#
#
# def reduce_train_tsfresh_features():
#     reduce_train_features(FEATURES_TSFRESH_DIR, FEATURES_TSFRESH_FILENAME)
