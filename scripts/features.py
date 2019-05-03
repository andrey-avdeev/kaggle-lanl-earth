# @staticmethod
# def timeseries_from_segment_cache(segment_id: str, train: bool = True) -> pd.DataFrame:
#     log.debug(f'timeseries_from_segment_cache.call.segment_id={segment_id}')
#
#     start = time.time()
#
#     segment = FileUtils.get_segment_from_cache(segment_id, train)
#     if not train:
#         segment.columns = ['x']
#     # segment.columns = ['x','y']
#     extracted_features = FeatureUtils._timeseries(segment)
#
#     end = time.time()
#     log.info(f'timeseries_from_segment_cache.success.segment_id={segment_id},minutes={(end - start) / 60}')
#
#     return extracted_features


@staticmethod
def _timeseries_from_segment_cache_parallel_train(segment_id: int) -> None:
    FeatureUtils.timeseries_from_segment_cache_parallel(segment_id, train=True)


@staticmethod
def _timeseries_from_segment_cache_parallel_test(segment_id: str) -> None:
    FeatureUtils.timeseries_from_segment_cache_parallel(segment_id, train=False)


# @staticmethod
# def timeseries_from_segment_cache_parallel(segment_id: str, train: bool = True) -> None:
#     global segments_left
#
#     log.debug(f'timeseries_from_segment_cache_parallel.call.segment_id={segment_id}')
#
#     if not FileUtils.check_features_in_cache(segment_id, train):
#         log.debug(f'timeseries_from_segment_cache_parallel.not_found.segment_id={segment_id}')
#
#         extracted_features = None
#         try:
#             extracted_features = FeatureUtils.timeseries_from_segment_cache(segment_id, train)
#         except Exception as e:
#             log.exception('extract_feature.error')
#             log.error(e)
#
#         try:
#             if extracted_features is not None:
#                 FileUtils.set_features_to_cache(extracted_features, segment_id, train)
#                 segments_left -= 1
#                 log.info(f'segments_left={segments_left} - extracted')
#         except Exception as e:
#             log.exception('set_feature_to_cache.error')
#             log.error(e)
#     else:
#         log.debug(f'timeseries_from_segment_cache_parallel.found_and_skip.segment_id={segment_id}')
#         segments_left -= 1
#         log.info(f'segments_left={segments_left} - found_and_skip')


# @staticmethod
# def timeseries_parallel(train: bool = True):
#     global segments_left
#
#     log.info(f'timeseries_parallel.call.train={train}')
#
#     pool = None
#
#     # if POOL_TYPE == 'process':
#     #     num_procs = multiprocessing.cpu_count() - LOCK_PROCS
#     #     log.info(f'timeseries_parallel.num_procs={num_procs},train={train}')
#     #     pool = Pool(num_procs)
#     # elif POOL_TYPE == 'thread':
#     #     threads = MAX_THREADS
#     #     pool = ThreadPool(processes=threads)
#
#     segments_count = FileUtils.get_cache_segments_count(train)
#     segments_left = segments_count
#
#     log.info(f'timeseries_parallel.segments_count={segments_count},train={train}')
#
#     segment_ids = get_segments_ids_from_count(segments_count)
#
#     log.info(f'timeseries_parallel.pool_start.train={train}')
#     if train:
#         pool.map(FeatureUtils._timeseries_from_segment_cache_parallel_train,
#                  [segment_id for segment_id in segment_ids])
#     else:
#         pool.map(FeatureUtils._timeseries_from_segment_cache_parallel_test,
#                  [segment_id for segment_id in segment_ids])


@staticmethod
def timeseries_parallel_test(train: bool = False):
    global segments_left

    log.info(f'timeseries_parallel.call.train={train}')

    pool = None

    if POOL_TYPE == 'process':
        num_procs = multiprocessing.cpu_count() - LOCK_PROCS
        log.info(f'timeseries_parallel.num_procs={num_procs},train={train}')
        pool = Pool(num_procs)
    elif POOL_TYPE == 'thread':
        threads = MAX_THREADS
        pool = ThreadPool(processes=threads)

    segments_filenames = FileUtils.get_file_names(CACHE_DIR + TEST_DIR + SEGMENTS_DIR)
    segments_count = len(segments_filenames)
    segments_left = segments_count

    log.info(f'timeseries_parallel.segments_count={segments_count},train={train}')
    log.info(f'timeseries_parallel.pool_start.train={train}')
    pool.map(FeatureUtils._timeseries_from_segment_cache_parallel_test,
             [filename.replace("seg_", "").replace(".csv", "") for filename in segments_filenames])