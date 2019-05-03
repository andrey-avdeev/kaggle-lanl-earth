import multiprocessing
from multiprocessing.dummy import Pool
from multiprocessing.pool import ThreadPool

from src.config.common import POOL_TYPE, LOCK_PROCESSES, MAX_THREADS


def get_pool():
    if POOL_TYPE == 'process':
        num_proccesses = multiprocessing.cpu_count() - LOCK_PROCESSES

        return Pool(num_proccesses)
    elif POOL_TYPE == 'thread':
        return ThreadPool(processes=MAX_THREADS)
    else:
        raise ValueError('undefined.POOL_TYPE')
