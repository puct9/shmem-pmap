from time import perf_counter

import numpy as np

from shmem_pmap import shmem_pmap

SIZE = 1024 * 1024 * 128


def single_thread():
    rng = np.random.RandomState(0)
    print("Creating array")
    arr = rng.randn(SIZE).astype(np.float32)
    print("Computing bins")
    start = perf_counter()
    bins = np.linspace(-1, 1, 51)
    res = np.digitize(arr, bins)
    end = perf_counter()
    print(f"{res=}")
    print(f"{end - start:.2f}s")


def _multi_thread_do_work(arr):
    bins = np.linspace(-1, 1, 51)
    return np.digitize(arr, bins)


def multi_thread():
    rng = np.random.RandomState(0)
    print("Creating array")
    arr = rng.randn(SIZE).astype(np.float32)
    print("Computing bins")
    start = perf_counter()
    res = shmem_pmap(_multi_thread_do_work, parallel=4)(arr, rv_shape=arr.shape, rv_dtype=np.int64)
    end = perf_counter()
    print(f"{res=}")
    print(f"{end - start:.2f}s")


if __name__ == "__main__":
    single_thread()
    multi_thread()
