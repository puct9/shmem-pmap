import numpy as np

from shmem_pmap import shmem_pmap


def do_work(arr):
    return arr + 1


def test_divisible():
    arr = np.arange(1024)
    res = shmem_pmap(do_work, parallel=4)(arr)
    assert (res == arr + 1).all()


def test_divisible_specified():
    arr = np.arange(1024)
    res = shmem_pmap(do_work, parallel=4)(arr, rv_shape=(1024,), rv_dtype=arr.dtype)
    assert (res == arr + 1).all()


def test_indivisible():
    arr = np.arange(1024)
    res = shmem_pmap(do_work, parallel=5)(arr)
    assert (res == arr + 1).all()


def test_indivisible_specified():
    arr = np.arange(1024)
    res = shmem_pmap(do_work, parallel=5)(arr, rv_shape=(1024,), rv_dtype=arr.dtype)
    assert (res == arr + 1).all()


def test_multidimensional():
    arr = np.arange(1024).reshape(16, 64)
    res = shmem_pmap(do_work, parallel=4)(arr)
    assert (res == arr + 1).all()


def test_multidimensional_specified():
    arr = np.arange(1024).reshape(16, 64)
    res = shmem_pmap(do_work, parallel=4)(arr, rv_shape=(16, 64), rv_dtype=arr.dtype)
    assert (res == arr + 1).all()


def do_work2(arr1_2):
    arr1, arr2 = arr1_2
    return arr1 + arr2


def test_multi_input_tuple():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work2, parallel=4)((arr1, arr2))
    assert (res == 1024).all()


def test_multi_input_tuple_specified():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work2, parallel=4)((arr1, arr2), rv_shape=(1024,), rv_dtype=arr1.dtype)
    assert (res == 1024).all()


def test_multi_input_list():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work2, parallel=4)([arr1, arr2])
    assert (res == 1024).all()


def test_multi_input_list_specified():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work2, parallel=4)([arr1, arr2], rv_shape=(1024,), rv_dtype=arr1.dtype)
    assert (res == 1024).all()


def do_work3(data):
    print(f"{data=}")
    return data["arr1"] + data["arr2"]


def test_multi_input_dict():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work3, parallel=4)({"arr1": arr1, "arr2": arr2})
    assert (res == 1024).all()


def test_multi_input_dict_specified():
    arr1 = np.arange(1024)
    arr2 = 1024 - np.arange(1024)
    res = shmem_pmap(do_work3, parallel=4)({"arr1": arr1, "arr2": arr2}, rv_shape=(1024,), rv_dtype=arr1.dtype)
    assert (res == 1024).all()
