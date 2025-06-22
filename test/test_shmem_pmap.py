from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import pytest

from shmem_pmap import NpyArray, shmem_pmap


def do_work(arr):
    return arr + 1


@pytest.mark.parametrize("parallel, shape", [(4, 1024), (5, 1024), (4, (16, 64))])
@pytest.mark.parametrize("specified", [False, True])
def test_single_input(parallel, shape, specified):
    arr = np.arange(1024).reshape(shape)
    kwargs = {"rv_shape": arr.shape, "rv_dtype": arr.dtype} if specified else {}
    res = shmem_pmap(do_work, parallel=parallel)(arr, **kwargs)
    assert (res == arr + 1).all()


def do_work2(arr_pair):
    arr1, arr2 = arr_pair
    return arr1 + arr2


@pytest.mark.parametrize("container", [tuple, list])
@pytest.mark.parametrize("specified", [False, True])
def test_multi_input_sequence(container, specified):
    arr1 = np.arange(1024)
    arr2 = 1024 - arr1
    data = container([arr1, arr2])
    kwargs = {"rv_shape": 1024, "rv_dtype": arr1.dtype} if specified else {}
    res = shmem_pmap(do_work2, parallel=4)(data, **kwargs)
    assert (res == 1024).all()


def do_work3(data):
    return data["arr1"] + data["arr2"]


@pytest.mark.parametrize("specified", [False, True])
def test_multi_input_dict(specified):
    arr1 = np.arange(1024)
    arr2 = 1024 - arr1
    data = {"arr1": arr1, "arr2": arr2}
    kwargs = {"rv_shape": 1024, "rv_dtype": arr1.dtype} if specified else {}
    res = shmem_pmap(do_work3, parallel=4)(data, **kwargs)
    assert (res == 1024).all()


class Pair(NamedTuple):
    arr1: NpyArray
    arr2: NpyArray


def do_work4(data: Pair):
    return data.arr1 + data.arr2


@pytest.mark.parametrize("specified", [False, True])
def test_multi_input_namedtuple(specified):
    arr1 = np.arange(1024)
    arr2 = 1024 - arr1
    data = Pair(arr1, arr2)
    kwargs = {"rv_shape": 1024, "rv_dtype": arr1.dtype} if specified else {}
    res = shmem_pmap(do_work4, parallel=4)(data, **kwargs)
    assert (res == 1024).all()


@dataclass
class PairDC:
    arr1: NpyArray
    arr2: NpyArray


def do_work5(data: PairDC):
    return data.arr1 + data.arr2


@pytest.mark.parametrize("specified", [False, True])
def test_multi_input_dataclass(specified):
    arr1 = np.arange(1024)
    arr2 = 1024 - arr1
    data = PairDC(arr1, arr2)
    kwargs = {"rv_shape": 1024, "rv_dtype": arr1.dtype} if specified else {}
    res = shmem_pmap(do_work5, parallel=4)(data, **kwargs)
    assert (res == 1024).all()
