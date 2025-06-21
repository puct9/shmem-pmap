import collections
from functools import wraps
from multiprocessing.shared_memory import SharedMemory
from typing import Any, Callable, Generic, TypeAlias, TypeVar, overload

import numpy as np
import numpy.typing as npt
from tqdm.contrib.concurrent import process_map

# fmt: off
T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
U = TypeVar("U")

PyTree: TypeAlias = (
    T
    | list["PyTree[T]"]
    | tuple["PyTree[T]", ...]
    | dict[object, "PyTree[T]"]
)
# fmt: on
NpyArray: TypeAlias = np.ndarray[tuple[Any, ...], np.dtype[Any]]


@overload
def tree_map(fn: Callable[[T], U], head: PyTree[T]) -> PyTree[U]: ...
@overload
def tree_map(fn: Callable[[T, T1], U], head: PyTree[T], rest1: PyTree[T1]) -> PyTree[U]: ...
@overload
def tree_map(fn: Callable[[T, T1, T2], U], head: PyTree[T], rest1: PyTree[T1], rest2: PyTree[T2]) -> PyTree[U]: ...


def tree_map(fn, head, *rest):
    if isinstance(head, list):
        assert all(type(head) is type(r) for r in rest)
        return [tree_map(fn, *args) for args in zip(head, *rest, strict=True)]
    elif isinstance(head, tuple):
        assert all(type(head) is type(r) for r in rest)
        return tuple(tree_map(fn, *args) for args in zip(head, *rest, strict=True))
    elif isinstance(head, collections.abc.Mapping):
        assert all(type(head) is type(r) for r in rest)
        assert all(sorted(head) == sorted(r) for r in rest)
        return {key: tree_map(fn, head[key], *[r[key] for r in rest]) for key in head}
    else:
        return fn(head, *rest)


U_np = TypeVar("U_np", bound=np.generic)


def shmem_pmap(fn: Callable[[PyTree[NpyArray]], npt.NDArray[U_np]], *, parallel: int):
    @wraps(fn)
    def wrapped(inp: PyTree[NpyArray]) -> npt.NDArray[U_np]:
        inp_lengths = []
        tree_map(lambda arr: inp_lengths.append(len(arr)), inp)
        assert all(inp_lengths[0] == s for s in inp_lengths)

        shm_inp = tree_map(lambda arr: SharedMemory(create=True, size=arr.nbytes), inp)

        def write(src: NpyArray, sh: SharedMemory) -> None:
            dst = np.ndarray(src.shape, dtype=src.dtype, buffer=sh.buf)
            dst[:] = src

        tree_map(write, inp, shm_inp)
        shm_inp_name = tree_map(lambda sh: sh.name, shm_inp)
        inp_shape = tree_map(lambda arr: wrapped_type(arr.shape), inp)
        inp_dtype = tree_map(lambda arr: arr.dtype, inp)

        res = process_map(
            run_sharded(
                fn,
                inp_length=inp_lengths[0],
                shm_inp_name=shm_inp_name,
                inp_shape=inp_shape,
                inp_dtype=inp_dtype,
                shards=parallel,
            ),
            [*range(parallel)],
        )

        tree_map(lambda sh: sh.close(), shm_inp)
        tree_map(lambda sh: sh.unlink(), shm_inp)

        return np.concat(res)

    return wrapped


class wrapped_type(Generic[T]):
    def __init__(self, data: T) -> None:
        self.data = data


class run_sharded(Generic[U_np]):
    def __init__(
        self,
        fn: Callable[[PyTree[NpyArray]], npt.NDArray[U_np]],
        *,
        inp_length: int,
        shm_inp_name: PyTree[str],
        inp_shape: PyTree[wrapped_type[tuple[int, ...]]],
        inp_dtype: PyTree[np.dtype[Any]],
        shards: int,
    ) -> None:
        self.fn = fn
        self.inp_length = inp_length
        self.shm_inp_name = shm_inp_name
        self.inp_shape = inp_shape
        self.inp_dtype = inp_dtype
        self.shards = shards

    def __call__(self, shard: int) -> npt.NDArray[U_np]:
        start = int(self.inp_length * shard / self.shards)
        stop = self.inp_length if shard == self.shards else int(self.inp_length * (shard + 1) / self.shards)
        shm = tree_map(lambda name: SharedMemory(name=name), self.shm_inp_name)
        inputs = tree_map(
            lambda sh, shape, dtype: np.ndarray(shape.data, dtype, buffer=sh.buf),
            shm,
            self.inp_shape,
            self.inp_dtype,
        )
        input_views = tree_map(lambda arr: arr[start:stop], inputs)
        res = self.fn(input_views)
        tree_map(lambda sh: sh.close(), shm)
        return res
