# Shared memory parallel map for Numpy

Distribute trivially parallelisable computations, typically element-wise operations on large Numpy arrays.

```py
from functools import partial

import numpy as np
from shmem_pmap import shmem_pmap

def do_work(data):
    return data["arr1"] + data["arr2"]

data = {
    "arr1": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    "arr2": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
}

shmem_pmap(do_work, parallel=4)(data)
# Output: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

shmem_pmap(do_work, parallel=4)(data, rv_shape=(10,), rv_dtype=np.int64)
```
