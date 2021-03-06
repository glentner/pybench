# This program is free software: you can redistribute it and/or modify it under the
# terms of the Apache License (v2.0) as published by the Apache Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache License for more details.
#
# You should have received a copy of the Apache License along with this program.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

"""NumPy benchmarks."""


# type annotations
from __future__ import annotations
from typing import List, Tuple

# external libs
import numpy as np

# internal libs
from ..core import Benchmark, BenchmarkError

# public interface
__all__ = ['Random', 'MatMul', 'DotProduct', ]


class Random(Benchmark):
    """Generate random values, any shape."""

    name = 'numpy.random.rand'
    annotation = '(*shape: int)'

    shape: List[int] = None

    def setup(self, *shape: int) -> None:
        try:
            self.shape = list(map(int, shape))
        except Exception as error:
            raise BenchmarkError(f'args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.random.rand(*self.shape)


class MatMul(Benchmark):
    """Matrix multiplication, 2D."""

    name = 'numpy.matmul'
    annotation = '(dtype: str, *shape: int)'

    dtype: str = None
    shape: List[int] = None
    arrays: Tuple[np.ndarray, np.ndarray] = None

    def setup(self, dtype: str, *shape: int) -> None:
        try:
            self.dtype = dtype
            self.shape = list(map(int, shape))
            if len(shape) == 2:
                self.arrays = None  # noqa: allow de-allocation
                self.arrays = np.random.rand(*self.shape).astype(dtype), np.random.rand(*self.shape).astype(dtype)
            else:
                raise BenchmarkError(f'expected 2D for \'{self.name}\', given {shape}')
        except Exception as error:
            raise BenchmarkError(f'args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.matmul(*self.arrays)


class DotProduct(Benchmark):
    """Compute dot product, 1D or 2D."""

    name = 'numpy.dot'
    annotation = '(dtype: str, *shape: int)'

    dtype: str = None
    shape: List[int] = None
    arrays: Tuple[np.ndarray, np.ndarray] = None

    def setup(self, dtype: str, *shape: int) -> None:
        try:
            self.dtype = dtype
            self.shape = list(map(int, shape))
            if len(shape) in (1, 2):
                self.arrays = None  # noqa: allow de-allocation
                self.arrays = np.random.rand(*self.shape).astype(dtype), np.random.rand(*self.shape).astype(dtype)
            else:
                raise BenchmarkError(f'expected 1D or 2D for \'{self.name}\', given {len(shape)}{shape}')
        except Exception as error:
            raise BenchmarkError(f'args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.dot(*self.arrays)
