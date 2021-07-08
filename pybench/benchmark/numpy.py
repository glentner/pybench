# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""NumPy benchmarks."""


# type annotations
from __future__ import annotations
from typing import List, Tuple, Callable

# external libs
import numpy as np

# internal libs
from ..core import Benchmark, BenchmarkError

# public interface
__all__ = ['Random', 'MatMul', 'DotProduct', 'MatInv', ]


def create_float64(shape: List[int]) -> np.ndarray:
    """Generate `ndarray` of random data of type `np.float64`."""
    return np.random.rand(*shape)


def create_float32(shape: List[int]) -> np.ndarray:
    """Generate `ndarray` of random data of type `np.float32`."""
    return create_float64(shape).astype(np.float32)


def create_complex64(shape: List[int]) -> np.ndarray:
    """Generate `ndarray` of random data of type `np.complex64`."""
    array = create_float64(shape).astype(np.complex64)
    array.imag = create_float64(shape)
    return array


INT32_SCALAR: int = 100_000_000
INT64_SCALAR: int = 1_000_000_000


def create_int32(shape: List[int]) -> np.ndarray:
    """Generate `ndarray` of random data of type `np.int32`."""
    return (create_float64(shape) * INT32_SCALAR).astype(np.int32)


def create_int64(shape: List[int]) -> np.ndarray:
    """Generate `ndarray` of random data of type `np.int64`."""
    return (create_float64(shape) * INT64_SCALAR).astype(np.int64)


FACTORY = {
    'float32': create_float32,
    'float64': create_float64,
    'complex64': create_complex64,
    'int32': create_int32,
    'int64': create_int64,
}


def get_factory(name: str) -> Callable[[List[int]], np.ndarray]:
    """Load array factory by `name`."""
    try:
        return FACTORY.get(name)
    except KeyError as error:
        raise TypeError(f'Unsupported type \'{name}\'') from error


class Random(Benchmark):
    """Generate random values, any shape."""

    name = 'numpy.random.rand'
    annotation = '(*shape: int)'

    shape: List[int]
    factory: Callable[[List[int]], np.ndarray]

    def setup(self, *shape: int) -> None:
        try:
            self.shape = list(map(int, shape))
            self.factory = get_factory('float64')
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        self.factory(self.shape)


class MatMul(Benchmark):
    """Matrix multiplication, 2D."""

    name = 'numpy.matmul'
    annotation = '(dtype: str, *shape: int)'

    dtype: str
    shape: List[int]
    arrays: Tuple[np.ndarray, np.ndarray]
    factory: Callable[[List[int]], np.ndarray]

    def setup(self, dtype: str, *shape: int) -> None:
        try:
            self.dtype = dtype
            self.shape = list(map(int, shape))
            self.factory = get_factory(self.dtype)
            if len(shape) == 2:
                self.arrays = None  # noqa: allow de-allocation
                self.arrays = self.factory(self.shape), self.factory(self.shape)
            else:
                raise BenchmarkError(f'Expected 2D for \'{self.name}\', given {shape}')
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.matmul(*self.arrays)


class DotProduct(Benchmark):
    """Compute inner product, 1D or 2D."""

    name = 'numpy.dot'
    annotation = '(dtype: str, *shape: int)'

    dtype: str
    shape: List[int]
    arrays: Tuple[np.ndarray, np.ndarray]
    factory: Callable[[List[int]], np.ndarray]

    def setup(self, dtype: str, *shape: int) -> None:
        try:
            self.dtype = dtype
            self.shape = list(map(int, shape))
            self.factory = get_factory(self.dtype)
            if len(shape) in (1, 2):
                self.arrays = None  # noqa: allow de-allocation
                self.arrays = self.factory(self.shape), self.factory(self.shape)
            else:
                raise BenchmarkError(f'Expected 1D or 2D for \'{self.name}\', given {len(shape)}{shape}')
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.dot(*self.arrays)


class MatInv(Benchmark):
    """Invert 2D matrix."""

    name = 'numpy.linalg.inv'
    annotation = '(dtype: str, *shape: int)'

    dtype: str
    shape: List[int]
    array: np.ndarray
    factory: Callable[[List[int]], np.ndarray]

    def setup(self, dtype: str, *shape: int) -> None:
        try:
            self.dtype = dtype
            self.shape = list(map(int, shape))
            self.factory = get_factory(self.dtype)
            if len(shape) == 2:
                self.array = None  # noqa: allow de-allocation
                self.array = self.factory(self.shape)
            else:
                raise BenchmarkError(f'Expected 2D for \'{self.name}\', given {len(shape)}{shape}')
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        np.linalg.inv(self.array)
