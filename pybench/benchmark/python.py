# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Pure Python benchmarks."""


# type annotations
from __future__ import annotations
from typing import Tuple, List, Optional

# standard libs
import secrets
from threading import Thread
from queue import Queue

# external libs
import numpy as np

# internal libs
from ..core import Benchmark, BenchmarkError

# public interface
__all__ = ['Iter', 'MapReduce', 'Fibonacci', 'QueueBenchmark', ]


class Iter(Benchmark):
    """Pure iteration."""

    name = 'python.iter'
    annotation = '(size: int)'

    size: int

    def setup(self, size: int) -> None:
        try:
            self.size = int(float(size))  # NOTE: permissive coercion (e.g., 1_000 or 1e9)
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        for _ in range(self.size):
            pass


class MapReduce(Benchmark):
    """Inner product of list of values with filter."""

    name = 'python.mapreduce'
    annotation = '(size: int)'

    size: int
    data: Tuple[List[float], List[float]]

    def setup(self, size: int) -> None:
        try:
            self.size = int(float(size))  # NOTE: permissive coercion (e.g., 1_000 or 1e9)
            self.data = list(np.random.rand(self.size)), list(np.random.rand(self.size))
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        sum(map(lambda pair: pair[0] * pair[1], filter(lambda pair: pair[0] > pair[1], zip(*self.data))))


def _fibonacci_impl(n: int) -> int:
    """Recursive implementation of the Fibonacci sequence."""
    if n <= 1:
        return n
    else:
        return _fibonacci_impl(n - 1) + _fibonacci_impl(n - 2)


class Fibonacci(Benchmark):
    """Recursive implementation of the Fibonacci sequence."""

    name = 'python.fibonacci'
    annotation = '(size: int)'

    size: int

    def setup(self, size: int) -> None:
        try:
            self.size = int(float(size))  # NOTE: permissive coercion (e.g., 1_000 or 1e9)
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        _fibonacci_impl(self.size)


class QueueConsumer(Thread):
    """Get items off the queue."""

    queue: Queue[Optional[bytes]]

    def __init__(self, queue: Queue[Optional[bytes]]) -> None:
        super().__init__()
        self.queue = queue

    def run(self) -> None:
        while _ := self.queue.get():
            pass


class QueueProducer(Thread):
    """Put items on the queue."""

    count: int
    payload: bytes
    queue: Queue[Optional[bytes]]

    def __init__(self, count: int, payload: bytes, queue: Queue[Optional[bytes]]) -> None:
        super().__init__()
        self.count = count
        self.payload = payload
        self.queue = queue

    def run(self) -> None:
        for _ in range(self.count):
            self.queue.put(self.payload)


class QueueBenchmark(Benchmark):
    """Threaded producer/consumer queue throughput."""

    name = 'python.queue'
    annotation = '(size: int, count: int, prods: int, cons: int)'

    size: int
    count: int
    producers: List[Thread]
    consumers: List[Thread]
    queue: Queue[Optional[bytes]]
    payload: bytes

    def setup(self, size: int, count: int, prods: int, cons: int) -> None:
        try:
            self.size = int(float(size))  # NOTE: permissive coercion (e.g., 1_000 or 1e9)
            self.count = int(float(count))
            self.payload = secrets.token_bytes(self.size)
            self.queue = Queue(maxsize=2*int(cons))
            self.consumers = [QueueConsumer(queue=self.queue) for _ in range(cons)]
            self.producers = [QueueProducer(count=self.count, payload=self.payload, queue=self.queue)
                              for _ in range(prods)]
        except Exception as error:
            raise BenchmarkError(f'Args for \'{self.name}\': {error}') from error

    def task(self) -> None:
        for thread in self.producers:
            thread.start()
        for thread in self.consumers:
            thread.start()
        for thread in self.producers:
            thread.join()
        for _ in self.consumers:
            self.queue.put(None)
        for thread in self.consumers:
            thread.join()
