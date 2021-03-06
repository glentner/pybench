# This program is free software: you can redistribute it and/or modify it under the
# terms of the Apache License (v2.0) as published by the Apache Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache License for more details.
#
# You should have received a copy of the Apache License along with this program.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

"""Core functionality for pybench."""


# type annotations
from __future__ import annotations
from typing import List, Callable, Any, TypeVar

# standard libraries
import socket
import logging
from time import sleep
from timeit import default_timer
from threading import Thread
from abc import ABC, abstractmethod, abstractproperty

# external libs
import psutil

# public interface
__all__ = ['Benchmark', 'BenchmarkError', 'CPUResource', 'MemoryResource', 'coerce_type', ]


HOSTNAME = socket.gethostname()
class LogRecord(logging.LogRecord):
    """Extends `logging.LogRecord` to include a hostname."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hostname = HOSTNAME


logging.setLogRecordFactory(LogRecord)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(hostname)s %(name)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class BenchmarkError(Exception):
    """Error in setup or running benchmark."""


class Benchmark(ABC):
    """Boilerplate for running a benchmark."""

    log: Callable[[str], None] = None
    args: List[Any] = None
    repeat: int = None
    spacing: float = None
    annotation: str = '()'

    def __init__(self, repeat: int = 1, spacing: float = 1.0, *args) -> None:
        """Initialize parameters."""
        self.log = logging.getLogger(f'benchmark.{self.name}').info
        self.args = list(args)
        self.repeat = int(repeat)
        self.spacing = float(spacing)
        self.prepare()

    @abstractproperty
    def name(self) -> str:
        """The name of the benchmark."""

    def setup(self, *args) -> None:
        """Initialize state or member data before run."""
        self.args = list(args)

    def prepare(self) -> None:
        """Wraps call to setup."""
        try:
            self.setup(*self.args)
        except Exception as error:
            raise BenchmarkError(f'setup for \'{self.name}\': {error}') from error

    @abstractmethod
    def task(self) -> None:
        """The task to be executed."""

    def run(self) -> None:
        """Run benchmark some number of times."""
        for i in range(1, self.repeat + 1):
            self.prepare()
            self.log(f'[{i}] start')
            time = default_timer()
            self.task()
            elapsed = default_timer() - time
            self.log(f'[{i}] {elapsed}')
            sleep(self.spacing)


class Resource(Thread):
    """Monitor resource and log usage."""

    log: Callable[[str], None] = None
    resolution: float = 1.0

    def __init__(self, resolution: float = resolution) -> None:
        """Initialize parameters."""
        super().__init__(name=self.name, daemon=True)
        self.resolution = resolution
        self.log = logging.getLogger(f'resource.{self.name}').info

    @abstractproperty
    def name(self) -> str:
        """The name of the resource."""

    @abstractmethod
    def gather_telemetry(self) -> List[float]:
        """Return list of data points to log."""

    def run(self) -> None:
        """Log telemetry."""
        for data in iter(self.gather_telemetry, None):
            if len(data) == 1:
                metric, = data
                self.log(str(metric))
            else:
                for i, metric in enumerate(data):
                    self.log(f'[{i}] {metric}')
            sleep(self.resolution)

    @classmethod
    def new(cls, *args, **kwargs) -> Resource:
        """Initialize and start thread."""
        thread = cls(*args, **kwargs)
        thread.start()
        return thread


class CPUResource(Resource):
    """Collect telemetry on CPU usage."""

    name = 'cpu'

    def gather_telemetry(self) -> List[float]:
        return psutil.cpu_percent(interval=self.resolution, percpu=True)


class MemoryResource(Resource):
    """Collect telemetry on Memory usage."""

    name = 'memory'

    def gather_telemetry(self) -> List[float]:
        return [psutil.virtual_memory().percent / 100, ]


T = TypeVar('T', int, float, bool, type(None), str)
def coerce_type(value: str) -> T:
    """Passively coerce `value` to available type if possible."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ('none', 'null'):
        return None
    if value.lower() == 'true':
        return True
    if value.lower() == 'false':
        return False
    else:
        return value
