# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Package initialization and entry-point for console application."""


# type annotations
from __future__ import annotations
from typing import List, Dict, Optional, Callable, Type, Any

# standard libs
import re
import sys
import time
import logging
import functools
from platform import python_version, python_implementation

# external libs
from cmdkit.app import Application, ApplicationGroup, exit_status
from cmdkit.cli import Interface
import numpy as np

# internal libs
from .__meta__ import (__appname__, __version__, __authors__, __description__,
                       __contact__, __license__, __copyright__, __keywords__, __website__)
from .core import CPUResource, MemoryResource, Benchmark, BenchmarkError, coerce_type
from .graph import PerfChart
from .logdata import LogRecord, LogData
from . import benchmark

# public interface
__all__ = ['PyBenchApp', 'main', ]


# inject logger setup into command-line framework
log = logging.getLogger(__name__)
Application.log_critical = log.critical
Application.log_exception = log.exception


def log_exception(exc: Exception, status: int) -> int:
    """Print exception and return exit status."""
    log.critical(f'error: {exc}')
    return status


list_desc = "List available benchmarks."
list_usage = f"""\
usage: pybench list [-h] [-l] [PATTERN]
{list_desc}\
"""
list_help = f"""\
{list_usage}

arguments:
PATTERN                  Regular expression filter.

options:
-l, --long               Show details on benchmark.
-h, --help               Show this message and exit.\
"""


# ansi color codes
ANSI_RESET = '\033[0m'
ANSI_BLUE = '\033[34m'


class ListApp(Application):
    """List available benchmarks."""

    interface = Interface('pybench list', list_usage, list_help)
    ALLOW_NOARGS = True

    pattern: re.Pattern = re.compile(f'.*')
    interface.add_argument('pattern', nargs='?', type=re.compile, default=pattern)

    long_mode: bool = False
    interface.add_argument('-l', '--long', action='store_true', dest='long_mode')

    def run(self) -> None:
        """List benchmarks."""
        for name, benchmark_type in benchmark.listing.items():
            if self.pattern.match(name):
                self.output(name, benchmark_type)

    @property
    def output(self) -> Callable[[str, Type[Benchmark]], None]:
        return self.detailed_output if self.long_mode else self.basic_output

    @staticmethod
    def basic_output(name: str, benchmark_type: Type[Benchmark]) -> None:
        name = f'{name:<17}' if not sys.stdout.isatty() else f'{ANSI_BLUE}{name:<18}{ANSI_RESET}'
        print(f'{name}   {benchmark_type.__doc__}')

    @staticmethod
    def detailed_output(name: str, benchmark_type: Type[Benchmark]) -> None:
        name = f'{name:<18}' if not sys.stdout.isatty() else f'{ANSI_BLUE}{name:<18}{ANSI_RESET}'
        print(f'{name} {benchmark_type.annotation:>27}   {benchmark_type.__doc__}')


run_desc = "Run benchmark."
run_usage = f"""\
usage: pybench run [-h] NAME [ARGS...] [-n COUNT] [-s SEC] [-cm] [-r SEC]
{run_desc}\
"""
run_help = f"""\
{run_usage}

arguments:
NAME                     Name of benchmark to run.
ARGS...                  Positional arguments passed to benchmark.

options:
-n, --repeat      COUNT  Number of times to run benchmark. (default: 1)
-s, --spacing     SEC    Time (seconds) between runs. (default: 1)
-c, --monitor-cpu        Collect telemetry on CPU usage.
-m, --monitor-memory     Collect telemetry on memory usage.
-r, --resolution  SEC    Time (seconds) between samples. (default: 1)
-h, --help               Show this message and exit.\
"""


class RunApp(Application):
    """Run benchmark."""

    interface = Interface('pybench run', run_usage, run_help)

    name: str = None
    interface.add_argument('name', choices=list(benchmark.listing))

    args: List[Any] = None
    interface.add_argument('args', nargs='*', type=coerce_type, default=[])

    repeat: int = 1
    interface.add_argument('-n', '--repeat', type=int, default=repeat)

    spacing: float = 1.0
    interface.add_argument('-s', '--spacing', type=float, default=spacing)

    monitor_cpu: bool = False
    interface.add_argument('-c', '--monitor-cpu', action='store_true')

    monitor_memory: bool = False
    interface.add_argument('-m', '--monitor-memory', action='store_true')

    resolution: float = 1.0
    interface.add_argument('-r', '--resolution', type=float, default=resolution)

    cpu_thread: Optional[CPUResource] = None
    mem_thread: Optional[MemoryResource] = None

    exceptions = {
        BenchmarkError: functools.partial(log_exception, status=exit_status.runtime_error)
    }

    def run(self) -> None:
        """Run requested benchmark."""
        self.setup_telemetry()
        self.run_benchmark()
        if self.cpu_thread or self.mem_thread:
            time.sleep(self.spacing)  # NOTE: pause to get one more metric

    def setup_telemetry(self) -> None:
        """Start telemetry threads if requested."""
        self.cpu_thread = None if not self.monitor_cpu else CPUResource.new(self.resolution)
        self.mem_thread = None if not self.monitor_memory else MemoryResource.new(self.resolution)

    def run_benchmark(self) -> None:
        """Setup and initiate benchmark."""
        benchmark_type = benchmark.listing.get(self.name)
        benchmark_type(self.repeat, self.spacing, *self.args).run()


graph_desc = "Graph benchmark log data."
graph_usage = f"""\
usage: pybench graph [-h] FILE [-o FILE] 
                     [--label-benchmark TEXT] [--label-build TEXT] [--label-version TEXT]
{graph_desc}\
"""
graph_help = f"""\
{graph_usage}

options:
-o, --output           FILE   Path to save output file.
    --label-benchmark  TEXT   Label text for benchmark name.
    --label-build      TEXT   Label text for build info.
    --label-version    TEXT   Label text for version info.
-h, --help                    Show this message and exit.\
"""


class GraphApp(Application):
    """Graph benchmark data."""

    interface = Interface('pybench graph', graph_usage, graph_help)

    source: str = '-'
    interface.add_argument('source')

    outpath: Optional[str] = None
    interface.add_argument('-o', '--output', dest='outpath', default=outpath)

    label_benchmark: Optional[str] = None
    interface.add_argument('--label-benchmark', default=label_benchmark)

    label_build: Optional[str] = None
    interface.add_argument('--label-build', default=label_build)

    label_version: Optional[str] = None
    interface.add_argument('--label-version', default=label_version)

    data: LogData
    graph: PerfChart

    def run(self) -> None:
        """List benchmarks."""
        self.load_data()
        self.graph = PerfChart(self.data, **self.get_labels())
        self.graph.draw()
        self.graph.save(self.outpath)

    def load_data(self) -> None:
        """Load data from source."""
        if self.source == '-':
            self.data = LogData.from_io(sys.stdin)
        else:
            self.data = LogData.from_local(self.source)

    def get_labels(self) -> Dict[str, str]:
        """Derive labels from data or from explicit assignment."""
        return {'label_benchmark': self.label_benchmark or self.data.benchmark_name,
                'label_build': self.label_build or self.get_python_info(),
                'label_version': self.label_version or self.get_numpy_info()}

    @staticmethod
    def get_python_info() -> str:
        """Load Python build information."""
        implementation = python_implementation()
        version = python_version()
        return f'{implementation} {version}'

    @staticmethod
    def get_numpy_info() -> str:
        """Load NumPy build information."""
        return f'NumPy {np.__version__}'


app_usage = f"""\
usage: pybench [-h] [-v] <command> ...
{__description__}\
"""
app_help = f"""\
{app_usage}

commands:
run                      {run_desc}
list                     {list_desc}
graph                    {graph_desc}

options:
-h, --help               Show this message and exit.
-v, --version            Show the version and exit.

Documentation and issue tracking at:
{__website__}\
"""


class PyBenchApp(ApplicationGroup):
    """Run benchmark."""

    interface = Interface('pybench', app_usage, app_help)
    interface.add_argument('-v', '--version', action='version', version=__version__)
    interface.add_argument('command')

    command = None
    commands = {
        'run': RunApp,
        'list': ListApp,
        'graph': GraphApp,
    }


def main() -> int:
    """Entry-point for console application."""
    return PyBenchApp.main(sys.argv[1:])
