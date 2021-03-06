# This program is free software: you can redistribute it and/or modify it under the
# terms of the Apache License (v2.0) as published by the Apache Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache License for more details.
#
# You should have received a copy of the Apache License along with this program.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

"""Package initialization and entry-point for console application."""


# type annotations
from __future__ import annotations
from typing import List, Optional, Callable, Type

# standard libs
import sys
import time
import logging
import functools

# external libs
from cmdkit.app import Application, ApplicationGroup, exit_status
from cmdkit.cli import Interface

# internal libs
from .__meta__ import (__appname__, __version__, __authors__, __description__,
                       __contact__, __license__, __copyright__, __keywords__, __website__)
from .core import CPUResource, MemoryResource, Benchmark, BenchmarkError
from . import benchmark

# public interface
__all__ = ['NPBenchApp', 'main', ]


# inject logger setup into command-line framework
log = logging.getLogger(__name__)
Application.log_critical = log.critical
Application.log_exception = log.exception


benchmark_map = {
    'random': benchmark.Random,
    'matmul': benchmark.MatMul,
}


benchmark_output = '\n'.join([f'{key}: {value.__doc__}'
                              for key, value in benchmark_map.items()])


def log_exception(exc: Exception, status: int) -> int:
    """Print exception and return exit status."""
    log.critical(f'error: {exc}')
    return status


list_desc = "List benchmarks."
list_usage = f"""\
usage: pybench list [-h] [-l]
{list_desc}\
"""
list_help = f"""\
{list_usage}

options:
-l, --long               Show details on benchmark.
-h, --help               Show this message and exit.\
"""


# ansi color codes
ANSI_RESET = '\033[0m'
ANSI_BLUE = '\033[34m'


class ListApp(Application):
    """List benchmarks."""

    interface = Interface('pybench list', list_usage, list_help)
    ALLOW_NOARGS = True

    long_mode: bool = False
    interface.add_argument('-l', '--long', action='store_true', dest='long_mode')

    def run(self) -> None:
        """List benchmarks."""
        for name, benchmark_type in benchmark_map.items():
            self.output(name, benchmark_type)

    @property
    def output(self) -> Callable[[str, Type[Benchmark]], None]:
        return self.detailed_output if self.long_mode else self.basic_output

    @staticmethod
    def basic_output(name: str, benchmark_type: Type[Benchmark]) -> None:
        name = f'{name:<10}' if not sys.stdout.isatty() else f'{ANSI_BLUE}{name:<10}{ANSI_RESET}'
        print(f'{name} {benchmark_type.__doc__}')

    @staticmethod
    def detailed_output(name: str, benchmark_type: Type[Benchmark]) -> None:
        name = name if not sys.stdout.isatty() else f'{ANSI_BLUE}{name}{ANSI_RESET}'
        print(f'{name} {benchmark_type.annotation:<16} {benchmark_type.__doc__}')


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
-n, --iterations  COUNT  Number of times to run benchmark. (default: 1)
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
    interface.add_argument('name', choices=list(benchmark_map))

    args: List[str] = None
    interface.add_argument('args', nargs='*')

    n_iterations: int = 1
    interface.add_argument('-n', '--iterations', type=int, default=n_iterations, dest='n_iterations')

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
        benchmark_type = benchmark_map[self.name]
        benchmark_type(self.n_iterations, self.spacing, *self.args).run()


graph_desc = "Graph benchmark data."
graph_usage = f"""\
usage: pybench graph [-h] FILE [--output FILE]
{graph_desc}\
"""
graph_help = f"""\
{graph_usage}

options:
-o, --output  FILE       Path to save output file.
-h, --help               Show this message and exit.\
"""


class GraphApp(Application):
    """Graph benchmark data."""

    interface = Interface('pybench graph', graph_usage, graph_help)

    outpath: str = None
    interface.add_argument('-o', '--output', dest='outpath')

    def run(self) -> None:
        """List benchmarks."""
        log.error('Not implemented.')


app_usage = f"""\
usage: pybench [-h] [-v] <command> ...
{__description__}\
"""
app_help = f"""\
{app_usage}

commands:
list                     {list_desc}
run                      {run_desc}
graph                    {graph_desc}

options:
-h, --help               Show this message and exit.
-v, --version            Show the version and exit.

Documentation and issue tracking at:
{__website__}\
"""


class NPBenchApp(ApplicationGroup):
    """Run benchmark."""

    interface = Interface('pybench', app_usage, app_help)
    interface.add_argument('-v', '--version', action='version', version=__version__)
    interface.add_argument('command')

    command = None
    commands = {
        'list': ListApp,
        'run': RunApp,
        'graph': GraphApp,
    }


def main() -> int:
    """Entry-point for console application."""
    return NPBenchApp.main(sys.argv[1:])
