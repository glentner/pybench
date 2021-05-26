# This program is free software: you can redistribute it and/or modify it under the
# terms of the Apache License (v2.0) as published by the Apache Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache License for more details.
#
# You should have received a copy of the Apache License along with this program.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

"""Create graphs based on log data."""


# type annotations
from __future__ import annotations
from typing import List, Tuple, Union, IO

# standard libs
from dataclasses import dataclass
from functools import cached_property
from collections import defaultdict
from psutil import virtual_memory

# external libs
import matplotlib.pyplot as plot
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pandas import DataFrame, Series
import numpy as np

# public interface
__all__ = ['LogRecord', 'LogData', 'PerfChart', ]


# total memory in gigabytes
TOTAL_MEM = virtual_memory().total / 1024 ** 3
if TOTAL_MEM != int(TOTAL_MEM):
    TOTAL_MEM = int(TOTAL_MEM)


@dataclass
class LogRecord:
    """Semi-structured tuple based on log records."""

    date: str
    time: str
    host: str
    topic: str
    message: str

    @classmethod
    def from_line(cls, text: str) -> LogRecord:
        """Parse single line of `text`."""
        date, time, host, topic, *msg_args = text.strip().split()
        return cls(date, time, host, topic, ' '.join(msg_args))


class LogData:
    """Representation of log data emitted by pybench runs."""

    __records: List[LogRecord]

    def __init__(self, records: Union[LogData, List[LogRecord]]) -> None:
        """Initialize directly with `data`."""
        self.records = records if not isinstance(records, LogData) else records.records

    @property
    def records(self) -> List[LogRecord]:
        """Access to raw underlying log records."""
        return self.__records

    @records.setter
    def records(self, other: List[LogRecord]) -> None:
        """Assign underlying log records."""
        __records = list(other)
        if all(isinstance(record, LogRecord) for record in __records):
            self.__records = __records
        else:
            raise AttributeError(f'LogData.records expects List[LogRecord]')

    @classmethod
    def from_io(cls, stream: IO) -> LogData:
        """Initialize from IO stream."""
        return cls(list(map(LogRecord.from_line, stream)))

    @classmethod
    def from_local(cls, filepath: str) -> LogData:
        """Initialize from local `filepath`."""
        with open(filepath, mode='r') as stream:
            return cls.from_io(stream)

    @cached_property
    def raw_data(self) -> DataFrame:
        """Representation of raw underlying log records as a DataFrame."""
        return DataFrame(self.records)

    @cached_property
    def datetime(self) -> Series:
        """Constructed datetime series from text based date and time stamps."""
        return (self.raw_data.date + ' ' + self.raw_data.time).astype('datetime64[ns]')

    @cached_property
    def data(self) -> DataFrame:
        """Representation of log records as partially processed DataFrame."""
        data = self.raw_data.assign(datetime=self.datetime).sort_values(by='datetime')
        data = data.assign(elapsed=(data.datetime - data.datetime.min()).dt.total_seconds())
        data = data.set_index('elapsed')
        return data

    @cached_property
    def hostname(self) -> str:
        """Singular hostname found in the underlying log records."""
        hostname, = self.data.host.unique()
        return hostname

    def split_by_host(self) -> Tuple[LogData, ...]:
        """Return collection of instances broken out by unique hostname."""
        record_map = defaultdict(lambda: [])  # Type: Dict[str, List[LogRecord]]
        for record in self.records:
            record_map[record.host].append(record)
        return tuple([LogData(records) for records in record_map.values()])

    @cached_property
    def mem_data(self) -> DataFrame:
        """Parsed memory usage data frame."""
        mem_data = self.data.loc[self.data.topic == 'resource.memory', ['message', ]]
        mem_data = mem_data.assign(value=mem_data.message.str.strip().astype('float32'))
        mem_data = mem_data.drop(['message', ], axis=1)
        return mem_data

    @cached_property
    def cpu_data(self) -> DataFrame:
        """Parsed cpu usage data frame."""
        cpu_data = self.data.loc[self.data.topic == 'resource.cpu', ['message', ]]
        cpu_data = cpu_data[[]].assign(core_id=cpu_data.message.str.split().str[0].str.strip('[]').astype('int16'),
                                       value=cpu_data.message.str.split().str[1].str.strip().astype('float32'))
        return cpu_data

    @cached_property
    def num_cores(self) -> int:
        """Number of CPU cores found in log data."""
        num_cores, = self.cpu_data.core_id.unique().shape
        return num_cores

    @cached_property
    def benchmark_name(self) -> str:
        """Unique benchmark name (topic) found in log data."""
        topic, = self.data.loc[self.data.topic.str.startswith('benchmark.'), 'topic'].unique()
        return topic[len('benchmark.'):]

    @cached_property
    def benchmark_data(self) -> DataFrame:
        """Parsed benchmark timing data."""
        bench_data = self.data.loc[self.data.topic == f'benchmark.{self.benchmark_name}', ['message', ]]
        message_partial = bench_data.message.str.split()
        bench_data = bench_data[[]].assign(run_id=message_partial.str[0].str.strip('[]').astype('int16'),
                                           value=message_partial.str[1].str.strip())
        return bench_data

    @cached_property
    def time_values(self) -> Series:
        """Parsed run times for benchmark iterations."""
        return self.benchmark_data.loc[self.benchmark_data.value != 'start', 'value'].astype('float32')

    @cached_property
    def mean_time(self) -> float:
        """Mean time duration of each benchmark run."""
        return self.time_values.mean()

    @cached_property
    def std_time(self) -> float:
        """Standard deviation of time duration for benchmark runs."""
        return self.time_values.std()


class PerfChart:
    """Create, render, and save performance charts."""

    data: LogData
    figure: Figure
    ax: Axes

    title: str
    label_build: str
    label_version: str
    label_benchmark: str

    def __init__(self, data: LogData, label_benchmark: str, label_build: str, label_version: str) -> None:
        """Initialize graph."""
        self.data = data
        self.label_build = label_build
        self.label_version = label_version
        self.label_benchmark = label_benchmark
        self.title = f'Performance Test: {label_benchmark}'
        self.figure = plot.figure(self.title, figsize=(10, 6))
        self.figure.set_facecolor('white')
        self.ax = self.figure.add_axes([0.08, 0.25, 0.86, 0.65])

    def draw(self) -> None:
        """Render chart."""

        cpu_format = dict(color='steelblue', lw=0.50, alpha=0.5, zorder=40)
        for core, group in self.data.cpu_data.groupby('core_id'):
            group.plot(y='value', ax=self.ax, legend=False, **cpu_format)

        mem_format = dict(color='firebrick', lw=1.0, alpha=1, zorder=50)
        self.data.mem_data.plot(y='value', ax=self.ax, legend=False, **mem_format)

        # plot benchmark run sequences
        fill_format = dict(color='slategray', alpha=0.50, zorder=100)
        for run_id, group in self.data.benchmark_data.groupby('run_id'):
            start, stop = group.index
            plot.fill_between([start, stop], 2 * [0, ], 2 * [1, ], **fill_format)

        # chart formatting
        self.ax.minorticks_on()
        for side in 'top', 'bottom', 'left', 'right':
            self.ax.spines[side].set_color('gray')
            self.ax.spines[side].set_alpha(0.50)

        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(0, self.data.data.index.max())
        self.ax.grid(True, axis='y', which='major', color='gray', lw=1, alpha=0.25, zorder=10)
        self.ax.grid(True, axis='y', which='minor', color='gray', lw=0.5, alpha=0.25, zorder=10)
        self.ax.grid(True, axis='x', which='major', color='gray', lw=1, alpha=0.25, zorder=10)
        self.ax.grid(True, axis='x', which='minor', color='gray', lw=0.5, alpha=0.25, zorder=10)
        self.ax.tick_params(axis='both', which='both', direction='in', length=0)
        self.ax.set_xlabel('Elapsed (seconds)', x=1, ha='right', fontsize=10, labelpad=10, fontweight='semibold')
        self.ax.set_ylabel('Resources (percent)', y=1, ha='right', fontsize=10, labelpad=10, fontweight='semibold')
        self.ax.set_title(self.title, fontsize=14, x=0, ha='left', va='bottom', fontweight='semibold')

        self.figure.text(0.08, 0.16, f'Benchmark:', fontsize=10, fontweight='semibold')
        self.figure.text(0.20, 0.16, f'{self.label_benchmark}', fontsize=10)

        self.figure.text(0.08, 0.12, f'Build:', fontsize=10, fontweight='semibold')
        self.figure.text(0.20, 0.12, f'{self.label_build}', fontsize=10)

        self.figure.text(0.08, 0.08, f'Version:', fontsize=10, fontweight='semibold')
        self.figure.text(0.20, 0.08, f'{self.label_version}', fontsize=10)

        self.figure.text(0.08, 0.04, f'Host:', fontsize=10, fontweight='semibold')
        self.figure.text(0.20, 0.04, f'{self.data.hostname}', fontsize=10)

        self.figure.text(0.48, 0.16, f'Time:', fontsize=10, fontweight='semibold')
        self.figure.text(0.53, 0.16, f'{self.data.mean_time:.2f} ± {self.data.std_time:.2f}  (seconds)', fontsize=10)

        legend = self.figure.add_axes([0.485, 0.02, 0.46, 0.12])
        legend.set_xlim(0, 1)
        legend.set_ylim(0, 1)
        legend.set_xticks([])
        legend.set_yticks([])
        for side in 'top', 'bottom', 'left', 'right':
            legend.spines[side].set_visible(False)

        cpu_label_x = np.linspace(0.00, 0.05, 20)
        cpu_label_y = 0.60 + 0.10 * np.random.rand(20)
        legend.plot(cpu_label_x, cpu_label_y, **cpu_format)
        legend.text(0.08, 0.60, f' CPU ({self.data.num_cores})', fontsize=10, fontweight='semibold')

        mem_label_x = np.linspace(0.00, 0.05, 20)
        mem_label_y = 0.28 + 0.10 * np.random.rand(20)
        legend.plot(mem_label_x, mem_label_y, **mem_format)
        legend.text(0.08, 0.28, f' Memory ({TOTAL_MEM} GB)', fontsize=10, fontweight='semibold')

        legend.fill_between([0.45, 0.50], [0.56, 0.56], [0.80, 0.80], **fill_format)
        legend.text(0.52, 0.61, ' Run Period', fontsize=10, fontweight='semibold')

    def save(self, *args, **kwargs) -> None:
        """Save figure to local file system."""
        self.figure.savefig(*args, **kwargs)
