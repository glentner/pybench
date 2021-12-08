# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Specification for LogRecord and LogData structures."""


# type annotations
from __future__ import annotations
from typing import List, Union, IO, Tuple

# standard libs
from collections import defaultdict
from dataclasses import dataclass
from functools import cached_property

# external libs
from pandas import DataFrame, Series

# public interface
__all__ = ['LogRecord', 'LogData', ]


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

    class Error(Exception):
        """Exception specific to LogData."""

    def take(self, n: int) -> LogData:
        """Take first `n` trials of data."""
        records = []
        count = 0
        for record in self.records:
            if record.topic.startswith('benchmark') and record.message.endswith('start'):
                count += 1
                if count > n:
                    return LogData(records)
            records.append(record)
        else:
            raise self.Error(f'Only found {count} trials in data')
