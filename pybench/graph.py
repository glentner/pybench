# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Create graphs based on log data."""


# type annotations
from __future__ import annotations

# standard libs
from psutil import virtual_memory

# external libs
import matplotlib.pyplot as plot
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np

# internal libs
from .logdata import LogData

# public interface
__all__ = ['PerfChart', ]


# total memory in gigabytes
TOTAL_MEM = virtual_memory().total / 1024 ** 3
if TOTAL_MEM != int(TOTAL_MEM):
    TOTAL_MEM = int(TOTAL_MEM)


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

    def print_stats(self) -> None:
        """Print timing data (mean and stddev)."""
        print(f'{self.data.mean_time:.2f} ± {self.data.std_time:.2f} (seconds)')

    def save(self, *args, **kwargs) -> None:
        """Save figure to local file system."""
        self.figure.savefig(*args, **kwargs)
