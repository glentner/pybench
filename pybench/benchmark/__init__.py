# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Library of benchmark definitions."""


# type annotations
from typing import Dict, Type

# internal libs
from ..core import Benchmark
from . import numpy, python, tensorflow

# public interface
__all__ = ['python', 'numpy', 'listing', ]


def find_benchmarks(module) -> Dict[str, Type[Benchmark]]:
    """Enumerate benchmarks in `module`."""
    found = {}
    for name in module.__all__:
        benchmark_type = getattr(module, name)
        found[benchmark_type.name] = benchmark_type
    return found


listing: Dict[str, Type[Benchmark]] = {
    **find_benchmarks(python),
    **find_benchmarks(numpy),
    **find_benchmarks(tensorflow)
}
