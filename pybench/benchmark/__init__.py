# This program is free software: you can redistribute it and/or modify it under the
# terms of the Apache License (v2.0) as published by the Apache Software Foundation.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the Apache License for more details.
#
# You should have received a copy of the Apache License along with this program.
# If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

"""Library of benchmark definitions."""


# type annotations
from typing import Dict, Type

# internal libs
from ..core import Benchmark
from . import numpy

# public interface
__all__ = ['numpy', 'listing', ]


def find_benchmarks(module) -> Dict[str, Type[Benchmark]]:
    """Enumerate benchmarks in `module`."""
    found = {}
    for name in module.__all__:
        benchmark_type = getattr(module, name)
        found[benchmark_type.name] = benchmark_type
    return found


listing: Dict[str, Type[Benchmark]] = {
    **find_benchmarks(numpy),
}
