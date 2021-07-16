# SPDX-FileCopyrightText: 2021 Geoffrey Lentner
# SPDX-License-Identifier: Apache-2.0

"""Build and installation script for PyBench."""


# standard libs
import re
from setuptools import setup, find_packages


# get long description from README.rst
with open('README.rst', mode='r') as readme:
    long_description = readme.read()


# get package metadata by parsing __meta__ module
with open('pybench/__meta__.py', mode='r') as source:
    content = source.read().strip()
    metadata = {key: re.search(key + r'\s*=\s*[\'"]([^\'"]*)[\'"]', content).group(1)
                for key in ['__version__', '__developer__', '__contact__', '__description__',
                            '__license__', '__keywords__', '__website__']}


# core dependencies
DEPS = ['cmdkit', 'psutil', 'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', ]


setup(
    name             = 'pybench',
    version          = metadata['__version__'],
    author           = metadata['__developer__'],
    author_email     = metadata['__contact__'],
    description      = metadata['__description__'],
    license          = metadata['__license__'],
    keywords         = metadata['__keywords__'],
    url              = metadata['__website__'],
    packages         = find_packages(),
    include_package_data = True,
    long_description = long_description,
    long_description_content_type = 'text/x-rst',
    classifiers      = ['Development Status :: 4 - Beta',
                        'Topic :: Utilities',
                        'Programming Language :: Python :: 3.8',
                        'Programming Language :: Python :: 3.9',
                        'Programming Language :: Python :: 3.10',
                        'Operating System :: POSIX :: Linux',
                        'Operating System :: MacOS',
                        'Operating System :: Microsoft :: Windows',
                        'License :: OSI Approved :: Apache Software License', ],
    install_requires = DEPS,
    entry_points     = {'console_scripts': ['pybench=pybench:main', ]},
)
