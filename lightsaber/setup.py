#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Copyright 2020 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from lightsaber import __version__
import os
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

try:
    with open('./README.md') as f:
        readme = f.read()
except FileNotFoundError:
    readme = ""

try:
    with open('./LICENSE') as f:
        license = f.read()
except FileNotFoundError:
    license = ""

install_reqs = parse_requirements(os.path.join(os.path.dirname(__file__), './requirements.txt'),
                                  session='hack')

# To handle all versions of pip
# source: https://stackoverflow.com/a/62127548
# Generator must be converted to list, or we will only have one chance to read each element, meaning that the first requirement will be skipped.
install_reqs = list(install_reqs) 
try:
    reqs = [str(ir.req) for ir in install_reqs]
except AttributeError:
    reqs = [str(ir.requirement) for ir in install_reqs]
#  reqs = [str(ir.req) for ir in install_reqs]

_t2e_deps = ["pysurvival==0.1.2"]
_doc_deps = ["mkdocs==1.2.2", 
             "mkdocs-material==7.2.6",
             "mkdocstrings==0.15.2",
             "mknotebooks==0.7.0",
			 "mkdocs-monorepo-plugin==0.4.16",
             "pytkdocs[numpy-style]==0.11.1",
             #  "mkdocs-pdf-export-plugin==0.5.8"
            ]

_full_deps = _t2e_deps + _doc_deps


setup(
    name='dpm360-lightsaber',
    version=__version__,
    description='Package to handle model training for dpm tasks',
    long_description=readme,
    long_description_content_type="text/markdown",
    author='Prithwish Chakraborty',
    author_email='prithwish.chakraborty@ibm.com',
    url='https://ibm.github.io/DPM360/Lightsaber/',	
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=reqs,
    extras_require={
                    "t2e": _t2e_deps,
                    "doc": _doc_deps,
                    "full": _full_deps
                   }
)
