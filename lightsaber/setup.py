#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from lightsaber import __version__
import os
try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

with open('./README.md') as f:
    readme = f.read()

# with open('./LICENSE') as f:
#     license = f.read()

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

#  _t2e_deps = ["pysurvival==0.1.2"]
_doc_deps = ["mkdocs==1.1.2", 
             "mkdocs-material==7.0.6",
             "mkdocstrings==0.15.0",
             "mknotebooks==0.7.0"
             #  "mkdocs-pdf-export-plugin==0.5.8"
            ]
_full_deps = _doc_deps  # + _t2e_deps 


setup(
    name='lightsaber',
    version=__version__,
    description='Package to handle model training for dpm tasks',
    long_description=readme,
    author='Prithwish Chakraborty',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=reqs,
    extras_require={
                    #  "t2e": _t2e_deps,
                    #  "doc": _doc_deps,
                    "full": _full_deps
                   }
)
