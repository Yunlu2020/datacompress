from __future__ import print_function
import os
import sys
import re
import os.path as op
from setuptools import find_packages, setup

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]
    
    

import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="datacompress",
    py_modules=["dateaset", "train", "utils"],
    version="0.1",
    description="",
    author="HCP",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
    # extras_require={'dev': ['pytest']},
)