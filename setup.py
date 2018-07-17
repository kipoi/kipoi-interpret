#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


requirements = [
    "kipoi[vep]",
    "tqdm",
    "matplotlib",
    "seaborn",
    # sometimes required
    "h5py",
    "deeplift",
]

test_requirements = [
    "bumpversion",
    "wheel",
    "jedi",
    "epc",
    "pytest>=3.3.1",
    "pytest-xdist",  # running tests in parallel
    "pytest-pep8",  # see https://github.com/kipoi/kipoi/issues/91
    "pytest-cov",
    "coveralls",
    "scikit-learn",
    "cython",
    # "genomelake",
    "keras",
    "tensorflow"
]

setup(
    name='kipoi_interpret',
    version='0.1.0',
    description="Kipoi interpret: interepretation plugin for Kipoi",
    author="Kipoi team",
    author_email='avsec@in.tum.de',
    url='https://github.com/kipoi/kipoi-interpret',
    long_description="Kipoi interpret: interepretation plugin for Kipoi",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        "develop": test_requirements,
    },
    entry_points={'console_scripts': ['kipoi_interpret = kipoi.cli:cli_main']},
    license="MIT license",
    zip_safe=False,
    keywords=["model zoo", "deep learning",
              "computational biology", "bioinformatics", "genomics"],
    test_suite='tests',
    include_package_data=True,
    tests_require=test_requirements
)
