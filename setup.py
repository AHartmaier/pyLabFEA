#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md") as readme_file:
    readme = readme_file.read()

setup(
    name="pylabfea",
    description="Python Laboratory for Finite Element Analysis",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Alexander Hartmaier",
    author_email="alexander.hartmaier@rub.de",
    python_requires=">=3",
    classifiers=[
        "Development Status :: 3 - Beta",
        "Intended Audience :: Experts",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    license="GPLv3",
    packages=find_packages(include=["pylabfea", "pylabfea.*"]),
    install_requires=[
        "numpy",
        "matplotlib",
        "scipy",
        "scikit-learn",
        "pytest"
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest>=3"],
)