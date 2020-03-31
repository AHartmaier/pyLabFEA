#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]

setup(
    author="Alexander Hartmaier",
    author_email='alexander.hartmaier@rub.de',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Python Laboratory for Finite Element Analysis",
    install_requires=['numpy', 'matplotlib', 'scipy', 'scikit-learn'],
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='FEA',
    name='pylabfea',
    packages=find_packages(exclude=["*tests*"]),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AHartmaier/pyLabFEA',
    version='1.1',
    zip_safe=False,
)
