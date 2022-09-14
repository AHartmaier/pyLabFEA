#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages, Extension

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]

setup(
    name='pylabfea',
    author="Alexander Hartmaier",
    author_email='alexander.hartmaier@rub.de',
    python_requires='>=3',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Experts',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    description="Python Laboratory for Finite Element Analysis",
    install_requires=['numpy', 'matplotlib', 'scipy', 'scikit-learn', 'pandas', 'fireworks',
                      'pytest'],
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='FEA',
    packages=find_packages('src', exclude=["*tests*"]),
    package_dir={'':'src'},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AHartmaier/pyLabFEA',
    version='4.3',
    zip_safe=False,
)
