#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

test_requirements = ['pytest>=3', ]
setup_requirements = ['pytest-runner', ]
install_requires = ['numpy', 'matplotlib', 'scipy', 'scikit-learn', 'pytest']

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
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Python Laboratory for Finite Element Analysis",
    install_requires=['numpy', 'matplotlib', 'scipy', 'scikit-learn', 'pytest'],
    license="GNU General Public License v3",
    long_description=readme,
    include_package_data=True,
    keywords='FEA',
    packages=find_packages('src', exclude=["*tests*"]),
    package_dir={'': 'src'},
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/AHartmaier/pyLabFEA',
    version='4.3.6',
    zip_safe=False,
)
