#!/usr/bin/env python

from distutils.core import setup

setup(
    name='ds-tools',
    version='0.3.1',
    description='Data Science Tools for Spark and scikit-learn',
    author='Dmitri Babaev',
    author_email='dmitri.babaev@gmail.com',
    packages=['scikit-learn', 'numpy', 'pandas'],
)
