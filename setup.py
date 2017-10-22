#!/usr/bin/env python

from distutils.core import setup

setup(
    name='Data Science Tools',
    version='0.3.1',
    description='Data Science Tools for Spark and scikit-learn',
    author='Dmitri Babaev',
    author_email='dmitri.babaev@gmail.com',
    packages=['ds-tools'],
    requires=['pandas', 'six', 'numpy', 'scikit-learn', ],
)
