#!/usr/bin/env python

from distutils.core import setup

setup(
    name=''ds-tools',
    version='0.3.1',
    description='Data Science Tools for Spark and scikit-learn',
    author='Dmitri Babaev',
    author_email='dmitri.babaev@gmail.com',
    install_requires=['pandas>=0.20', 'numpy>=1.14', 'scikit-learn>=0.19', ],
)
