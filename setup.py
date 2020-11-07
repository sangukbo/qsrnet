# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='qsrnet',
    version='0.1.0',
    description='qsrnet',
    author='Sang Uk Lee',
    author_email='sangukbo@mit.edu',
    url='https://github.com/sangukbo/qsrnet',
    packages=find_packages(exclude=('tests', 'docs'))
)
