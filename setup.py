# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

setup(
    name='qsrnet',
    version='0.1.0',
    description='qsrnet',
    long_description=readme,
    author='Sang Uk Lee',
    author_email='sangukbo@mit.edu',
    url='https://github.com/sangukbo/qsrnet',
    packages=find_packages(exclude=('tests', 'docs'))
)
