# -*- coding: utf-8 -*-
# setup.py
# author: Antoine Passemiers

from setuptools import setup


packages = [
    'geneRNI',
    'geneRNI.grn',
    'geneRNI.models'
]


setup(
    name='geneRNI',
    version='1.0.0',
    description='',
    url='https://github.com/janursa/geneRNI',
    author='Jalil Nourisa, Antoine Passemiers',
    packages=packages
)
