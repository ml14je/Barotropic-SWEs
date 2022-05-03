#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joseph Elmes: NERC-Funded PhD Researcher in Applied Mathematics
University of Leeds : Leeds LS2 9JT : ml14je@leeds.ac.uk

Python 3.7: Tue Nov  1 18:20:21 2021
"""

##from setuptools import setup
from distutils.core import setup

setup(name='Barotropic SWEs DG-FEM',
      version= '0.1',
      description= 'Barotropic SWEs DG-FEM',
      url= 'http://github.com/ml14je/barotropicSWEs',
      author= 'Joseph Elmes',
      author_email= 'ml14je@leeds.ac.uk',
      license='None',
      install_requires=[
          'numpy', 'scipy', 'pandas', 'matplotlib',
          'sympy', 'bottleneck', 'cython', 'numba',
          'oceanmesh'
#          'ChannelWaves1D', 'ppp', 'oceanmesh'
      ],
      packages=['barotropicSWEs'],
      package_data={'defaults': ['supporting_files/defaults.json']},
      zip_safe=False)
