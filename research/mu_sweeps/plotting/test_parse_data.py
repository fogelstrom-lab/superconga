#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_parse_data.py
~~~~~~~~~~~~~~~~~~~~
@description: Test parsing data (.json, .dat or .grd files).
"""

#======================================================================#
# IMPORTS
#======================================================================#
from __future__ import division

import os
import sys
import numpy as np

#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
sys.path.insert(0, '../../../tools/')

from pythonLibraries.fileIO import readFiles as RF
from pythonLibraries.plotting import plotFields as PF
from pythonLibraries.dataAnalysis import dataStructs as DS

fieldList = ['delta']
filenameList = ['order_parameter']

testFilePath = '../../../build/data/T0.5000'
run = DS.runData(testFilePath, fields_in=fieldList, fileNames_in=filenameList,verbose=True)

print('\n')

# Print parameters
print(40*'=')
print(' Parsed parameters from config file: ')
print(40*'=')
print("\n".join("{}\t{}".format(k, v) for k, v in run.paramsParsed.items()))
print(40*'=')

print('\n')

print(40*'=')
print(' Parameters with aliases added: ')
print(40*'=')
print("\n".join("{}\t{}".format(k, v) for k, v in run.params.items()))
print(40*'=')
