#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_phase_2d_inset.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot phase in small format to be used as inset in Fig. 6.
"""

# BdG data files are found according to:
# ../data/bdg/bs2/Curr.dat
# ../data/bdg/bs2/OP.dat
# ../data/bdg/bs4/Curr.dat
# ../data/bdg/bs4/OP.dat
# QC data files are found according to:
# ../data/qc/fs1_R40.000_r4.000_T0.1000

#======================================================================#
# IMPORTS
#======================================================================#
from __future__ import division

import os
import sys
import colorsys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

from scipy.ndimage.interpolation import rotate

#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
sys.path.insert(0, '../../../tools/')
from pythonLibraries.plotting import cmapsExtra as cm

from pythonLibraries.fileIO import readFiles as RF
from pythonLibraries.plotting import plotFields as PF
from pythonLibraries.dataAnalysis import dataStructs as DS

#======================================================================#
# PARAMETERS
#======================================================================#

# Read raw data even if numpy data file exists
updateNumpyDataFile = False
temperature = 0.1
pbSideLength = 60.0
grainCornerLength = 4.0
fermiSurfaceLabel = 3
# Fermisurfaces:
# 1: circular
# 2: FS1
# 3: FS2

# Plot a figure of the full grain, with a box indicating the cropped version.
# Useful to make sure that the correct min/max are set.
plotFullFigure = False

# Indices for cropping. Selected by looking at phase oscillations
xMin = 523
xMax = 693
yMin = 520
yMax = 620


xRange = xMax - xMin
yRange = yMax - yMin

#======================================================================#
# PLOTTING PARAMETERS
#======================================================================#
# Taken from guides on Python publication quality plotting, ages ago
goldenRatio = (np.sqrt(5.0)-1.0)/2.0
# Taken from APS guidelines
inchesPerPt = 1.0/72.27
fig2Width = 512.0
# see: https://www.overleaf.com/learn/latex/Lengths_in_LaTeX

# Python latex plot stuff
plt.rc('font',**{'family':'sans-serif',
                 'sans-serif':['Computer Modern Sans serif']})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}",
                                     r"\usepackage{amsmath}",
                                     r"\usepackage{bm}"]
plt.rc('text', usetex=True)
plt.rc('svg', fonttype='none')

tick_size = 18
font_size = 18
cbar_size = 18

dpi = 100

# Color outside grain
badColor = '#c0c0c0'

# Examples of interpolations: bicubic, bilinear, none. None is more accurate now
interpolation = 'none'

# Aspect ratio for imshow: 1 or auto. 1 means that a square is not distorted.
aspect = 1.0

#======================================================================#
# FIELDS
#======================================================================#
fieldList = [
'delta',
'current',
'ps',
]

filenameList = [
'order_parameter',
'current_density',
'order_parameter',
]

#======================================================================#
# FILES
#======================================================================#
dataBasePath = '../data'
figSavePath = 'figures'

# Replace dots with dashes in filenames, so numpy plays nice
temperature_str = str('%2.4f' % temperature).replace('.','-')
pbSideLength_str = str('%2.3f' % pbSideLength).replace('.','-')
grainCornerLength_str = str('%2.3f' % grainCornerLength).replace('.','-')

# Path to quasiclassics data
filePathQC = str('%s/qc/fs%d_R%2.3f_r%2.3f_T%2.4f' % (dataBasePath, fermiSurfaceLabel, pbSideLength, grainCornerLength, temperature))
# Path to numpy data file (for faster load times)
filePathNumpy = str('%s/python/rundata_fs%d_R%s_r%s_T%s.npy' % (dataBasePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str))

# Create numpy data directory if not existing
if not os.path.exists(str('%s/python/' % dataBasePath)):
    pythonDirectory = '%s/python/' % dataBasePath
    print('Create directory: %s' % pythonDirectory)
    os.mkdir(pythonDirectory)

# Read quasiclassics data
if (not os.path.isfile(filePathNumpy)) or updateNumpyDataFile:
    # Read raw data
    run = DS.runData(filePathQC, fields_in=fieldList, fileNames_in=filenameList,
                     zeroToNaN=True)
    
    # Save numpy data file
    np.save(filePathNumpy, run)
    print('Save: %s' % filePathNumpy)
else:
    # Read numpy data file
    run = np.load(filePathNumpy).item()
    print('Load: %s' % filePathNumpy)

#======================================================================#
# GENERAL DATA
#======================================================================#
# Print meta data parameters
print(run.params)

# Fetch some useful parameters
sideLength = run.params['sideLength']
numLatticePoints = run.params['numLatticePoints']
grainMargin = run.params['grainMargin']

xLength = sideLength
yLength = sideLength

# 100% - %(grain margin)
grainPercent = 1.0 - grainMargin

# Calculate the index where a square grain starts/ends. Is useful when testing
# that the correct thing is plotted, so pleace leave it here even if not used.
minIndex = run.params['minIndex']
maxIndex = run.params['maxIndex']
sideLengthLatticePoints = maxIndex - minIndex

# Calculate ratio between coherence lengths and discrete lattice points
coherenceSize = float(sideLengthLatticePoints) / sideLength

extentScale = coherenceSize
pixelMargin = 0.5
pixelMarginScale = pixelMargin/float(extentScale)

extentMinX = -pixelMarginScale
extentMaxX = (xRange+1)/float(extentScale) - pixelMarginScale
extentMinY = -pixelMarginScale
extentMaxY = (yRange+1)/float(extentScale) - pixelMarginScale
cropExtent = (extentMinX, extentMaxX, extentMinY, extentMaxY)

print('Min index: % s, max index: %s, points per coherence length: %s' % (minIndex, maxIndex, coherenceSize))


##################################################
# DATA
##################################################
# Fetch data field
dataField = run.delta.phase / np.pi

# Center data around 0
dataMaxUncentered = np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1])
dataMinUncentered = np.nanmin(dataField[yMin:yMax+1,xMin:xMax+1])
dataRange = abs(dataMaxUncentered - dataMinUncentered)
dataMin = -0.5*dataRange
dataMax = 0.5*dataRange
dataField = (dataField - dataMinUncentered) + dataMin

print('Data min: %s' % np.nanmin(dataField[yMin:yMax+1,xMin:xMax+1]))
print('Data max: %s' % np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1]))

# Rotate field
dataField[dataField != dataField] = 0
dataField = rotate(dataField, angle=45)
# Try to set nan outside grain, but does not work fully for rotated fields
#dataField[dataField == 0] = np.nan

# Create cropped data
dataCrop = dataField[yMin:yMax+1,xMin:xMax+1]

#======================================================================#
# FIGURE SETUP
#======================================================================#
fig = plt.figure(figsize=(0.4*fig2Width*inchesPerPt, 0.6*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig.subplots_adjust(left=0.14,
                     bottom=0.0,
                     right=0.91,
                     top=0.72,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar = plt.subplot(gs[0:10,:])
ax = plt.subplot(gs[10:,:])

# Set colormap and its 'bad color' (color outside grain)
cmap = plt.cm.get_cmap("RdBu_r")
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW 
##################################################
cax = ax.imshow(dataCrop, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                extent=cropExtent,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')


# Axis ticks and labels
ax.tick_params( axis='both', which='major', labelsize=tick_size )
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([])
ax.set_yticks([])

##################################################
# COLOR BAR
##################################################
cbarTicks = [dataMin, 0, dataMax]
    
cbarLabels = [r'$\bm{%2.2f}$' % dataMin, '', r'$\bm{%2.2f}$' % dataMax]
cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
cbar.set_ticks(cbarTicks)
cbar.set_ticklabels(cbarLabels)
cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Text label
cbarLabel = r'$\bm{\chi / \pi}$'
cbar.set_label(cbarLabel,
                #rotation=270,
                labelpad=-10,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")

# Save to file
fig.savefig('%s/svg/qc_phase_small_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/qc_phase_small_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/qc_phase_small_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

plt.show()