#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_op_amplitude_2d.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot the order parameter amplitude with imshow, for Fig. 6.
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
updateNumpyDataFile = True
temperature = 0.1
pbSideLength = 60.0
grainCornerLength = 4.0
fermiSurfaceLabel = 3
# Fermisurfaces:
# 1: circular
# 2: FS1
# 3: FS2

# Plot the full grain in separate figure (not saved to file).
plotFullFigure = False

if fermiSurfaceLabel == 2:
    plotLabel = r'$\mathrm{(a)}\, \#1$'
elif fermiSurfaceLabel == 3:
    plotLabel = r'$\mathrm{(b)}\, \#2$'

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

# Python latex plot stuff
plt.rc('font',**{'family':'sans-serif',
                 'sans-serif':['Computer Modern Sans serif']})
plt.rcParams['text.latex.preamble']=[r"\usepackage{amssymb}",
                                     r"\usepackage{amsmath}"]
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
dataField = run.delta.abs

# Center data around 0 and/or set data min/max
dataMin = 0.0
dataMax = np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1])
imshowMin = 0.0
imshowMax = 1.5

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
# PLOT DATA
#======================================================================#
# Labels for color bar
cbarTicks = [0, imshowMax]
cbarLabels = [r'$0$', r'$%2.1f$' % imshowMax]
    
# Set colormap and its 'bad color' (color outside grain)
cmap = plt.cm.get_cmap('jet')
cmap.set_bad(badColor,1.0)
#======================================================================#
# FULL GRAIN PLOT
#======================================================================#
if plotFullFigure:
    # Create figure and subplots
    fig = plt.figure(figsize=(fig2Width*inchesPerPt, 1.85*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
    fig.subplots_adjust(left=0.07,
                        bottom=0.06,
                        right=0.98,
                        top=0.92,
                        wspace=0,
                        hspace=0)

    gs = gridspec.GridSpec(100,100)

    ax_cbar = plt.subplot(gs[0:5,:])
    ax = plt.subplot(gs[5:,:])

    # Plot data
    cax = ax.imshow(dataField, cmap=cmap,
                    vmin=imshowMin, vmax=imshowMax,
                    interpolation=interpolation,
                    aspect=aspect,
                    origin='lower')

    # Draw cut portion
    ax.plot([xMin,xMax],[yMin,yMin],'r-',lw=0.5)
    ax.plot([xMin,xMax],[yMax,yMax],'r-',lw=0.5)
    ax.plot([xMin,xMin],[yMin,yMax],'r-',lw=0.5)
    ax.plot([xMax,xMax],[yMin,yMax],'r-',lw=0.5)

    # Axis ticks and labels
    ax.set_xlabel(r'$x\, \mathrm{(index)}$',fontsize=font_size,labelpad=-5)
    ax.set_ylabel(r'$y\, \mathrm{(index)}$',fontsize=font_size)
    ax.tick_params( axis='both', which='major', labelsize=tick_size )
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Color bar
    cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
    cbar.set_ticks(cbarTicks)
    cbar.set_ticklabels(cbarLabels)
    cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    # Text label
    cbar.set_label(r'$|\Delta_d|/k_{\mathrm{B}}T_{\mathrm{c}}$',
                    #rotation=270,
                    labelpad=-10,
                    fontsize=font_size)

    # Smoother appearance
    cbar.solids.set_edgecolor("face")

#======================================================================#
# CROP PLOT
#======================================================================#
fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 1.2*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.08,
                     bottom=0.04,
                     right=0.97,
                     top=0.91,
                     wspace=0,
                     hspace=0)



gs = gridspec.GridSpec(100,100)

ax_cbar2 = plt.subplot(gs[0:5,:])
ax2 = plt.subplot(gs[5:,:])

##################################################
# IMSHOW 
##################################################
cax2 = ax2.imshow(dataCrop, cmap=cmap,
                vmin=imshowMin, vmax=imshowMax,
                extent=cropExtent,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax2.set_xlabel(r'$x/\xi_0$',fontsize=font_size, labelpad=-6)
ax2.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
ax2.tick_params( axis='both', which='major', labelsize=tick_size )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

props = dict(boxstyle='square', facecolor='white')#, alpha=0.5)
ax2.text(0.02, 0.96, plotLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2.transAxes,
            fontsize=font_size,
            bbox=props,
            )

# Colorbar
cbar2 = fig2.colorbar(cax2, cax=ax_cbar2, orientation='horizontal', ticks=cbarTicks)
cbar2.set_ticks(cbarTicks)
cbar2.set_ticklabels(cbarLabels)
cbar2.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

# Text label
cbar2.set_label(r'$|\Delta_d|/k_{\mathrm{B}}T_{\mathrm{c}}$',
                #rotation=270,
                labelpad=-10,
                fontsize=font_size)
# Smoother appearance
cbar2.solids.set_edgecolor("face")

# Save to file
fig2.savefig('%s/svg/qc_delta_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/qc_delta_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/qc_delta_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)


plt.show()