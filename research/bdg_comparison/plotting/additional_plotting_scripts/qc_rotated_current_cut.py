#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_rotated_current_cut.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a rotated current from quasiclassical data, with a cut specified by coordinates.
"""
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

from scipy.ndimage.interpolation import rotate

#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
#sys.path.insert(0, '/home/patric/projects/suprcongpu/tools/')
sys.path.insert(0, '../../../../tools/')
#from pythonLibraries.plotting import latexPlot
#latexFontSize = latexPlot.latexPlot(usetex = False, fsize_in=14)
from pythonLibraries.plotting import cmapsExtra as cm

from pythonLibraries.fileIO import readFiles as RF
#from pythonLibraries.plotting import miscPlot as MP
from pythonLibraries.plotting import plotFields as PF
from pythonLibraries.dataAnalysis import dataStructs as DS

from pythonLibraries.plotting.old_streamplot import streamplot

#======================================================================#
# FUNCTIONS
#======================================================================#

#======================================================================#
# PARAMETERS
#======================================================================#
# BdG data files:
# ../data/bdg/bs2/Curr.dat
# ../data/bdg/bs2/OP.dat
# ../data/bdg/bs4/Curr.dat
# ../data/bdg/bs4/OP.dat
# QC data files:
# ../data/qc/fs1_R40.000_r4.000_T0.1000

fieldList = [
'delta',
'current',
]

filenameList = [
'order_parameter',
'current_density',
]

dataBasePath = '../../data'
figSavePath = '../figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = True
temperature = 0.1
pbSideLength = 60.0
grainCornerLength = 4.0
fermiSurfaceLabel = 2

plotFullFigure = True

#======================================================================#
# PLOTTING PARAMETERS
#======================================================================#
goldenRatio = (np.sqrt(5.0)-1.0)/2.0
inchesPerPt = 1.0/72.27
fig2Width = 512.0
# \textwidth = 449.55365pt = 15.859253764 cm

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
#badColor = '#ffffff'
badColor = '#c0c0c0'

interpolation = 'none' #bicubic, bilinear


if plotFullFigure:
    fig = plt.figure(figsize=(fig2Width*inchesPerPt, 1.85*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
    fig.subplots_adjust(left=0.07,
                        bottom=0.06,
                        right=0.96,
                        top=0.92,
                        wspace=0,
                        hspace=0)

    gs = gridspec.GridSpec(100,100)

    ax_cbar = plt.subplot(gs[0:3,:])
    ax = plt.subplot(gs[5:,:])



fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 1.3*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.12,
                     bottom=0.03,
                     right=0.96,
                     top=0.88,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar2 = plt.subplot(gs[0:3,:])
ax2 = plt.subplot(gs[5:,:])

#======================================================================#
# FIELDS
#======================================================================#

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
                     zeroToNaN=False)
    
    # Save numpy data file
    np.save(filePathNumpy, run)
    print('Save: %s' % filePathNumpy)
else:
    # Read numpy data file
    run = np.load(filePathNumpy).item()
    print('Load: %s' % filePathNumpy)


#======================================================================#
# PLOT DATA
#======================================================================#

##################################################
# FIELDS
##################################################
print run.params

#for s_key in systems.keys():
sideLength = run.params['sideLength']
numLatticePoints = run.params['numLatticePoints']
grainMargin = run.params['grainMargin']
b = (sideLength/(1.0 - grainMargin) - sideLength) * 0.5

extent = (-b, sideLength+b, -b, sideLength+b)#(0,sideLength,0,sideLength*0.25)
aspect = 1.0#'auto'

xLength = sideLength
yLength = sideLength

minIndex = run.params['minIndex']
maxIndex = run.params['maxIndex']

##################################################
# DATA
##################################################
currentMin = 0.0#np.nanmin(run.current.abs)
currentMax = np.nanmax(run.current.abs)

# Mikaels characteristic colormap for currents
cmap = mpl.cm.get_cmap('myred')

# Set color outside grain: This is problematic with rotation...
cmap.set_bad(badColor,1.0)
currentPlot = run.current.abs
currentPlotX = run.current.x
currentPlotY = run.current.y

# Failed attempt to apply mask to remove elements outside grain - does not work with rotation
#mask = (currentPlot == 0)
#currentPlot = rotate(currentPlot, angle=45)
#maskRotated = np.ceil(rotate(mask, angle=45))
#currentPlot[maskRotated] = np.nan
#thresholdIndices = currentPlot < 1e-12
#currentPlot[thresholdIndices] = np.nan

currentPlot = rotate(currentPlot, angle=45)
currentPlotX = rotate(currentPlotX, angle=45)
currentPlotY = rotate(currentPlotY, angle=45)
currentPlot[currentPlot == 0] = np.nan
#currentPlotX[currentPlotX == 0] = np.nan
#currentPlotY[currentPlotY == 0] = np.nan

# Indices for cut
xMin = 400
xMax = 733
yMin = 520
yMax = 650

cbarTicks = [currentMin, currentMax]

if np.allclose(0.0, currentMin):
    currentMin_str = r'$0$'
else:
    currentMin_str = r'$%2.2f$' % currentMin
    
cbarLabels = [currentMin_str, r'$%2.2f$' % currentMax]
cbarLabel = r'$\left|\mathbf{j}\right|/j_{\mathrm{d}}$'

##################################################
# IMSHOW PLOT: FIG 1
##################################################
if plotFullFigure:
    cax = ax.imshow(currentPlot, cmap=cmap,
                    vmin=currentMin, vmax=currentMax,
                    #extent=extent,
                    interpolation=interpolation,
                    aspect=aspect,
                    origin='lower')
    #del(currentPlot)

    # Draw cut portion
    ax.plot([xMin,xMax],[yMin,yMin],'r-',lw=0.5)
    ax.plot([xMin,xMax],[yMax,yMax],'r-',lw=0.5)
    ax.plot([xMin,xMin],[yMin,yMax],'r-',lw=0.5)
    ax.plot([xMax,xMax],[yMin,yMax],'r-',lw=0.5)

    # Axis ticks and labels
    #ax.set_xticks(tickLabels)
    #ax.set_yticks(tickLabels)
    ax.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
    ax.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
    ax.tick_params( axis='both', which='major', labelsize=tick_size )
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #ax.set_xticks([])
    #ax.set_yticks([])

    ##################################################
    # COLOR BAR: FIG 1
    ##################################################
    cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
    cbar.set_ticks(cbarTicks)
    cbar.set_ticklabels(cbarLabels)
    cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
    cbar.ax.xaxis.set_ticks_position('top')
    cbar.ax.xaxis.set_label_position('top')

    # Text label
    cbar.set_label(cbarLabel,
                    #rotation=270,
                    #labelpad=10,
                    fontsize=font_size)

    # Smoother appearance
    cbar.solids.set_edgecolor("face")

##################################################
# QUIVER PLOT: FIG 2
##################################################
currentCut = currentPlot[yMin:yMax+1,xMin:xMax+1]
rotationAngle = -(np.pi*0.25)
cosAngle = np.cos(rotationAngle)
sinAngle = np.sin(rotationAngle)
currentCutX = currentPlotX[yMin:yMax+1,xMin:xMax+1] * cosAngle - currentPlotY[yMin:yMax+1,xMin:xMax+1] * sinAngle
currentCutY = currentPlotX[yMin:yMax+1,xMin:xMax+1] * sinAngle + currentPlotY[yMin:yMax+1,xMin:xMax+1] * cosAngle

quiverDensity   = 10  #[!!!]
quiverScale      = 50
quiverWidth      = 0.004#0.003#0.0015
quiverHeadWidth  = 5.0#7.0
quiverHeadLength = 5.0
quiverColor = 'w'
quiverEdgeColor = 'k'
quiverEdgeLineWidth = 0.5
quiverPivot = 'middle'
quiverDataScale = 1.0

yNum, xNum = np.shape(currentCutX)

# Center ticks on pixel
pixelMargin = 0.5
pixelMarginScale = pixelMargin/float(quiverDensity)

# Calculate extent to make quiver overlap with imshow correctly
extentMinX = -0.5
extentMaxX = xNum/float(quiverDensity)
extentMinY = -0.5
extentMaxY = yNum/float(quiverDensity)
extentQuiver = (extentMinX, extentMaxX, extentMinY, extentMaxY)

# Select and scale data for quiver 
quiverX = quiverDataScale*currentCutX[::quiverDensity, ::quiverDensity]
quiverY = quiverDataScale*currentCutY[::quiverDensity, ::quiverDensity]

#Q = ax_quiv.quiver(-U, -V, width = 0.002)#, headwidth=100.0, headlength=100.0)
quiverAx = ax2.quiver(quiverX/currentMax,#np.sqrt(quiverX**2+quiverY**2),  #[!!!]
                    quiverY/currentMax,#np.sqrt(quiverX**2+quiverY**2),  #[!!!]
                    scale      = quiverDensity,
                    width      = quiverWidth,
                    headwidth  = quiverHeadWidth,
                    headlength = quiverHeadLength,
                    color = quiverColor,
                    pivot = quiverPivot,
                    edgecolor = quiverEdgeColor,
                    linewidth = quiverEdgeLineWidth,
                    )

##################################################
# IMSHOW PLOT: FIG 2
##################################################

cax2 = ax2.imshow(currentCut, cmap=cmap,
                vmin=currentMin, vmax=currentMax,
                extent=extentQuiver,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax2.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
ax2.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
ax2.tick_params( axis='both', which='major', labelsize=tick_size )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

##################################################
# COLOR BAR: FIG 2
##################################################
cbar2 = fig2.colorbar(cax2, cax=ax_cbar2, orientation='horizontal', ticks=cbarTicks)
cbar2.set_ticks(cbarTicks)
cbar2.set_ticklabels(cbarLabels)
cbar2.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

# Text label
cbar2.set_label(cbarLabel,
                #rotation=270,
                #labelpad=10,
                fontsize=font_size)
# Smoother appearance
cbar2.solids.set_edgecolor("face")

##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.savefig('%s/svg/qc_current_rotated_cut_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/qc_current_rotated_cut_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/qc_current_rotated_cut_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

plt.show()