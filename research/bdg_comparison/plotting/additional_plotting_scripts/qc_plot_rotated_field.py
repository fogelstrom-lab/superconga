#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_plot_rotated_field.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a rotated field from quasiclassical data, with a cut specified by coordinates.
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
sys.path.insert(0, '../../../../tools/')
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
# Fermisurfaces
# 1: circular
# 2: FS1
# 3: FS2

# Possible fields to plot are: ['current', 'phase', 'delta', 'ps']
# current: will plot quiver on imshow
# phase and delta: plot imshow
# ps: to be implemented, will plot quiver on imshow
plotField = 'ps'

# Plot a figure of the full grain, with a box indicating the cropped version.
# Useful to make sure that the correct min/max are set.
plotFullFigure = True

# Field-specific settings
# centerData: center data range around 0 for phase, else from 0 to dataMax
fs = {
'current': {'centerData': False,
            'colormap': mpl.cm.get_cmap('myred'),
            'label': r'$\left|\mathbf{j}\right|/j_{\mathrm{d}}$',
            'labelPad': -10,
            'plotCuts': True,
            'cutMin': 0,
            'cutMax': 'data',
            'cutMinY': 0,
            'cutMaxY': 'data',
            'imshowMin': 0,
            'imshowMax': 0.18,
            'cutLabelX': r'$\left|\mathbf{j}(x)\right|/j_{\mathrm{d}}$',
            'cutLabelY': r'$\left|\mathbf{j}(y)\right|/j_{\mathrm{d}}$',
            'xOffset': -42,
            'quiver': True,
            },
'delta': {  'centerData': False,
            'colormap': plt.cm.get_cmap('jet'),
            'label': r'$\Delta_d/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'labelPad': -10,
            'plotCuts': True,
            'cutMin': 0,
            'cutMax': 1.5,
            'cutMinY': 0,
            'cutMaxY': 1.5,
            'imshowMin': 0,
            'imshowMax': 1.5,
            'cutLabelX': r'$\Delta_d(x)/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'cutLabelY': r'$\Delta_d(y)/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'xOffset': 0,
            'quiver': False,
            },
'phase': {  'centerData': True, 
            'colormap': plt.cm.get_cmap('RdBu_r'),
            'label': r'$\chi/\pi$',
            'labelPad': 0,
            'plotCuts': True,
            'cutMin': -0.5,
            'cutMax': 0.5,
            'cutMinY': 0,
            'cutMaxY': 0.5,
            'imshowMin': -0.5,
            'imshowMax': 0.5,
            'cutLabelX': r'$\chi(x)/\pi$',
            'cutLabelY': r'$\chi(y)/\pi$',
            'xOffset': 0,
            'quiver': False,
            },
'ps': {     'centerData': False,
            'colormap': mpl.cm.get_cmap('myred'),
            'label': r'$\left|\mathbf{p}_{\mathrm{s}}\right|/p_0$',
            'labelPad': 0,
            'plotCuts': True,
            'cutMin': 0,
            'cutMax': 'data',
            'cutMinY': 0,
            'cutMaxY': 'data',
            'imshowMin': 0,
            'imshowMax': 'data',
            'cutLabelX': r'$\left|\mathbf{p}_{\mathrm{s}}(x)\right|/p_0$',
            'cutLabelY': r'$\left|\mathbf{p}_{\mathrm{s}}(y)\right|/p_0$',
            'xOffset': 0,#-42,
            'quiver': True,
            },
}

if not fs.has_key(plotField):
    print('Field %s is not implemented!' % (plotField))
    exit()

if not fermiSurfaceLabel in [1,2,3]:
    print('Unknown Fermi surface label %d !' % (fermiSurfaceLabel))
    exit()

if fermiSurfaceLabel == 2:
    imshowPlotLabel = r'$\mathrm{(a)}\, \#1$'
elif fermiSurfaceLabel == 3:
    imshowPlotLabel = r'$\mathrm{(b)}\, \#2$'
else:
    imshowPlotLabel = r'$\mathrm{(c)}\, \mathrm{circular}$'

# Indices for cropping. Selected by looking at phase oscillations
xMin = 523 + fs[plotField]['xOffset']
xMax = 693 + fs[plotField]['xOffset']
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
dataBasePath = '../../data'
figSavePath = '../figures'

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

# Calculate the index where a square grain starts/ends. Is useful when testing
# that the correct thing is plotted, so pleace leave it here even if not used.
minIndex = run.params['minIndex']
maxIndex = run.params['maxIndex']
sideLengthLatticePoints = maxIndex - minIndex

# Calculate ratio between coherence lengths and discrete lattice points
coherenceSize = float(sideLengthLatticePoints) / sideLength

print('Min index: % s, max index: %s, points per coherence length: %s' % (minIndex, maxIndex, coherenceSize))


##################################################
# DATA
##################################################
# Fetch data field
if plotField == 'current':
    dataField = run.current.abs
elif plotField == 'delta':
    dataField = run.delta.abs
elif plotField == 'phase':
    dataField = run.delta.phase / np.pi
elif plotField == 'ps':
    # Scale by 2pi to have superflow in units of kB*Tc/vF
    dataField = run.ps.abs * (2.0 * np.pi)
else:
    print('Field %s is not implemented!' % (plotField))
    exit()



# Center data around 0 and/or set data min/max
if fs[plotField]['centerData']:
    dataMaxUncentered = np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1])
    dataMinUncentered = np.nanmin(dataField[yMin:yMax+1,xMin:xMax+1])
    dataRange = abs(dataMaxUncentered - dataMinUncentered)
    dataMin = -0.5*dataRange
    dataMax = 0.5*dataRange
    dataField = (dataField - dataMinUncentered) + dataMin
else:
    dataMin = 0.0
    dataMax = np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1])

print('Data min: %s' % np.nanmin(dataField[yMin:yMax+1,xMin:xMax+1]))
print('Data max: %s' % np.nanmax(dataField[yMin:yMax+1,xMin:xMax+1]))

if fs[plotField]['cutMax'] == 'data':
    fs[plotField]['cutMax'] = dataMax
if fs[plotField]['cutMaxY'] == 'data':
    fs[plotField]['cutMaxY'] = dataMax
if fs[plotField]['imshowMax'] == 'data':
    fs[plotField]['imshowMax'] = dataMax

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
if fs[plotField]['centerData']:
    cbarTicks = [fs[plotField]['imshowMin'], 0, fs[plotField]['imshowMax']]
    cbarLabels = [r'$%2.2f$' % fs[plotField]['imshowMin'], r'$0$', r'$%2.2f$' % fs[plotField]['imshowMax']]
else:
    cbarTicks = [0, fs[plotField]['imshowMax']]
    cbarLabels = [r'$0$', r'$%2.2f$' % fs[plotField]['imshowMax']]
    
# Set colormap and its 'bad color' (color outside grain)
cmap = fs[plotField]['colormap']
cmap.set_bad(badColor,1.0)
#======================================================================#
# FULL GRAIN PLOT
#======================================================================#
if plotFullFigure:
    # Create figure and subplots
    fig = plt.figure(figsize=(fig2Width*inchesPerPt, 1.85*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
    fig.subplots_adjust(left=0.07,
                        bottom=0.06,
                        right=0.96,
                        top=0.92,
                        wspace=0,
                        hspace=0)

    gs = gridspec.GridSpec(100,100)

    ax_cbar = plt.subplot(gs[0:5,:])
    ax = plt.subplot(gs[5:,:])

    # Plot data
    cax = ax.imshow(dataField, cmap=cmap,
                    vmin=fs[plotField]['imshowMin'], vmax=fs[plotField]['imshowMax'],
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
    cbar.set_label(fs[plotField]['label'],
                    #rotation=270,
                    labelpad=-10,
                    fontsize=font_size)

    # Smoother appearance
    cbar.solids.set_edgecolor("face")

#======================================================================#
# CROP PLOT
#======================================================================#
fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 1.2*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.09,
                     bottom=0.05,
                     right=0.97,
                     top=0.91,
                     wspace=0,
                     hspace=0)



gs = gridspec.GridSpec(100,100)

ax_cbar2 = plt.subplot(gs[0:5,:])
ax2 = plt.subplot(gs[5:,:])

##################################################
# QUIVER PLOT
##################################################
if fs[plotField]['quiver']:
    # X and Y components for quiver
    if plotField == 'current':
        dataFieldX = run.current.x
        dataFieldY = run.current.y
        quiverDataScale = 1.0/dataMax
    elif plotField == 'ps':
        dataFieldX = run.ps.x * (2.0 * np.pi)
        dataFieldY = run.ps.y * (2.0 * np.pi)
        quiverDataScale = 1.0/(dataMax)
    else:
        print('Unknown quiver field %s !' % (plotField))
        exit()

    # Rotate fields
    dataFieldX[dataFieldX != dataFieldX] = 0
    dataFieldY[dataFieldY != dataFieldY] = 0
    dataFieldX = rotate(dataFieldX, angle=45)
    dataFieldY = rotate(dataFieldY, angle=45)

    # Rotate data values for quiver
    rotationAngle = -(np.pi*0.25)
    cosAngle = np.cos(rotationAngle)
    sinAngle = np.sin(rotationAngle)
    dataCropX = dataFieldX[yMin:yMax+1,xMin:xMax+1] * cosAngle - dataFieldY[yMin:yMax+1,xMin:xMax+1] * sinAngle
    dataCropY = dataFieldX[yMin:yMax+1,xMin:xMax+1] * sinAngle + dataFieldY[yMin:yMax+1,xMin:xMax+1] * cosAngle

    # Quiver settings
    quiverDensity   = 10
    quiverScale      = 50
    quiverWidth      = 0.004
    quiverHeadWidth  = 5.0
    quiverHeadLength = 5.0
    quiverColor = 'w'
    quiverEdgeColor = 'k'
    quiverEdgeLineWidth = 0.5
    quiverPivot = 'middle'

    yNum, xNum = np.shape(dataCropX)

    # Center ticks on pixel
    extentScale = quiverDensity
    pixelMargin = 0.5
    pixelMarginScale = pixelMargin/float(extentScale)

    # Calculate extent to make quiver overlap with imshow correctly
    extentMinX = -pixelMarginScale
    extentMaxX = xNum/float(extentScale) - pixelMarginScale
    extentMinY = -pixelMarginScale
    extentMaxY = yNum/float(extentScale) - pixelMarginScale
    cropExtent = (extentMinX, extentMaxX, extentMinY, extentMaxY)

    # Select and scale data for quiver 
    quiverX = quiverDataScale*dataCropX[::quiverDensity, ::quiverDensity]
    quiverY = quiverDataScale*dataCropY[::quiverDensity, ::quiverDensity]

    quiverAx = ax2.quiver(quiverX,
                        quiverY,
                        scale      = quiverDensity,
                        width      = quiverWidth,
                        headwidth  = quiverHeadWidth,
                        headlength = quiverHeadLength,
                        color = quiverColor,
                        pivot = quiverPivot,
                        edgecolor = quiverEdgeColor,
                        linewidth = quiverEdgeLineWidth,
                        )
else:
    extentScale = coherenceSize
    pixelMargin = 0.5
    pixelMarginScale = pixelMargin/float(extentScale)
    
    extentMinX = -pixelMarginScale
    extentMaxX = (xRange+1)/float(extentScale) - pixelMarginScale
    extentMinY = -pixelMarginScale
    extentMaxY = (yRange+1)/float(extentScale) - pixelMarginScale
    cropExtent = (extentMinX, extentMaxX, extentMinY, extentMaxY)

##################################################
# IMSHOW 
##################################################
cax2 = ax2.imshow(dataCrop, cmap=cmap,
                vmin=fs[plotField]['imshowMin'], vmax=fs[plotField]['imshowMax'],
                extent=cropExtent,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax2.set_xlabel(r'$x/\xi_0$',fontsize=font_size, labelpad=-5)
ax2.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
ax2.tick_params( axis='both', which='major', labelsize=tick_size )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

props = dict(boxstyle='square', facecolor='white')#, alpha=0.5)
ax2.text(0.02, 0.96, imshowPlotLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2.transAxes,
            fontsize=font_size,
            bbox=props,
            )

# Some extra tick scaling is required when plotting quiver on top of imshow
if fs[plotField]['quiver']:
    xTicks = np.array([0,2,4,6,8,10])*coherenceSize/quiverDensity
    xTickLabels = [r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$']
    ax2.xaxis.set_ticks(xTicks)
    ax2.xaxis.set_ticklabels(xTickLabels)

    yTicks = np.array([0,1,2,3,4,5,6])*coherenceSize/quiverDensity
    yTickLabels = [r'$0$',r'$1$',r'$2$',r'$3$',r'$4$',r'$5$',r'$6$']
    ax2.yaxis.set_ticks(yTicks)
    ax2.yaxis.set_ticklabels(yTickLabels)

# Colorbar
cbar2 = fig2.colorbar(cax2, cax=ax_cbar2, orientation='horizontal', ticks=cbarTicks)
cbar2.set_ticks(cbarTicks)
cbar2.set_ticklabels(cbarLabels)
cbar2.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar2.ax.xaxis.set_ticks_position('top')
cbar2.ax.xaxis.set_label_position('top')

# Text label
cbar2.set_label(fs[plotField]['label'],
                #rotation=270,
                labelpad=fs[plotField]['labelPad'],
                fontsize=font_size)
# Smoother appearance
cbar2.solids.set_edgecolor("face")

# Save to file
fig2.savefig('%s/svg/qc_old_%s_fs%d_R%s_r%s_T%s.svg' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/qc_old_%s_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/qc_old_%s_fs%d_R%s_r%s_T%s.png' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

#======================================================================#
# CUTS
#======================================================================#
if fs[plotField]['plotCuts']:
    fig3, axGrid3 = plt.subplots(2,1, figsize=(fig2Width*inchesPerPt, 1.5*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)

    ax3a = axGrid3[0]
    ax3b = axGrid3[1]

    if plotField == 'delta':
        xCuts = [0, int(coherenceSize*0.5),int(coherenceSize*1.0),int(coherenceSize*2.0)]
        yCuts = [0,10,21,42]
    else:
        xCuts = [int(coherenceSize*0.5),int(coherenceSize*1.0),int(coherenceSize*2.0)]
        yCuts = [10,21,42]

    linestyles = ['-','--','-','--','-.','-.']
    colors = ['r','g','b','c','m','k']
    linewidths = [1,2,2,3,2,3]

    linestylesY = ['-','--','-.','--','-.','-.']
    colorsY = ['m','#0000c0','#00c0c0','#c0c000']
    linewidthsY = [2,2,2,2,2,3]

    # Ranges in lattice points
    xArange = np.arange(0,xRange+1)
    yArange = np.arange(0,yRange+1)

    # Ranges in coherence lengths
    xArangeCoherence = xArange / coherenceSize
    yArangeCoherence = yArange / coherenceSize

    # Test fit to phase crystal 
    # The different fermisurfaces have nodes at slightly different coordinates
    if fermiSurfaceLabel == 1:
        testOffset = 2
        testXMin = 525
        testXMax = 692
        testDecay = 20.0
    elif fermiSurfaceLabel == 2:
        testOffset = 0
        testXMin = 523
        testXMax = 692
        testDecay = 20.0
    elif fermiSurfaceLabel == 3:
        testOffset = 2
        testXMin = 525
        testXMax = 689
        testDecay = 16.0

    testPeriod = testXMax-testXMin

    def analyticPhase(inX, inY, inDecay, inPeriod, inAmplitude):
        waveVector = 2.0 * np.pi / inPeriod
        return inAmplitude * np.sin(waveVector * inX) * (1.0 + inY/inDecay) * np.exp(-inY/inDecay)


    cutMaxX = 0
    for cutIdx in range(0,len(xCuts)):
        xCut = xCuts[cutIdx]
        xCutLabel = r'$y/\xi_0=%2.1f$' % (xCut / coherenceSize)
        ax3a.plot(xArangeCoherence, dataCrop[xCut,:],
                  ls=linestyles[cutIdx],c=colors[cutIdx],
                  lw=linewidths[cutIdx],label=xCutLabel)
        if max(dataCrop[xCut,:]) > cutMaxX:
            cutMaxX = max(dataCrop[xCut,:])
    
    if plotField == 'phase':
        ax3a.plot(xArangeCoherence, analyticPhase(xArange-testOffset,0,1,testPeriod,cutMaxX),
                'k-',lw=1,label=r'$\mathrm{Fit}$')
              
    # Used to find zeros of phase
    #ax3a.plot(xArangeCoherence, np.zeros_like(xArange), 'k-')

    ax3a.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
    ax3a.set_ylabel(fs[plotField]['cutLabelX'], fontsize=font_size)
    ax3a.tick_params( axis='both', which='major', labelsize=tick_size )
    ax3a.xaxis.set_ticks_position('bottom')
    ax3a.yaxis.set_ticks_position('left')
    ax3a.set_xlim(0,max(xArangeCoherence))
    ax3a.set_ylim(fs[plotField]['cutMin'],fs[plotField]['cutMax'])
    leg3a = ax3a.legend(prop={'size':font_size},
                    loc=1,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)
    ax3a.text(0.02, 0.96, r'$\mathrm{(b)}$',
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax3a.transAxes,
                fontsize=font_size)

    cutMaxY = 0
    for cutIdx in range(0,len(yCuts)):
        yCut = yCuts[cutIdx]
        yCutLabel = r'$x/\xi_0=%2.1f$' % (yCut / coherenceSize)
        ax3b.plot(yArangeCoherence, dataCrop[:,yCut],
                  ls=linestylesY[cutIdx],c=colorsY[cutIdx],
                  lw=linewidthsY[cutIdx],label=yCutLabel)
        if max(dataCrop[:,yCut]) > cutMaxY:
            cutMaxY = min(0.5, max(dataCrop[:,yCut]))

    if plotField == 'phase':
        ax3b.plot(yArangeCoherence, analyticPhase(0.25,yArange,testDecay,1,cutMaxY),
                'k-',lw=1,label=r'$\mathrm{Fit}$')
              
    ax3b.set_xlabel(r'$y/\xi_0$',fontsize=font_size)
    ax3b.set_ylabel(fs[plotField]['cutLabelY'], fontsize=font_size)
    ax3b.tick_params( axis='both', which='major', labelsize=tick_size )
    ax3b.xaxis.set_ticks_position('bottom')
    ax3b.yaxis.set_ticks_position('left')
    ax3b.set_xlim(0,max(yArangeCoherence))
    ax3b.set_ylim(fs[plotField]['cutMinY'],fs[plotField]['cutMaxY'])
    leg3b = ax3b.legend(prop={'size':font_size},
                    loc=1,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)
    ax3b.text(0.02, 0.96, r'$\mathrm{(c)}$',
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax3b.transAxes,
                fontsize=font_size)
    # Attempt to remove margin whitespace around figure
    fig3.tight_layout()

    # Save to file
    fig3.savefig('%s/svg/qc_%s_cuts_sep_fs%d_R%s_r%s_T%s.svg' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
    fig3.savefig('%s/pdf/qc_%s_cuts_sep_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
    fig3.savefig('%s/png/qc_%s_cuts_sep_fs%d_R%s_r%s_T%s.png' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)


plt.show()