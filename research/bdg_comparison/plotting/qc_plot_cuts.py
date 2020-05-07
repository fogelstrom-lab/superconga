#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_plot_cuts.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot quasiclassical data, with a cut specified by coordinates.
              Toggle between plotting OP amplitude and phase with plotField,
              and Fermi surface by changing fermiSurfaceLabel. These plots are
              made into a collage with inkscape, i.e. Fig. 6 in the paper.
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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
sys.path.insert(0, '../../../tools/')

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
plotLegend = True
# Fermisurfaces:
# 1: circular
# 2: FS1
# 3: FS2

# Possible fields to plot are: ['current', 'phase', 'delta', 'ps']
plotField = 'delta'

# Field-specific settings
# centerData: center data range around 0 for phase, else from 0 to dataMax
fs = {
'delta': {  'centerData': False,
            'colormap': plt.cm.get_cmap('jet'),
            'label': r'$\Delta_d/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'labelPad': -10,
            'imshowMin': 0,
            'imshowMax': 1.5,
            'cutLabelX': r'$|\Delta_d(x)|/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'cutLabelY': r'$|\Delta_d(y)|/k_{\mathrm{B}}T_{\mathrm{c}}$',
            'labelPadInY': -1,
            'labelPadMainY': -17,
            'quiver': False,
            },
'phase': {  'centerData': True, 
            'colormap': plt.cm.get_cmap('RdBu_r'),
            'label': r'$\chi/\pi$',
            'labelPad': 0,
            'imshowMin': -0.5,
            'imshowMax': 0.5,
            'cutLabelX': r'$\chi(x)/\pi$',
            'cutLabelY': r'$\chi(y)/\pi$',
            'labelPadInY': -14,
            'labelPadMainY': -17,
            'quiver': False,
            },
}

if not fs.has_key(plotField):
    print('Field %s is not implemented!' % (plotField))
    exit()

if not fermiSurfaceLabel in [1,2,3]:
    print('Unknown Fermi surface label %d !' % (fermiSurfaceLabel))
    exit()

if plotField == 'delta':
    xCutMin = 0
    xCutMax = 1.5
    yCutMin = 0
    yCutMax = 1.5
    xCutTicks = [0, 0.5, 1, 1.5]
    xCutTickLabels = [r'$0$', r'', r'$1$', r'']
    yCutTicks = [0, 0.5, 1, 1.5]
    yCutTickLabels = [r'$0$', r'', r'', r'$1.5$']

    if fermiSurfaceLabel == 2:
        cutLabel = r'$\mathrm{(c)}\, \#1$'
    else:
        cutLabel = r'$\mathrm{(d)}\, \#2$'
else:
    if fermiSurfaceLabel == 2:
        cutLabel = r'$\mathrm{(e)}\, \#1$'
        xCutMin = -0.45
        xCutMax = 0.45
        yCutMin = 0
        yCutMax = 0.45
        xCutTicks = [-0.4, 0, 0.4]
        xCutTickLabels = [r'$-0.4$', r'$0$', r'$0.4$']
        yCutTicks = [0, 0.2, 0.4]
        yCutTickLabels = [r'$0$', r'', r'$0.4$']
    else:
        cutLabel = r'$\mathrm{(f)}\, \#2$'
        xCutMin = -0.55
        xCutMax = 0.55
        yCutMin = 0
        yCutMax = 0.55
        xCutTicks = [-0.5, 0, 0.5]
        xCutTickLabels = [r'$-0.5$', r'$0$', r'$0.5$']
        yCutTicks = [0, 0.25, 0.5]
        yCutTickLabels = [r'$0$', r'', r'$0.5$']

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

# Cuts figure
fig, axMain = plt.subplots(1,1,
                           figsize=(fig2Width*inchesPerPt,
                                    0.8*fig2Width*inchesPerPt*goldenRatio),
                           dpi=dpi)
fig.subplots_adjust(left=0.07,
                    bottom=0.11,
                    right=0.99,
                    top=0.97,
                    wspace=0,
                    hspace=0)

# Set up inset plot
if plotField == 'delta':
    axIn = inset_axes(axMain, width="100%", height="100%", loc=3,
                      bbox_to_anchor=(.385, .145, .6, .55),
                      bbox_transform=axMain.transAxes)
else:
    axIn = inset_axes(axMain, width="60%", height="50%", loc=1)


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

# Rotate field
dataField[dataField != dataField] = 0
dataField = rotate(dataField, angle=45)
# Try to set nan outside grain, but does not work fully for rotated fields
#dataField[dataField == 0] = np.nan

# Create cropped data
dataCrop = dataField[yMin:yMax+1,xMin:xMax+1]

#======================================================================#
# CUTS
#======================================================================#
xCuts = [0, int(coherenceSize*0.5),int(coherenceSize*1.0),int(coherenceSize*2.0)]
yCuts = [10,21,42]


matteColors = ['#43ae5f','#ea4234','#4184f4']
brownOrangeYellow = ['#51280c','#945044','#ff5100','#feae01']
brownOrangeYellow2 = ['#945044','#ff5100','#feae01']


linestyles = ['-','--','-','--','-.','-.']
colorsX = brownOrangeYellow
lineWidthX = 1.5
lineStyleX = '-'

linestylesY = ['-','--','-.','--','-.','-.']
colorsY = matteColors
lineWidthY = 1.5
lineStyleY = '--'

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
    axIn.plot(xArangeCoherence, dataCrop[xCut,:],
                ls=lineStyleX,c=colorsX[cutIdx],
                lw=lineWidthX,label=xCutLabel)
    if max(dataCrop[xCut,:]) > cutMaxX:
        cutMaxX = max(dataCrop[xCut,:])

if fermiSurfaceLabel == 2:
    maxPhaseFitAmp = cutMaxX
else:
    maxPhaseFitAmp = 0.95*cutMaxX

if plotField == 'phase':
    axIn.plot(xArangeCoherence, analyticPhase(xArange-testOffset,0,1,testPeriod,maxPhaseFitAmp),
            'k-.',lw=1.6,label=r'$\mathrm{Fit}$')
            
# Used to find zeros of phase
#axIn.plot(xArangeCoherence, np.zeros_like(xArange), 'k-')

axIn.set_xlabel(r'$x/\xi_0$',fontsize=font_size,labelpad=-10)
axIn.set_ylabel(fs[plotField]['cutLabelX'], fontsize=font_size,labelpad=fs[plotField]['labelPadInY'])
axIn.tick_params( axis='both', which='major', labelsize=tick_size )
axIn.xaxis.set_ticks_position('bottom')
axIn.yaxis.set_ticks_position('left')
axIn.set_xlim(0,max(xArangeCoherence))
axIn.set_ylim(xCutMin, xCutMax)
axIn.set_xticks([0, 2, 4, 6, 8, 10])
axIn.set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$',r'$8$',r'$10$'])
axIn.set_yticks(xCutTicks)
axIn.set_yticklabels(xCutTickLabels)
if plotLegend:
    legIn = axIn.legend(prop={'size':font_size},
                    loc=1,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)

cutMaxY = 0
for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/\xi_0=%2.1f$' % (yCut / coherenceSize)
    axMain.plot(yArangeCoherence, dataCrop[:,yCut],
                ls=lineStyleY,c=colorsY[cutIdx],
                lw=lineWidthY,label=yCutLabel)
    if max(dataCrop[:,yCut]) > cutMaxY:
        cutMaxY = min(0.5, max(dataCrop[:,yCut]))

if plotField == 'phase':
    axMain.plot(yArangeCoherence, analyticPhase(0.25,yArange,testDecay,1,cutMaxY),
            'k-.',lw=1.6,label=r'$\mathrm{Fit}$')

axMain.set_xlabel(r'$y/\xi_0$',fontsize=font_size,labelpad=-14)
axMain.set_ylabel(fs[plotField]['cutLabelY'], fontsize=font_size,labelpad=fs[plotField]['labelPadMainY'])
axMain.tick_params( axis='both', which='major', labelsize=tick_size )
axMain.xaxis.set_ticks_position('bottom')
axMain.yaxis.set_ticks_position('left')
axMain.set_xlim(0,max(yArangeCoherence))
axMain.set_ylim(yCutMin, yCutMax)
axMain.set_xticks([0, 2, 4, 6])
axMain.set_xticklabels([r'$0$',r'$2$',r'$4$',r'$6$'])
axMain.set_yticks(yCutTicks)
axMain.set_yticklabels(yCutTickLabels)
if plotLegend:
    legMain = axMain.legend(prop={'size':font_size},
                    loc=1,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)
axMain.text(0.02, 0.96, cutLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = axMain.transAxes,
            fontsize=font_size)
# Attempt to remove margin whitespace around figure

# Save to file
fig.savefig('%s/svg/qc_%s_cuts_fs%d_R%s_r%s_T%s.svg' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/qc_%s_cuts_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/qc_%s_cuts_fs%d_R%s_r%s_T%s.png' % (figSavePath, plotField, fermiSurfaceLabel-1, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)


plt.show()