#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_phase_cuts.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot BdG phase cuts for Fig. 3.
"""
#======================================================================#
# IMPORTS
#======================================================================#
from __future__ import division

import sys
import colorsys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp2d
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
sys.path.insert(0, '../../../tools/')

from utils_bdg_comparison import *

#======================================================================#
# PLOTTING PARAMETERS
#======================================================================#
goldenRatio = (np.sqrt(5.0)-1.0)/2.0
inchesPerPt = 1.0/72.27
fig2Width = 512.0

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

# Cuts figure
fig2, ax2main = plt.subplots(1,1, figsize=(fig2Width*inchesPerPt, 0.8*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)

"""
fig2.subplots_adjust(left=0.13,
                     bottom=0.15,
                     right=0.98,
                     top=0.96,
                     wspace=0,
                     hspace=0)
"""
ax2in = inset_axes(ax2main, width="60%", height="50%", loc=1)

#======================================================================#
# FIELDS AND PARAMETERS
#======================================================================#

# BdG data files:
# ../data/bdg/bs2/Curr.dat
# ../data/bdg/bs2/OP.dat
# ../data/bdg/bs4/Curr.dat
# ../data/bdg/bs4/OP.dat
# QC data files:
# ../data/qc/fs1_R40.000_r4.000_T0.1000

dataBasePath = '../data'
figSavePath = 'figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = True
fermiSurfaceBdG = 2
plotLegend = False

##################################################
# DATA
##################################################
# Indices for cut
if fermiSurfaceBdG == 2:
    xMin = 100
    xMax = 153
    xCutTicks = [-0.5, -0.25, 0, 0.25, 0.5]
    xCutTickLabels = [r'$-0.50$', '', r'$0$', '', r'$0.50$']
    yCutMin = 0
    yCutMax = 0.5
    yCutTicks = [0, 0.25, 0.5]
    yCutTickLabels = [r'$0$', '', r'$0.5$']
    cutLabel = r'$\mathrm{(e)}\, \#1$'
    mainLabelPadY = -20
else:
    xMin = 100
    xMax = 153
    xCutTicks = [-0.15, 0, 0.15]
    xCutTickLabels = [r'$-0.15$', r'$0$', r'$0.15$']
    yCutMin = 0
    yCutMax = 0.25
    yCutTicks = [0, 0.1, 0.2]
    yCutTickLabels = [r'$0$', '', r'$0.2$']
    cutLabel = r'$\mathrm{(f)}\, \#2$'
    mainLabelPadY = -20

yMin = 1
yMax = 31

xRange = xMax-xMin
yRange = yMax-yMin
xArange = np.arange(xMin,xMax+1)

# Fetch data
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)
#extent = (data.linkRot.xMin, data.linkRot.xMax, data.linkRot.yMin, data.linkRot.yMax)
dataField = data.linkRot.orderParameter.phase[yMin:yMax+1,xMin:xMax+1] / np.pi
dataMaxUncentered = np.nanmax(dataField)
dataMinUncentered = np.nanmin(dataField)
dataField[dataField == 0] = np.nan

# Center phase around 0
dataRange = abs(dataMaxUncentered - dataMinUncentered)
dataMin = -0.5*dataRange
dataMax = 0.5*dataRange
dataField = (dataField - dataMinUncentered) + dataMin

print('Min: %s, Max: %s' % (dataMin, dataMax))

# Center scale on pixels
dpt = 0.5


##################################################
# TEST PLOTTING CUTS
##################################################
orangeToBrown = ['#ffbf80','#e67300','#ff3300','#4d2600']
brownToOrange = ['#ffbf80','#e67300','#ff3300','#4d2600']
bluesToPurple = ['#002699','#3366ff','#9933ff','#99b3ff']
bluesNice = ['#00429d','#4771b2','#8abccf','#c5eddf']
bluesNice_r = ['#c5eddf','#8abccf','#4771b2','#00429d']
blueToRed = ['#00429d', '#73a2c6', '#f4777f', '#93003a']
redToBlue = blueToRed[::-1]

redBrownOrange = ['#804000','#FF0000', '#cc9966', '#aad400']
matteColors = ['#43ae5f','#ea4234','#4184f4','#b84dff']

colorsX = redBrownOrange
lineWidthX = 1.5
lineStyleX = '-'

colorsY = matteColors
lineWidthY = 1.5
lineStyleY = '--'

xCuts = [0,5,10,20]
yCuts = [14,20,26]
yTest = np.arange(0,yRange+1)

if fermiSurfaceBdG == 2:
    testDecay = 6.4
    testPeriod = xRange
    xTest = np.arange(0,xRange+1)
else: 
    testDecay = 4.6
    testPeriod = xRange -2.5#1.5
    xTest = np.arange(0,xRange+1)-2.0
    

def analyticPhase(inX, inY, inDecay, inPeriod, inAmplitude):
    waveVector = 2.0 * np.pi / inPeriod
    return inAmplitude * np.sin(waveVector * inX) * (1.0 + inY/inDecay) * np.exp(-inY/inDecay)

maxPhase = 0.0
for cutIdx in range(0,len(xCuts)):
    xCut = xCuts[cutIdx]
    xCutLabel = r'$y/a=%d$' % xCut
    ax2in.plot(xArange, dataField[xCut,:],
              ls=lineStyleX,c=colorsX[cutIdx],
              lw=lineWidthX,label=xCutLabel)
    if np.max(dataField[xCut,:]) > maxPhase:
        maxPhase = np.max(dataField[xCut,:])

# Plot analytic phase crystal (manual fit)
if fermiSurfaceBdG == 2:
    maxPhaseFit_a = maxPhase
    maxPhaseFit_b = 0.22#0.9*maxPhase
else:
    maxPhaseFit_a = maxPhase
    maxPhaseFit_b = 0.95*dataMax

ax2in.plot(xArange, analyticPhase(xTest,0,1,testPeriod,maxPhaseFit_a),'k-.',lw=1.6,label=r'$\mathrm{Fit}$')
ax2main.plot(yTest, analyticPhase(1,yTest,testDecay,4.0,maxPhaseFit_b),'k-.',lw=1.6,label=r'$\mathrm{Fit}$')
#ax2in.plot(xArange, np.zeros_like(xArange))


ax2in.set_xlabel(r'$x/a$',fontsize=font_size,labelpad=-11)
ax2in.set_ylabel(r'$\chi(x)/\pi$',fontsize=font_size,labelpad=-15)
ax2in.tick_params( axis='both', which='major', labelsize=tick_size )
ax2in.xaxis.set_ticks_position('bottom')
ax2in.yaxis.set_ticks_position('left')
ax2in.set_xlim(xMin,xMax)
ax2in.set_ylim(xCutTicks[0],xCutTicks[-1])
ax2in.set_yticks(xCutTicks)
ax2in.set_yticklabels(xCutTickLabels)
if plotLegend:
    leg2a = ax2in.legend(prop={'size':font_size},
                    loc=1,
                    ncol=2,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)

for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/a=%d$' % (yCut+xMin)
    ax2main.plot(dataField[:,yCut],
              ls=lineStyleY,c=colorsY[cutIdx],
              lw=lineWidthY,label=yCutLabel)

ax2main.set_xlabel(r'$y/a$',fontsize=font_size,labelpad=-3)
ax2main.set_ylabel(r'$\chi(y)/\pi$',fontsize=font_size,labelpad=mainLabelPadY)
ax2main.tick_params( axis='both', which='major', labelsize=tick_size )
ax2main.xaxis.set_ticks_position('bottom')
ax2main.yaxis.set_ticks_position('left')
ax2main.set_xlim(0,yRange)
ax2main.set_ylim(yCutMin,yCutMax)
ax2main.set_yticks(yCutTicks)
ax2main.set_yticklabels(yCutTickLabels)
if plotLegend:
    leg2b = ax2main.legend(prop={'size':font_size},
                    loc=2,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)
ax2main.text(0.005, 0.99, cutLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2main.transAxes,
            fontsize=font_size)


##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.tight_layout()
fig2.savefig('%s/svg/bdg_phase_cuts_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_phase_cuts_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_phase_cuts_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()