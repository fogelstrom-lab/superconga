#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_op_amplitude.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot BdG OP amplitude cuts for Fig. 3.
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
# Import user-defined colormaps and register them in the matplotlib pipeline
from pythonLibraries.plotting import cmapsExtra as cm
cm.reg_cmaps()

from utils_bdg_comparison import *

#======================================================================#
# PLOTTING PARAMETERS
#======================================================================#
goldenRatio = (np.sqrt(5.0)-1.0)/2.0
inchesPerPt = 1.0/72.27
fig2Width = 512.0
# see: https://www.overleaf.com/learn/latex/Lengths_in_LaTeX

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
whiteColor = '#ffffff'
grayColor = '#c0c0c0'
badColor = whiteColor

# Examples of interpolations: bicubic, bilinear, none. None is more accurate now
interpolation = 'none'

# Aspect ratio for imshow: 1 or auto. 1 means that a square is not distorted.
aspect = 1.0


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
ax2in = inset_axes(ax2main, width="100%", height="100%", loc=3,
                   bbox_to_anchor=(.385, .145, .6, .4),
                   bbox_transform=ax2main.transAxes)

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

fermiSurfaceIndices = [2,4]
# Read raw data even if numpy data file exists
updateNumpyDataFile = True
fermiSurfaceBdG = 2
cmap = plt.cm.get_cmap("jet")#mpl.cm.get_cmap('myred')#
cmap.set_bad(badColor,1.0)
plotLegend = False

##################################################
# DATA
##################################################
# Indices for cut
if fermiSurfaceBdG == 2:
    xMin = 100
    xMax = 153
    mainLabel = r'$\mathrm{(c)}\, \#1$'
    insetLabelOffset = 0.5
else:
    xMin = 100
    xMax = 153
    mainLabel = r'$\mathrm{(d)}\, \#2$'
    insetLabelOffset = 0.3

yMin = 1
yMax = 31

xRange = xMax-xMin
yRange = yMax-yMin
xArange = np.arange(xMin,xMax+1)

# Fetch data
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)
#extent = (data.linkRot.xMin, data.linkRot.xMax, data.linkRot.yMin, data.linkRot.yMax)
dataField = data.linkRot.orderParameter.abs[yMin:yMax+1,xMin:xMax+1]
dataMax = np.nanmax(dataField)
dataMin = np.nanmin(dataField)
dataField[dataField == 0] = np.nan

print('Min: %s, Max: %s' % (dataMin, dataMax))


##################################################
# PLOT CUTS
##################################################
orangeToBrown = ['#ffbf80','#e67300','#ff3300','#4d2600']
brownToOrange = ['#ffbf80','#e67300','#ff3300','#4d2600']
bluesToPurple = ['#002699','#3366ff','#9933ff','#99b3ff']
bluesNice = ['#00429d','#4771b2','#8abccf','#c5eddf']
bluesNice_r = ['#c5eddf','#8abccf','#4771b2','#00429d']
blueToRed = ['#00429d', '#73a2c6', '#f4777f', '#93003a']
redToBlue = blueToRed[::-1]

redBrownOrange = ['#804000','#f5ad18', '#d400aa', '#aad400']
matteColors = ['#43ae5f','#ea4234','#4184f4']

brownOrangeYellow = ['#51280c','#945044','#ff5100','#feae01']

# Plot parallel to interface
xCuts = [0,5,10,20]
colorsX = brownOrangeYellow
lineWidthX = 1.5
lineStyleX = '-'

# Plot perpendicular to interface
yCuts = [14,20,26]
colorsY = matteColors
lineWidthY = 1.5
lineStyleY = '--'

colorNonPB = '#7137c8'#'#0366FC'


# Origin for non-pairbreaking edge plotting
if fermiSurfaceBdG == 2:
    parallelX0 = 20
else: 
    parallelX0 = 21
parallelY0 = 38
plotParallelNonPB = False

# Plot parallel to pairbreaking edge: fixed y
for cutIdx in range(0,len(xCuts)):
    xCut = xCuts[cutIdx]
    xCutLabel = r'$y/a=%d$' % xCut
    ax2in.plot(xArange, dataField[xCut,0:],
              ls=lineStyleX,c=colorsX[cutIdx],
              lw=lineWidthX,label=xCutLabel)

    # Plot parallel to non-pairbreaking edge
    if plotParallelNonPB:
        dataNonPB = []
        for diagIdx in range(0, xRange+1):
            dataNonPB.append(data.linkRot.orderParameter.abs[parallelY0+diagIdx-xCut,
                                                             parallelX0+diagIdx])
        if cutIdx == 0:
            labelParallel = r'$\mathrm{Non-PB\, edge}$'
        else:
            labelParallel = None
        ax2in.plot(xArange, dataNonPB,
                ls='-.',c=colors[cutIdx],
                lw=2,label=labelParallel)

ax2in.set_xlabel(r'$x/a$',fontsize=font_size,labelpad=-11)
ax2in.set_ylabel(r'$|\Delta_d(x)|/t$',fontsize=font_size)
ax2in.tick_params( axis='both', which='major', labelsize=tick_size )
ax2in.xaxis.set_ticks_position('bottom')
ax2in.yaxis.set_ticks_position('left')
ax2in.set_xlim(xMin,xMax)
ax2in.set_ylim(0.0,0.08)
ax2in.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
ax2in.set_yticklabels([r'$0$', r'', r'$0.04$', r'', r'$0.08$'])
if plotLegend:
    leg2a = ax2in.legend(prop={'size':font_size},
                    loc=2,
                    ncol=2,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)

# Plot perpendicular to pairbreaking edge: fixed x
for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/a=%d$' % (yCut+xMin)
    ax2main.plot(dataField[:,yCut],
              ls=lineStyleY,c=colorsY[cutIdx],
              lw=lineWidthY,label=yCutLabel)

# Origin for non-pairbreaking edge plotting
if fermiSurfaceBdG == 2:
    parallelX0 = 100
else: 
    parallelX0 = 101
parallelY0 = 118

# Plot perpendicular against non-pairbreaking edge
perpDataNonPB = []
for diagIdx in range(0, yRange+2):
    perpDataNonPB.append(data.linkRot.orderParameter.abs[parallelY0-diagIdx,
                                                         parallelX0+diagIdx])
ax2main.plot(perpDataNonPB,ls='-.',lw=2,c=colorNonPB,label=r'$\mathrm{Non-PB\, edge}$')

ax2main.set_xlabel(r'$y/a$',fontsize=font_size, labelpad=-3)
ax2main.set_ylabel(r'$|\Delta_d(y)|/t$',fontsize=font_size, labelpad=-20)
ax2main.tick_params( axis='both', which='major', labelsize=tick_size )
ax2main.xaxis.set_ticks_position('bottom')
ax2main.yaxis.set_ticks_position('left')
ax2main.set_xlim(0,yRange)
ax2main.set_ylim(0,0.10)
ax2main.set_yticks([0,0.05,0.10])
ax2main.set_yticklabels([r'$0$', r'', r'$0.1$'])
#ax2main.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
#ax2main.set_yticklabels([r'$0$', r'$0.02$', r'$0.04$', r'$0.06$', r'$0.08$'])
if plotLegend:
    leg2b = ax2main.legend(prop={'size':font_size},
                    loc=1,
                    ncol=1,
                    borderpad=0.5,
                    labelspacing=0.2,
                    borderaxespad=1.1,
                    fancybox=True,
                    #framealpha=1.0,
                    )#, handlelength=1.0)
ax2main.text(0.02, 0.96, mainLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2main.transAxes,
            fontsize=font_size)
##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.tight_layout()
fig2.savefig('%s/svg/bdg_order_parameter_cuts_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_order_parameter_cuts_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_order_parameter_cuts_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()