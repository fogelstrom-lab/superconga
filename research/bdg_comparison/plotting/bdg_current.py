#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_current.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot 2D BdG current with imshow for comparison-figure.
"""
#======================================================================#
# IMPORTS
#======================================================================#
from __future__ import division

import sys
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
# Import user-defined colormaps and register them in the matplotlib pipeline
from pythonLibraries.plotting import cmapsExtra as cm
cm.reg_cmaps()

from utils_bdg_comparison import *

#======================================================================#
# SYSTEMS AND FILES
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
updateNumpyDataFile = False
fermiSurfaceLabels = [2, 4]

#======================================================================#
# PLOTTING PARAMETERS
#======================================================================#
goldenRatio = (np.sqrt(5.0)-1.0)/2.0
inchesPerPt = 1.0/72.27
fig2Width = 1078.0
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

# Current colormap: use one of Mikaels colormaps
cmap = mpl.cm.get_cmap('myred')
cmap.set_bad(badColor,1.0)

fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 2.1*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.09,
                     bottom=0.05,
                     right=0.97,
                     top=0.95,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(105,105)

ax_cbar = plt.subplot(gs[0:2,:])
ax = [  plt.subplot(gs[5:55,:]),
        plt.subplot(gs[55:,:]) ]
cax = {}
cbar = {}

#======================================================================#
# DATA
#======================================================================#
for fs_i in range(0, len(fermiSurfaceLabels)):
    fermiSurfaceBdG = fermiSurfaceLabels[fs_i]
    data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)

    extent = (data.node.xMin, data.node.xMax, data.node.yMin, data.node.yMax)
    currentAbs = data.node.current.abs
    currentX = data.node.current.x
    currentY = data.node.current.y
    currentMin = 0.0#np.nanmin(currentAbs)
    currentMax = 0.18#np.nanmax(currentAbs)
    currentAbs = rotate(currentAbs, angle=45)
    currentX = rotate(currentX, angle=45)
    currentY = rotate(currentY, angle=45)
    currentAbs[currentAbs == 0] = np.nan

    print('Min: %s, Max: %s' % (np.nanmin(currentAbs), np.nanmax(currentAbs)))

    # Indices for cut
    if fermiSurfaceBdG == 2:
        xMin = 100
        xMax = 136
        yMin = 85
        yMax = yMin+int(30.0/np.sqrt(2))
        plotLabel = r'$\mathrm{(a)}\, \#1, \mathrm{(BdG)}$'
    else:
        xMin = 99
        xMax = 135
        yMin = 85
        yMax = yMin+int(30.0/np.sqrt(2))
        plotLabel = r'$\mathrm{(b)}\, \#2, \mathrm{(BdG)}$'

    cbarTicks = [currentMin, currentMax]

    if np.allclose(0.0, currentMin):
        currentMin_str = r'$0$'
    else:
        currentMin_str = r'$%2.2f$' % currentMin
        
    cbarLabels = [currentMin_str, r'$%2.2f$' % currentMax]
    cbarLabel = r'$\left|\mathbf{j}\right|/j_{\mathrm{d}}$'

    ##################################################
    # QUIVER PLOT
    ##################################################
    currentCut = currentAbs[yMin:yMax+1,xMin:xMax+1]
    rotationAngle = -(np.pi*0.25)
    cosAngle = np.cos(rotationAngle)
    sinAngle = np.sin(rotationAngle)
    currentCutX = currentX[yMin:yMax+1,xMin:xMax+1] * cosAngle - currentY[yMin:yMax+1,xMin:xMax+1] * sinAngle
    currentCutY = currentX[yMin:yMax+1,xMin:xMax+1] * sinAngle + currentY[yMin:yMax+1,xMin:xMax+1] * cosAngle

    quiverDensity   = 2
    quiverScale      = 50
    quiverWidth      = 0.004
    quiverHeadWidth  = 5.0
    quiverHeadLength = 5.0
    quiverColor = 'w'
    quiverEdgeColor = 'k'
    quiverEdgeLineWidth = 0.5
    quiverPivot = 'middle'
    quiverDataScale = 0.2

    yNum, xNum = np.shape(currentCutX)

    # Center ticks on pixel
    pixelMargin = 0.5
    pixelMarginScale = pixelMargin/float(quiverDensity)

    # Calculate extent to make quiver overlap with imshow correctly
    extentMinX = -pixelMarginScale#0.5#-0.5
    extentMaxX = xNum/float(quiverDensity) - pixelMarginScale#-0.5
    extentMinY = -pixelMarginScale#.5#-0.5
    extentMaxY = yNum/float(quiverDensity) - pixelMarginScale#-0.5
    extentQuiver = (extentMinX, extentMaxX, extentMinY, extentMaxY)

    # Select and scale data for quiver 
    quiverX = quiverDataScale*currentCutX[::quiverDensity, ::quiverDensity]
    quiverY = quiverDataScale*currentCutY[::quiverDensity, ::quiverDensity]

    # Plot quiver
    quiverAx = ax[fs_i].quiver(quiverX/currentMax,
                        quiverY/currentMax,
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
    # IMSHOW PLOT
    ##################################################

    cax[fs_i] = ax[fs_i].imshow(currentCut, cmap=cmap,
                    vmin=currentMin, vmax=currentMax,
                    extent=extentQuiver,
                    interpolation=interpolation,
                    aspect=aspect,
                    origin='lower')

    # Axis ticks and labels
    ax[fs_i].set_ylabel(r'$y/a$',fontsize=font_size)
    ax[fs_i].tick_params( axis='both', which='major', labelsize=tick_size )
    ax[fs_i].xaxis.set_ticks_position('bottom')
    ax[fs_i].yaxis.set_ticks_position('left')

    props = dict(boxstyle='square', facecolor='white')#, alpha=0.5)
    ax[fs_i].text(0.02, 0.96, plotLabel,
                horizontalalignment='left',
                verticalalignment='top',
                transform = ax[fs_i].transAxes,
                fontsize=font_size,
                bbox=props,
                )

    xTicks = np.array([0,10,20,30,40,50])/(quiverDensity*np.sqrt(2))
    xTickLabels = [r'$0$',r'$10$',r'$20$',r'$30$',r'$40$',r'$50$']
    
    if fs_i == 1:
        ax[fs_i].xaxis.set_ticks(xTicks)
        ax[fs_i].xaxis.set_ticklabels(xTickLabels)
        ax[fs_i].set_xlabel(r'$x/a$',fontsize=font_size,labelpad=-5)
    else:
        ax[fs_i].xaxis.set_ticks(xTicks)
        ax[fs_i].xaxis.set_ticklabels(len(xTickLabels)*[''])

    yTicks = np.array([0,10,20,30])/(quiverDensity*np.sqrt(2))
    yTickLabels = [r'$0$',r'$10$',r'$20$',r'$30$']
    ax[fs_i].yaxis.set_ticks(yTicks)
    ax[fs_i].yaxis.set_ticklabels(yTickLabels)

##################################################
# COLOR BAR: FIG 2
##################################################

cbar = fig2.colorbar(cax[0], cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
cbar.set_ticks(cbarTicks)
cbar.set_ticklabels(cbarLabels)
cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Text label
cbar.set_label(cbarLabel,
                #rotation=270,
                labelpad=-10,
                fontsize=font_size)
# Smoother appearance
cbar.solids.set_edgecolor("face")

##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.savefig('%s/svg/bdg_currents.svg' % (figSavePath),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_currents.pdf' % (figSavePath),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_currents.png' % (figSavePath),format='png',dpi=300,transparent=False)

plt.show()