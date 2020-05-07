#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_rotated_current_cut.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a rotated current from quasiclassical data, with a cut specified by coordinates.
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
sys.path.insert(0, '../../../../tools/')
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

dataBasePath = '../../data'
figSavePath = '../figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = False
fermiSurfaceBdG = 2

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
aspect = 1.0#'auto'
interpolation = 'none' #bicubic, bilinear
cmap = mpl.cm.get_cmap('myred')#plt.cm.get_cmap("RdBu")#
cmap.set_bad(badColor,1.0)

plotFullFigure = False

if plotFullFigure:
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

if fermiSurfaceBdG == 2:
    bottomMargin = 0.05
else:
    bottomMargin = 0.05

fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 1.2*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.09,
                     bottom=bottomMargin,
                     right=0.97,
                     top=0.91,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar2 = plt.subplot(gs[0:5,:])
ax2 = plt.subplot(gs[5:,:])


#======================================================================#
# DATA
#======================================================================#
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)

extent = (data.node.xMin, data.node.xMax, data.node.yMin, data.node.yMax)
currentAbs = data.node.current.abs
currentX = data.node.current.x
currentY = data.node.current.y
currentMin = 0.0#np.nanmin(currentAbs)
currentMax = np.nanmax(currentAbs)
currentAbs = rotate(currentAbs, angle=45)
currentX = rotate(currentX, angle=45)
currentY = rotate(currentY, angle=45)
currentAbs[currentAbs == 0] = np.nan

print('Min: %s, Max: %s' % (currentMin, currentMax))

# Indices for cut
if fermiSurfaceBdG == 2:
    xMin = 100
    xMax = 136
    yMin = 85
    yMax = yMin+int(30.0/np.sqrt(2))
    plotLabel = r'$\mathrm{(a)}\, \#1$'
else:
    xMin = 99
    xMax = 135
    yMin = 85
    yMax = yMin+int(30.0/np.sqrt(2))
    plotLabel = r'$\mathrm{(a)}\, \#1$'

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
    cax = ax.imshow(currentAbs, cmap=cmap,
                    vmin=currentMin, vmax=currentMax,
                    #extent=extent,
                    interpolation=interpolation,
                    aspect=aspect,
                    origin='lower')

    # Draw cut portion
    ax.plot([xMin,xMax],[yMin,yMin],'r-',lw=0.5)
    ax.plot([xMin,xMax],[yMax,yMax],'r-',lw=0.5)
    ax.plot([xMin,xMin],[yMin,yMax],'r-',lw=0.5)
    ax.plot([xMax,xMax],[yMin,yMax],'r-',lw=0.5)

    # Axis ticks and labels
    #ax.set_xticks(tickLabels)
    #ax.set_yticks(tickLabels)
    ax.set_xlabel(r'$x/a$',fontsize=font_size)
    ax.set_ylabel(r'$y/a$',fontsize=font_size)
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
currentCut = currentAbs[yMin:yMax+1,xMin:xMax+1]
rotationAngle = -(np.pi*0.25)
cosAngle = np.cos(rotationAngle)
sinAngle = np.sin(rotationAngle)
currentCutX = currentX[yMin:yMax+1,xMin:xMax+1] * cosAngle - currentY[yMin:yMax+1,xMin:xMax+1] * sinAngle
currentCutY = currentX[yMin:yMax+1,xMin:xMax+1] * sinAngle + currentY[yMin:yMax+1,xMin:xMax+1] * cosAngle

quiverDensity   = 2  #[!!!]
quiverScale      = 50
quiverWidth      = 0.004#0.003#0.0015
quiverHeadWidth  = 5.0#7.0
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
quiverAx = ax2.quiver(quiverX/currentMax,
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
# IMSHOW PLOT: FIG 2
##################################################

cax2 = ax2.imshow(currentCut, cmap=cmap,
                vmin=currentMin, vmax=currentMax,
                extent=extentQuiver,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax2.set_xlabel(r'$x/a$',fontsize=font_size,labelpad=-5)
ax2.set_ylabel(r'$y/a$',fontsize=font_size)
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

"""ticks_x = ticker.FuncFormatter(lambda x, pos: r'${0:g}$'.format(x*quiverDensity))#+xMin))
ax2.xaxis.set_major_formatter(ticks_x)
if fermiSurfaceBdG == 4:
    ax2.xaxis.set_ticks([0.5,3.0,5.5,8.0,10.5,13,15.5,18])

ticks_y = ticker.FuncFormatter(lambda y, pos: r'${0:g}$'.format(y*quiverDensity))
ax2.yaxis.set_major_formatter(ticks_y)"""

xTicks = np.array([0,10,20,30,40,50])/(quiverDensity*np.sqrt(2))
xTickLabels = [r'$0$',r'$10$',r'$20$',r'$30$',r'$40$',r'$50$']
ax2.xaxis.set_ticks(xTicks)
ax2.xaxis.set_ticklabels(xTickLabels)

yTicks = np.array([0,10,20,30])/(quiverDensity*np.sqrt(2))
yTickLabels = [r'$0$',r'$10$',r'$20$',r'$30$']
ax2.yaxis.set_ticks(yTicks)
ax2.yaxis.set_ticklabels(yTickLabels)

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
                labelpad=-10,
                fontsize=font_size)
# Smoother appearance
cbar2.solids.set_edgecolor("face")


##################################################
# CUTS
##################################################
"""
f3, ax3 = plt.subplots()

iOffests = [0,1,2,5,10,15,-1]
colors = ['r','g','b','c','k','m','y']
for i in range(0, len(iOffests)):
    ax3.plot(currentCut[i,:],c=colors[i])
"""
##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.savefig('%s/svg/bdg_current_sep_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_current_sep_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_current_sep_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()