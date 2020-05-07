#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_phase_2d_inset.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot phase in 2D with imshow, to be used as inset in Fig. 3.
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


#======================================================================#
# IMPORTS - CUSTOM LIBRARIES
#======================================================================#
sys.path.insert(0, '../../../tools/')
from pythonLibraries.plotting import cmapsExtra as cm

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

# Color outside grain
whiteColor = '#ffffff'
grayColor = '#c0c0c0'
badColor = whiteColor

# Examples of interpolations: bicubic, bilinear, none. None is more accurate now
interpolation = 'none'

# Aspect ratio for imshow: 1 or auto. 1 means that a square is not distorted.
aspect = 1.0


# Imshow figure
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
cmap = plt.cm.get_cmap("RdBu_r")
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW PLOT
##################################################
# Indices for cut
xMin = 100
xMax = 153
yMin = 1
yMax = 31

xRange = xMax-xMin
yRange = yMax-yMin
xArange = np.arange(xMin,xMax+1)

# Fetch data
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)
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

# Plot phase
cax = ax.imshow(dataField, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                extent=(xMin-dpt,xMax+dpt,-dpt,yRange+dpt),
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
    
cbarLabels = [r'$%2.2f$' % dataMin, '', r'$%2.2f$' % dataMax]
cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
cbar.set_ticks(cbarTicks)
cbar.set_ticklabels(cbarLabels)
cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Text label
cbarLabel = r'$\chi / \pi$'
cbar.set_label(cbarLabel,
                #rotation=270,
                labelpad=-10,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")


##################################################
# SAVE PLOTS TO FILE
##################################################
fig.savefig('%s/svg/bdg_phase_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/bdg_phase_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/bdg_phase_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()