#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_full_grain_op.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot full grain OP amplitude from BdG data, rotated so that
              the pair-breaking interface is horizontal. Used for Fig. 3.
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

interpolation = 'none' #bicubic, bilinear
fig = plt.figure(figsize=(fig2Width*inchesPerPt, 1.3*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig.subplots_adjust(left=0.11,
                     bottom=0.0,
                     right=0.96,
                     top=0.88,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar = plt.subplot(gs[0:5,:])
ax = plt.subplot(gs[5:,:])
#======================================================================#
# FIELDS AND PARAMETERS
#======================================================================#

# Read raw data even if numpy data file exists
updateNumpyDataFile = True

# Paths to data and figures
dataBasePath = '../data'
figSavePath = 'figures'

# BdG data files:
# ../data/bdg/bs2/Curr.dat
# ../data/bdg/bs2/OP.dat
# ../data/bdg/bs4/Curr.dat
# ../data/bdg/bs4/OP.dat
# QC data files:
# ../data/qc/fs1_R40.000_r4.000_T0.1000

# Set to 2 or 4: FS1 = BS2, FS2 = BS4
fermiSurfaceBdG = 4

if fermiSurfaceBdG == 2:
    figLabel = r'$\mathrm{(a)}\, \#1$'
else:
    figLabel = r'$\mathrm{(b)}\, \#2$'

# Colormap
cmap = plt.cm.get_cmap("jet")
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW PLOT
##################################################
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)

extent = (data.linkRot.xMin, data.linkRot.xMax, data.linkRot.yMin, data.linkRot.yMax)
dataField = data.linkRot.orderParameter.abs

# Have a consistent max between different plots
dataMax = 0.111
dataMin = 0
dataField[dataField == 0] = np.nan
print('Min: %s, Max: %s' % (np.nanmin(dataField), np.nanmax(dataField)))

cax = ax.imshow(dataField, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax.set_xlabel(r'$x/a$',fontsize=font_size)
ax.set_ylabel(r'$y/a$',fontsize=font_size)
ax.tick_params( axis='both', which='major', labelsize=tick_size )
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

props = dict(boxstyle='square', facecolor='white')
ax.text(0.02, 0.96, figLabel,
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes,
            fontsize=font_size,
            bbox=props,
            )

# Plot region of cut
xMin = 100
xMax = 153
yMin = 1
yMax = 31
ax.plot([xMin,xMax],[yMin,yMin],'r-',lw=1)
ax.plot([xMin,xMax],[yMax,yMax],'r-',lw=1)
ax.plot([xMin,xMin],[yMin,yMax],'r-',lw=1)
ax.plot([xMax,xMax],[yMin,yMax],'r-',lw=1)

##################################################
# COLOR BAR
##################################################
cbarTicks = [dataMin, dataMax]

if np.allclose(0.0, dataMin):
    dataMin_str = r'$0$'
else:
    dataMin_str = r'$%2.2f$' % dataMin
    
cbarLabels = [dataMin_str, r'$%2.2f$' % dataMax]
cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
cbar.set_ticks(cbarTicks)
cbar.set_ticklabels(cbarLabels)
cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Text label
cbarLabel = r'$|\Delta_d|/t$'
cbar.set_label(cbarLabel,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")

##################################################
# SAVE PLOTS TO FILE
##################################################
fig.savefig('%s/svg/bdg_order_parameter_full_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/bdg_order_parameter_full_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/bdg_order_parameter_full_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()