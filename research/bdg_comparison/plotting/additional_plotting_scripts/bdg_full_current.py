#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_full_current.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a single current field from BDG data. Convenient to get an overview of the data.
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

from utils_bdg_comparison import *

#======================================================================#
# SYSTEMS AND FILES
#======================================================================#

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

dataBasePath = '../../data'
figSavePath = '../figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = True
fermiSurfaceBdG = 4
cmap = mpl.cm.get_cmap('myred')
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW PLOT
##################################################
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)

extent = (data.node.xMin, data.node.xMax, data.node.yMin, data.node.yMax)
dataField = data.node.current.abs
dataMax = np.nanmax(dataField)
dataMin = np.nanmin(dataField)
dataField[dataField == 0] = np.nan

print('Min: %s, Max: %s' % (dataMin, dataMax))

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
cbarLabel = r'$\left|\mathbf{j}\right|/j_{\mathrm{d}}$'
cbar.set_label(cbarLabel,
                #rotation=270,
                #labelpad=10,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")

##################################################
# TEST PLOTTING CUTS
##################################################
"""
f2, ax2 = plt.subplots()

iNum, jNum = np.shape(dataField)

iOffests = [5,10,15,16,17,18,19,20,21,22]
for iOffset in iOffests:
    iDiagonal = np.arange(0,iNum-iOffset)
    ax2.plot(iDiagonal,dataField[iDiagonal,iDiagonal+iOffset])

plt.show()
exit()
"""
##################################################
# SAVE PLOTS TO FILE
##################################################
fig.savefig('%s/svg/bdg_current_full_fs%d.svg' % (figSavePath, fermiSurfaceBdG),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/bdg_current_full_fs%d.pdf' % (figSavePath, fermiSurfaceBdG),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/bdg_current_full_fs%d.png' % (figSavePath, fermiSurfaceBdG),format='png',dpi=300,transparent=False)

plt.show()