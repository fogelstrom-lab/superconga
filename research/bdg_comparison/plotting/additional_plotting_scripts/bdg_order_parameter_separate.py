#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_order_parameter_separate.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a single order parameter field from BDG data, with a cut specified by coordinates.
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

# Imshow figure
fig = plt.figure(figsize=(fig2Width*inchesPerPt, 1.3*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig.subplots_adjust(left=0.10,
                     bottom=0.05,
                     right=0.97,
                     top=0.85,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar = plt.subplot(gs[0:5,:])
ax = plt.subplot(gs[5:,:])

# Cuts figure
fig2, axGrid2 = plt.subplots(2,1, figsize=(fig2Width*inchesPerPt, 1.5*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
"""
fig2.subplots_adjust(left=0.13,
                     bottom=0.15,
                     right=0.98,
                     top=0.96,
                     wspace=0,
                     hspace=0)
"""
ax2a = axGrid2[0]
ax2b = axGrid2[1]

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
fermiSurfaceBdG = 2
cmap = plt.cm.get_cmap("jet")#mpl.cm.get_cmap('myred')#
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW PLOT
##################################################
# Indices for cut
if fermiSurfaceBdG == 2:
    xMin = 100
    xMax = 153
else:
    xMin = 100
    xMax = 153

yMin = 1
yMax = 31

xRange = xMax-xMin
yRange = yMax-yMin
xArange = np.arange(xMin,xMax+1)

# Fetch data
data = readDataBdG(dataBasePath, fermiSurfaceBdG, updateNumpyDataFile)
#extent = (data.linkRot.xMin, data.linkRot.xMax, data.linkRot.yMin, data.linkRot.yMax)
dataField = data.linkRot.orderParameter.abs[yMin:yMax+1,xMin:xMax+1]
dataMax = 0.11#np.nanmax(dataField)
dataMin = 0#np.nanmin(dataField)
dataField[dataField == 0] = np.nan

cax = ax.imshow(dataField, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                #extent=extent,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

print('Min: %s, Max: %s' % (dataMin, dataMax))

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
cbarLabel = r'$|\Delta_d|(\mathbf{R}) / t$'
cbar.set_label(cbarLabel,
                #rotation=270,
                #labelpad=10,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")

##################################################
# TEST PLOTTING CUTS
##################################################
xCuts = [0,5,10,20]
yCuts = [14,20,26]

colors = ['r','g','b','c','m']
colorsY = ['m','k','#FFc0c0','g','b']
linewidths = [2,2,2,2,2,2]
linewidthsY = [2,2,2,3,2,2]

if fermiSurfaceBdG == 2:
    linestyles = ['-','-','-','-','-','-']
    linestylesY = ['-','-','-','-','-','-']
else:
    linestyles = ['--','--','--','--','--','--']
    linestylesY = ['--','--','--','--','--','--']


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
    ax2a.plot(xArange, dataField[xCut,0:],
              ls=linestyles[cutIdx],c=colors[cutIdx],
              lw=linewidths[cutIdx],label=xCutLabel)

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
        ax2a.plot(xArange, dataNonPB,
                ls='-.',c=colors[cutIdx],
                lw=2,label=labelParallel)

ax2a.set_xlabel(r'$x/a$',fontsize=font_size)
ax2a.set_ylabel(r'$|\Delta_d(x)|/t$',fontsize=font_size)
ax2a.tick_params( axis='both', which='major', labelsize=tick_size )
ax2a.xaxis.set_ticks_position('bottom')
ax2a.yaxis.set_ticks_position('left')
ax2a.set_xlim(xMin,xMax)
ax2a.set_ylim(0.0,0.08)
ax2a.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
ax2a.set_yticklabels([r'$0$', r'$0.02$', r'$0.04$', r'$0.06$', r'$0.08$'])
leg2a = ax2a.legend(prop={'size':font_size},
                  loc=4,
                  ncol=2,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  fancybox=True,
                  #framealpha=1.0,
                  )#, handlelength=1.0)
ax2a.text(0.02, 0.96, r'$\mathrm{(b)}$',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2a.transAxes,
            fontsize=font_size)

# Plot perpendicular to pairbreaking edge: fixed x
for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/a=%d$' % (yCut+xMin)
    ax2b.plot(dataField[:,yCut],
              ls=linestylesY[cutIdx],c=colorsY[cutIdx],
              lw=linewidthsY[cutIdx],label=yCutLabel)

# Origin for non-pairbreaking edge plotting
if fermiSurfaceBdG == 2:
    parallelX0 = 100
else: 
    parallelX0 = 101
parallelY0 = 118

# Plot perpendicular against non-pairbreaking edge
perpDataNonPB = []
for diagIdx in range(0, yRange):
    perpDataNonPB.append(data.linkRot.orderParameter.abs[parallelY0-diagIdx,
                                                         parallelX0+diagIdx])
ax2b.plot(perpDataNonPB,ls=linestyles[0],lw=3,c='#0366FC',label=r'$\mathrm{Non-PB\, edge}$')

ax2b.set_xlabel(r'$y/a$',fontsize=font_size)
ax2b.set_ylabel(r'$|\Delta_d(y)|/t$',fontsize=font_size)
ax2b.tick_params( axis='both', which='major', labelsize=tick_size )
ax2b.xaxis.set_ticks_position('bottom')
ax2b.yaxis.set_ticks_position('left')
ax2b.set_xlim(0,yRange)
ax2b.set_ylim(0,0.11)
#ax2b.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
#ax2b.set_yticklabels([r'$0$', r'$0.02$', r'$0.04$', r'$0.06$', r'$0.08$'])
leg2b = ax2b.legend(prop={'size':font_size},
                  loc=4,
                  ncol=1,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  fancybox=True,
                  #framealpha=1.0,
                  )#, handlelength=1.0)
ax2b.text(0.02, 0.96, r'$\mathrm{(c)}$',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2b.transAxes,
            fontsize=font_size)
##################################################
# SAVE PLOTS TO FILE
##################################################
fig.savefig('%s/svg/bdg_order_parameter_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/bdg_order_parameter_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/bdg_order_parameter_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

fig2.tight_layout()
fig2.savefig('%s/svg/bdg_order_parameter_cuts_sep_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_order_parameter_cuts_sep_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_order_parameter_cuts_sep_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()