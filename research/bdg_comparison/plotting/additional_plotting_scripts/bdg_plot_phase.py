#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdg_plot_phase.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a single phase field from BDG data, with a cut specified by coordinates.
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

ax_cbar = plt.subplot(gs[0:5,:65])
ax = plt.subplot(gs[5:,:65])

# Cuts figure
fig2, ax2b = plt.subplots(1,1, figsize=(fig2Width*inchesPerPt, 0.8*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)

"""
fig2.subplots_adjust(left=0.13,
                     bottom=0.15,
                     right=0.98,
                     top=0.96,
                     wspace=0,
                     hspace=0)
"""
ax2in = inset_axes(ax2b, width="60%", height="50%", loc=1)

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
cmap = plt.cm.get_cmap("RdBu_r")
cmap.set_bad(badColor,1.0)

##################################################
# IMSHOW PLOT
##################################################
# Indices for cut
if fermiSurfaceBdG == 2:
    xMin = 100
    xMax = 153
    xCutTicks = [-0.5, -0.25, 0, 0.25, 0.5]
    yCutTicks = [0, 0.25, 0.5]
else:
    xMin = 100
    xMax = 153
    xCutTicks = [-0.15, 0, 0.15]
    yCutTicks = [0, 0.25]

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

# Plot phase
cax = ax.imshow(dataField, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                extent=(xMin-dpt,xMax+dpt,-dpt,yRange+dpt),
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')


# Axis ticks and labels
#ax.set_xticks(tickLabels)
#ax.set_yticks(tickLabels)
ax.set_xlabel(r'$x/a$',fontsize=font_size)
ax.set_ylabel(r'$y/a$',fontsize=font_size)
ax.tick_params( axis='both', which='major', labelsize=tick_size )
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks([100, 110, 120, 130, 140, 150])
#ax.set_yticks([])
ax.text(0.02, 0.96, r'$\mathrm{(a)}$',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax.transAxes,
            fontsize=font_size)

##################################################
# COLOR BAR
##################################################
cbarTicks = [dataMin, 0, dataMax]

if np.allclose(0.0, dataMin):
    dataMin_str = r'$0$'
else:
    dataMin_str = r'$%2.2f$' % dataMin
    
cbarLabels = [dataMin_str, r'$0$', r'$%2.2f$' % dataMax]
cbar = fig.colorbar(cax, cax=ax_cbar, orientation='horizontal', ticks=cbarTicks)
cbar.set_ticks(cbarTicks)
cbar.set_ticklabels(cbarLabels)
cbar.ax.tick_params( axis='both', which='major', labelsize=cbar_size )
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# Text label
cbarLabel = r'$\chi(\mathbf{R}) / \pi$'
cbar.set_label(cbarLabel,
                #rotation=270,
                #labelpad=10,
                fontsize=font_size)

# Smoother appearance
cbar.solids.set_edgecolor("face")

##################################################
# TEST PLOTTING CUTS
##################################################
linestyles = ['-','-','--','--','-.','-.']
colors = ['r','g','b','c','m','k']
linewidths = [1,2,2,3,2,3]

linestylesY = ['-','--','-.','--','-.','-.']
colorsY = ['m','#0000c0','#00c0c0','#c0c000']
linewidthsY = [2,2,2,2,2,3]

xCuts = [0,5,10]
yTest = np.arange(0,yRange+1)

if fermiSurfaceBdG == 2:
    yCuts = [14,20,26]
    testDecay = 6.4
    testPeriod = xRange
    xTest = np.arange(0,xRange+1)
else: 
    yCuts = [14,20,26]
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
    ax2c.plot(xArange, dataField[xCut,:],
              ls=linestyles[cutIdx],c=colors[cutIdx],
              lw=linewidths[cutIdx],label=xCutLabel)
    if np.max(dataField[xCut,:]) > maxPhase:
        maxPhase = np.max(dataField[xCut,:])

# Plot analytic phase crystal (manual fit)
if fermiSurfaceBdG == 2:
    maxPhaseFit_a = maxPhase
    maxPhaseFit_b = 0.22#0.9*maxPhase
else:
    maxPhaseFit_a = maxPhase
    maxPhaseFit_b = 0.95*dataMax

ax2c.plot(xArange, analyticPhase(xTest,0,1,testPeriod,maxPhaseFit_a),'k-',lw=1,label=r'$\mathrm{Fit}$')
ax2b.plot(yTest, analyticPhase(1,yTest,testDecay,4.0,maxPhaseFit_b),'k-',lw=1,label=r'$\mathrm{Fit}$')
#ax2c.plot(xArange, np.zeros_like(xArange))


ax2c.set_xlabel(r'$x/a$',fontsize=font_size)
ax2c.set_ylabel(r'$\chi(x)/\pi$',fontsize=font_size)
ax2c.tick_params( axis='both', which='major', labelsize=tick_size )
ax2c.xaxis.set_ticks_position('bottom')
ax2c.yaxis.set_ticks_position('left')
ax2c.set_xlim(xMin,xMax)
ax2c.set_ylim(xCutTicks[0],xCutTicks[-1])
ax2c.set_yticks(xCutTicks)
#ax2c.set_yticklabels([r'$-0.5$',r'$-0.25$',r'$0$',r'$0.25$',r'$0.5$'])
leg2a = ax2c.legend(prop={'size':font_size},
                  loc=1,
                  ncol=2,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  fancybox=True,
                  #framealpha=1.0,
                  )#, handlelength=1.0)
ax2c.text(0.02, 0.96, r'$\mathrm{(c)}$',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2c.transAxes,
            fontsize=font_size)

for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/a=%d$' % (yCut+xMin)
    ax2b.plot(dataField[:,yCut],
              ls=linestylesY[cutIdx],c=colorsY[cutIdx],
              lw=linewidthsY[cutIdx],label=yCutLabel)

ax2b.set_xlabel(r'$y/a$',fontsize=font_size)
ax2b.set_ylabel(r'$\chi(y)/\pi$',fontsize=font_size)
ax2b.tick_params( axis='both', which='major', labelsize=tick_size )
ax2b.xaxis.set_ticks_position('bottom')
ax2b.yaxis.set_ticks_position('left')
ax2b.set_xlim(0,yRange)
ax2b.set_ylim(yCutTicks[0],yCutTicks[-1])
ax2b.set_yticks(yCutTicks)
#ax2b.set_yticklabels([r'$0$',r'$0.25$',r'$0.5$'])
leg2b = ax2b.legend(prop={'size':font_size},
                  loc=2,
                  ncol=1,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  fancybox=True,
                  #framealpha=1.0,
                  )#, handlelength=1.0)
ax2b.text(0.02, 0.96, r'$\mathrm{(b)}$',
            horizontalalignment='left',
            verticalalignment='top',
            transform = ax2b.transAxes,
            fontsize=font_size)


##################################################
# SAVE PLOTS TO FILE
##################################################
fig.savefig('%s/svg/bdg_phase_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/bdg_phase_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/bdg_phase_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

fig2.tight_layout()
fig2.savefig('%s/svg/bdg_phase_cuts_fs%d.svg' % (figSavePath, fermiSurfaceBdG/2),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/bdg_phase_cuts_fs%d.pdf' % (figSavePath, fermiSurfaceBdG/2),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/bdg_phase_cuts_fs%d.png' % (figSavePath, fermiSurfaceBdG/2),format='png',dpi=300,transparent=False)

plt.show()