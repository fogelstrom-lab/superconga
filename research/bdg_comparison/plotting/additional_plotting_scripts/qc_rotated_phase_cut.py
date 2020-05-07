#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_rotated_phase_cut.py
~~~~~~~~~~~~~~~~~~~~
@description: Plot a rotated phase from quasiclassical data, with a cut specified by coordinates.
"""
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

from scipy.ndimage.interpolation import rotate

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

#======================================================================#
# FUNCTIONS
#======================================================================#

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



fig2 = plt.figure(figsize=(fig2Width*inchesPerPt, 1.3*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
fig2.subplots_adjust(left=0.12,
                     bottom=0.03,
                     right=0.96,
                     top=0.88,
                     wspace=0,
                     hspace=0)

gs = gridspec.GridSpec(100,100)

ax_cbar2 = plt.subplot(gs[0:3,:])
ax2 = plt.subplot(gs[5:,:])


# Cuts figure
fig3, axGrid3 = plt.subplots(2,1, figsize=(fig2Width*inchesPerPt, 1.5*fig2Width*inchesPerPt*goldenRatio), dpi=dpi)
"""
fig2.subplots_adjust(left=0.13,
                     bottom=0.15,
                     right=0.98,
                     top=0.96,
                     wspace=0,
                     hspace=0)
"""
ax3a = axGrid3[0]
ax3b = axGrid3[1]

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

fieldList = [
'delta',
'current',
]

dataBasePath = '../../data'
figSavePath = '../figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = False
temperature = 0.16
pbSideLength = 60.0
grainCornerLength = 4.0
fermiSurfaceLabel = 2

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
    run = DS.runData(filePathQC, fields_in=fieldList,
                     #fileNames_in=fileNames,
                     zeroToNaN=False)
    
    # Save numpy data file
    np.save(filePathNumpy, run)
    print('Save: %s' % filePathNumpy)
else:
    # Read numpy data file
    run = np.load(filePathNumpy).item()
    print('Load: %s' % filePathNumpy)


#======================================================================#
# PLOT DATA
#======================================================================#

##################################################
# FIELDS
##################################################
print(run.params)

#for s_key in systems.keys():
sideLength = run.params['sideLength']
numLatticePoints = run.params['numLatticePoints']
grainMargin = run.params['grainMargin']
b = (sideLength/(1.0 - grainMargin) - sideLength) * 0.5

extent = (-b, sideLength+b, -b, sideLength+b)#(0,sideLength,0,sideLength*0.25)
aspect = 1.0#'auto'

xLength = sideLength
yLength = sideLength

minIndex = run.params['minIndex']
maxIndex = run.params['maxIndex']


##################################################
# IMSHOW PLOT: FIG 1
##################################################
currentMin = 0.0#np.nanmin(run.current.abs)
currentMax = np.nanmax(run.current.abs)

cmap = plt.cm.get_cmap("RdBu")#mpl.cm.get_cmap('myred')

# Set color outside grain: This is problematic with rotation...
cmap.set_bad(badColor,1.0)
dataField = run.delta.phase / np.pi
dataMaxUncentered = np.nanmax(dataField)
dataMinUncentered = np.nanmin(dataField)

# Center phase around 0
dataRange = abs(dataMaxUncentered - dataMinUncentered)
dataMin = -0.5*dataRange
dataMax = 0.5*dataRange
dataField = (dataField - dataMinUncentered) + dataMin

# Rotate field 45 degrees
dataField = rotate(dataField, angle=45)
dataField[dataField == 0] = np.nan
# Indices for cut
xMin = 400
xMax = 733
yMin = 520
yMax = 650
xRange = xMax - xMin
yRange = yMax - yMin

cax = ax.imshow(dataField, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
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
ax.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
ax.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
ax.tick_params( axis='both', which='major', labelsize=tick_size )
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
#ax.set_xticks([])
#ax.set_yticks([])

##################################################
# COLOR BAR: FIG 1
##################################################
cbarTicks = [dataMin, dataMax]
    
cbarLabels = [r'$%2.2f$' % dataMin, r'$%2.2f$' % dataMax]
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
# QUIVER PLOT: FIG 2
##################################################
phaseCut = dataField[yMin:yMax+1,xMin:xMax+1]

cax2 = ax2.imshow(phaseCut, cmap=cmap,
                vmin=dataMin, vmax=dataMax,
                #extent=extent,
                interpolation=interpolation,
                aspect=aspect,
                origin='lower')

# Axis ticks and labels
ax2.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
ax2.set_ylabel(r'$y/\xi_0$',fontsize=font_size)
ax2.tick_params( axis='both', which='major', labelsize=tick_size )
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')

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
                #labelpad=10,
                fontsize=font_size)
# Smoother appearance
cbar2.solids.set_edgecolor("face")

##################################################
# TEST PLOTTING CUTS
##################################################

xCuts = [0,2,4,6,8,10]
yCuts = [0,10,21,42]
linestyles = ['-','-','--','--','-.','-.']
colors = ['r','g','b','c','m','k']
linewidths = [1,2,2,3,2,3]

xTest = np.arange(0,xRange+1)
yTest = np.arange(0,yRange+1)

testPeriod = 0.5*xRange
testDecay = 20.0

def analyticPhase(inX, inY, inDecay, inPeriod, inAmplitude):
    waveVector = 2.0 * np.pi / inPeriod
    return inAmplitude * np.cos(waveVector * inX) * (1.0 + inY/inDecay) * np.exp(-inY/inDecay)

ax3a.plot(xTest, analyticPhase(xTest,0,1,testPeriod,dataMax),'k-',lw=1,label=r'$\mathrm{Phase\, crystal},\, \rho=%d\xi_0$' % testPeriod)
ax3b.plot(yTest, analyticPhase(0,yTest,testDecay,1,dataMax),'k-',lw=1,label=r'$\mathrm{Phase\, crystal},\, y_0=%d\xi_0$' % testDecay)

for cutIdx in range(0,len(xCuts)):
    xCut = xCuts[cutIdx]
    xCutLabel = r'$y/\xi_0=%d$' % xCut
    ax3a.plot(phaseCut[xCut,:],ls=linestyles[cutIdx],c=colors[cutIdx],lw=linewidths[cutIdx],label=xCutLabel)

ax3a.set_xlabel(r'$x/\xi_0$',fontsize=font_size)
ax3a.set_ylabel(r'$\chi(x)/\pi$',fontsize=font_size)
ax3a.tick_params( axis='both', which='major', labelsize=tick_size )
ax3a.xaxis.set_ticks_position('bottom')
ax3a.yaxis.set_ticks_position('left')
ax3a.set_xlim(0,xRange)
ax3a.set_ylim(-0.53,0.53)
leg3a = ax3a.legend(prop={'size':font_size},
                  loc=1,
                  ncol=1,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  )#, handlelength=1.0)

for cutIdx in range(0,len(yCuts)):
    yCut = yCuts[cutIdx]
    yCutLabel = r'$x/\xi_0=%d$' % yCut
    ax3b.plot(phaseCut[:,yCut],ls=linestyles[cutIdx],c=colors[cutIdx],lw=linewidths[cutIdx],label=yCutLabel)

ax3b.set_xlabel(r'$y/\xi_0$',fontsize=font_size)
ax3b.set_ylabel(r'$\chi(y)/\pi$',fontsize=font_size)
ax3b.tick_params( axis='both', which='major', labelsize=tick_size )
ax3b.xaxis.set_ticks_position('bottom')
ax3b.yaxis.set_ticks_position('left')
ax3b.set_xlim(0,yRange)
ax3b.set_ylim(0,0.53)
leg3b = ax3b.legend(prop={'size':font_size},
                  loc=1,
                  ncol=1,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  )#, handlelength=1.0)

##################################################
# SAVE PLOTS TO FILE
##################################################
fig2.savefig('%s/svg/qc_phase_rotated_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig2.savefig('%s/pdf/qc_phase_rotated_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig2.savefig('%s/png/qc_phase_rotated_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

fig3.tight_layout()
fig3.savefig('%s/svg/qc_phase_rotated_cuts_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig3.savefig('%s/pdf/qc_phase_rotated_cuts_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig3.savefig('%s/png/qc_phase_rotated_cuts_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

plt.show()