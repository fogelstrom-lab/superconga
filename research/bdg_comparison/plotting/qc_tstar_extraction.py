#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qc_tstar_extraction.py
~~~~~~~~~~~~~~~~~~~~
@description: Extract the transition temperature from the average current
              magnitude.
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
sys.path.insert(0, '../../../tools/')

from pythonLibraries.fileIO import readFiles as RF
from pythonLibraries.plotting import plotFields as PF
from pythonLibraries.dataAnalysis import dataStructs as DS

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

fig, ax = plt.subplots(1,1,figsize=(fig2Width*inchesPerPt, fig2Width*inchesPerPt*goldenRatio), dpi=dpi)

linestyles = ['-','--','-.']
colors = ['r','g','b']
linewidths = [1,2,2]

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

filenameList = [
'order_parameter',
'current_density',
]

dataBasePath = '../data'
figSavePath = 'figures'

# Read raw data even if numpy data file exists
updateNumpyDataFile = False
pbSideLength = 60.0
grainCornerLength = 4.0

pbSideLength_str = str('%2.3f' % pbSideLength).replace('.','-')
grainCornerLength_str = str('%2.3f' % grainCornerLength).replace('.','-')

temperatures = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15,
                         0.155, 0.157, 0.159, 0.16, 0.17,
                         0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
                         0.247, 0.248, 0.249,
                         0.25, 0.251, 0.2515, 0.252, 0.253, 0.255, 0.257,
                         0.26, 0.265, 0.27,
                         0.29, 0.3, 0.4])

fermiSurfaceLabels = np.array([1, 2, 3])

# Empty lists for storing results
resultDict = {}


for fermiSurfaceIdx in range(0, len(fermiSurfaceLabels)):
    fermiSurfaceLabel = fermiSurfaceLabels[fermiSurfaceIdx]

    resultDict[fermiSurfaceIdx] = {
        'currents': [],
        'currentAverages': [],
        'numLatticePoints': [],
        'freeEnergies': [],
        'freeEnergiesMetaStable': [],
        'temperatures': [],
        'temperaturePhaseCrystal': [],
        'temperatureUniformPhase': [],
        'currentPhaseCrystal': [],
        'currentUniformPhase': [],
    }

    for tempIdx in range(0, len(temperatures)):
        temperature = temperatures[tempIdx]
        temperature_str = str('%2.4f' % temperature).replace('.','-')

        # Path to quasiclassics data
        filePathQC = str('%s/qc/fs%d_R%2.3f_r%2.3f_T%2.4f' % (dataBasePath, fermiSurfaceLabel, pbSideLength, grainCornerLength, temperature))
        filePathMetaStableQC = str('%s/qc/fs%d_R%2.3f_r%2.3f_T%2.4f_ms' % (dataBasePath, fermiSurfaceLabel, pbSideLength, grainCornerLength, temperature))
        # Path to numpy data file (for faster load times)
        filePathNumpy = str('%s/python/rundata_fs%d_R%s_r%s_T%s.npy' % (dataBasePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str))

        # Continue with next temperature if data does not exist
        if ((not os.path.isfile( str('%s/free_energy.dat' % filePathQC) )) and
            (not os.path.isfile( filePathNumpy )) ):
            continue

        # Create numpy data directory if not existing
        if not os.path.isdir(str('%s/python/' % dataBasePath)):
            pythonDirectory = '%s/python/' % dataBasePath
            print('Create directory: %s' % pythonDirectory)
            os.mkdir(pythonDirectory)

        # Read quasiclassics data
        if (not os.path.isfile(filePathNumpy)) or updateNumpyDataFile:
            # Read raw data
            run = DS.runData(filePathQC, fields_in=fieldList, fileNames_in=filenameList,
                            zeroToNaN=False)
            
            # Save numpy data file
            np.save(filePathNumpy, run)
            print('Save: %s' % filePathNumpy)
        else:
            # Read numpy data file
            run = np.load(filePathNumpy).item()
            print('Load: %s' % filePathNumpy)

        # Read free energy data
        freeEnergyDict = RF.getLastFreeEnergy(filePathQC, fileName='free_energy.dat')
        freeEnergyDictMetaStable = RF.getLastFreeEnergy(filePathMetaStableQC, fileName='free_energy.dat')
        
        freeEnergy = np.float(freeEnergyDict['freeEnergyPerUnitArea'])
        freeEnergyMetaStable = np.float(freeEnergyDictMetaStable['freeEnergyPerUnitArea'])

        # Check which state has lowest free energy
        if freeEnergyMetaStable > freeEnergy:
            resultDict[fermiSurfaceIdx]['currentPhaseCrystal'].append( np.sum(run.current.abs) / run.params['internalLatticePoints'] )
            resultDict[fermiSurfaceIdx]['temperaturePhaseCrystal'].append( temperature )
        else:
            resultDict[fermiSurfaceIdx]['currentUniformPhase'].append( 0 )
            resultDict[fermiSurfaceIdx]['temperatureUniformPhase'].append( temperature )

        # Append results
        resultDict[fermiSurfaceIdx]['currents'].append( np.sum(run.current.abs) )
        resultDict[fermiSurfaceIdx]['currentAverages'].append( np.sum(run.current.abs) / run.params['internalLatticePoints'] )
        resultDict[fermiSurfaceIdx]['numLatticePoints'].append( run.params['internalLatticePoints'] )
        resultDict[fermiSurfaceIdx]['freeEnergies'].append( freeEnergy )
        resultDict[fermiSurfaceIdx]['freeEnergiesMetaStable'].append( freeEnergyMetaStable )
        resultDict[fermiSurfaceIdx]['temperatures'].append( temperature )
    
    print(resultDict[fermiSurfaceIdx])

    ##################################################
    # PLOT AVERAGE CURRENT MAGNITUDE
    ##################################################
    if fermiSurfaceIdx == 0:
        plotLabel = r'$\mathrm{Circular\, FS}$'
    elif fermiSurfaceIdx == 1:
        plotLabel = r'$\mathrm{FS\, \#1}: \mu=0, t^\prime=-0.25$'
    elif fermiSurfaceIdx == 2:
        plotLabel = r'$\mathrm{FS\, \#2}: \mu=-1.267, t^\prime=-0.495, t^{\prime\prime}=0.156$'
    else: 
        plotLabel = r'$\mathrm{Unknown\, FS!?}$'

    # Plot Phase Crystal points
    if len(resultDict[fermiSurfaceIdx]['currentPhaseCrystal']) > 0:
        ax.plot(np.array(resultDict[fermiSurfaceIdx]['temperaturePhaseCrystal']),
                np.array(resultDict[fermiSurfaceIdx]['currentPhaseCrystal']) * 1e2,
                lw=linewidths[fermiSurfaceIdx],
                c=colors[fermiSurfaceIdx],
                ls=linestyles[fermiSurfaceIdx],
                marker='o',
                ms=10,
                label=plotLabel)
    # Plot Uniform Phase points
    if len(resultDict[fermiSurfaceIdx]['currentUniformPhase']) > 0:
        ax.plot(np.array(resultDict[fermiSurfaceIdx]['temperatureUniformPhase']),
                np.array(resultDict[fermiSurfaceIdx]['currentUniformPhase']),
                lw=linewidths[fermiSurfaceIdx],
                c=colors[fermiSurfaceIdx],
                marker='s',
                ms=10,
                ls=linestyles[fermiSurfaceIdx])


ax.set_xlabel(r'$T/T_{\mathrm{c}}$',fontsize=font_size)
ax.set_ylabel(r'$\overline{j} / j_{\mathrm{d}}$  $(\times 10^{-2})$',fontsize=font_size)
ax.tick_params( axis='both', which='major', labelsize=tick_size )
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xlim(0.1,0.3)
ax.set_ylim(0,3)
leg = ax.legend(prop={'size':font_size},
                  loc=1,
                  ncol=1,
                  borderpad=0.5,
                  labelspacing=0.2,
                  borderaxespad=1.1,
                  )#, handlelength=1.0)

##################################################
# SAVE PLOTS TO FILE
##################################################
fig.tight_layout()
fig.savefig('%s/svg/qc_tstar_fs%d_R%s_r%s_T%s.svg' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='svg',dpi=300,transparent=False)
fig.savefig('%s/pdf/qc_tstar_fs%d_R%s_r%s_T%s.pdf' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='pdf',dpi=300,transparent=False)
fig.savefig('%s/png/qc_tstar_fs%d_R%s_r%s_T%s.png' % (figSavePath, fermiSurfaceLabel, pbSideLength_str, grainCornerLength_str, temperature_str),format='png',dpi=300,transparent=False)

plt.show()