#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
utils_bdg_comparison.py
~~~~~~~~~~~~~~~~~~~~
@description: Common Python functionality for reading and rotating BdG data.
"""
#======================================================================#
# IMPORTS
#======================================================================#
from __future__ import division

import os
import sys
import numpy as np

#======================================================================#
# CLASSES
#======================================================================#
# Helper class for creating new object
class Object(object):
    pass

# Class for reading BdG data and creating data structure
class dataBdG:
    def __init__(self, inFilePath, inFermiSurfaceLabel):
        # Create empty objects for storing data
        self.node = Object()
        self.link = Object()
        self.linkRot = Object()
        self.node.current = Object()
        self.node.orderParameter = Object()
        self.link.orderParameter = Object()
        self.linkRot.orderParameter = Object()

        ##################################################
        # READ NODE DATA
        ##################################################
        currentFile = '%s/Curr.dat' % inFilePath
        currentData = np.loadtxt(currentFile)
        
        # Node coordinates
        self.node.x = currentData[:,0]
        self.node.y = currentData[:,1]

        # Calculate and create uniform coordinate ranges
        self.node.xMin = np.nanmin(self.node.x)
        self.node.xMax = np.nanmax(self.node.x)
        self.node.yMin = np.nanmin(self.node.y)
        self.node.yMax = np.nanmax(self.node.y)
        self.node.xLinspace = np.linspace(self.node.xMin, self.node.xMax, len(np.unique(self.node.x)))
        self.node.yLinspace = np.linspace(self.node.yMin, self.node.yMax, len(np.unique(self.node.y)))
        nodeBins = [len(self.node.yLinspace), len(self.node.xLinspace)]

        # Create uniform grids with all data points defined
        self.node.orderParameter.abs = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,2])[0]
        self.node.orderParameter.re = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,3])[0]
        self.node.orderParameter.im = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,4])[0]
        self.node.current.x = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,5])[0]
        self.node.current.y = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,6])[0]
        self.node.current.abs = np.histogram2d(self.node.y, self.node.x, bins=nodeBins, weights=currentData[:,7])[0]

        ##################################################
        # READ LINK DATA
        ##################################################
        orderParameterFile = '%s/OP.dat' % inFilePath
        orderParameterData = np.loadtxt(orderParameterFile)

        # Link coordinates
        self.link.x = orderParameterData[:,0]
        self.link.y = orderParameterData[:,1]

        # Calculate and create uniform coordinate ranges
        self.link.xMin = np.nanmin(self.link.x)
        self.link.xMax = np.nanmax(self.link.x)
        self.link.yMin = np.nanmin(self.link.y)
        self.link.yMax = np.nanmax(self.link.y)
        xLinkRange = self.link.xMax-self.link.xMin
        yLinkRange = self.link.yMax-self.link.yMin
        self.link.xLinspace = np.linspace(self.link.xMin, self.link.xMax, len(np.unique(self.link.x)))
        self.link.yLinspace = np.linspace(self.link.yMin, self.link.yMax, len(np.unique(self.link.y)))

        # Multiply by two due to staggered grid
        linkBins = [len(self.link.yLinspace), len(self.link.xLinspace)]

        self.link.orderParameter.re = np.histogram2d(self.link.y, self.link.x, bins=linkBins, weights=orderParameterData[:,2])[0]
        self.link.orderParameter.im = np.histogram2d(self.link.y, self.link.x, bins=linkBins, weights=orderParameterData[:,3])[0]
        self.link.orderParameter.abs = np.histogram2d(self.link.y, self.link.x, bins=linkBins, weights=orderParameterData[:,4])[0]
        self.link.orderParameter.phase = np.histogram2d(self.link.y, self.link.x, bins=linkBins, weights=orderParameterData[:,5])[0]

        ##################################################
        # ROTATE LINK DATA 45 DEGREES
        ##################################################
        # Move origin to center of grain, with ranges [-max,max]
        self.link.xTranslated = self.link.x - 0.5*(self.link.xMax + self.link.xMin)
        self.link.yTranslated = self.link.y - 0.5*(self.link.yMax + self.link.yMin)

        # Rotate coordinates
        rotationAngle = -np.pi * 0.25
        cosAngle = np.cos(rotationAngle)
        sinAngle = np.sin(rotationAngle)
        self.linkRot.x = self.link.xTranslated * cosAngle - self.link.yTranslated * sinAngle
        self.linkRot.y = self.link.xTranslated * sinAngle + self.link.yTranslated * cosAngle

        # Calculate and create uniform coordinate ranges
        self.linkRot.xMin = np.nanmin(self.linkRot.x)
        self.linkRot.xMax = np.nanmax(self.linkRot.x)
        self.linkRot.yMin = np.nanmin(self.linkRot.y)
        self.linkRot.yMax = np.nanmax(self.linkRot.y)
            
        # Note that we want the bins from the non-rotated grid - we are not making the grid "larger" by rotating, we are just re-organizing the points 
        self.linkRot.xLinspace, xStep = np.linspace(self.linkRot.xMin, self.linkRot.xMax, len(np.unique(self.link.xLinspace)), retstep=True)
        self.linkRot.yLinspace = np.arange(self.linkRot.yMin, self.linkRot.yMax, xStep)
        linkRotBins = [len(self.linkRot.yLinspace)-1, len(self.linkRot.xLinspace)-1]

        # Create uniform grids with all data points defined
        self.linkRot.orderParameter.re = np.histogram2d(self.linkRot.y, self.linkRot.x, bins=linkRotBins, weights=orderParameterData[:,2])[0]
        self.linkRot.orderParameter.im = np.histogram2d(self.linkRot.y, self.linkRot.x, bins=linkRotBins, weights=orderParameterData[:,3])[0]
        self.linkRot.orderParameter.abs = np.histogram2d(self.linkRot.y, self.linkRot.x, bins=linkRotBins, weights=orderParameterData[:,4])[0]
        self.linkRot.orderParameter.phase = np.histogram2d(self.linkRot.y, self.linkRot.x, bins=linkRotBins, weights=orderParameterData[:,5])[0]
        self.linkRot.test = np.histogram2d(self.linkRot.y, self.linkRot.x, bins=linkRotBins, weights=orderParameterData[:,5])

#======================================================================#
# FUNCTIONS
#======================================================================#
def readDataBdG(inBasePath, inFermiSurfaceLabel, inUpdateNumpyDataFile):
    # Path to BdG data from Micke
    filePath = str('%s/bdg/bs%d' % (inBasePath, inFermiSurfaceLabel))
    # Path to numpy data file (for faster load times)
    filePathNumpy = str('%s/python/bdg_fs%d.npy' % (inBasePath, inFermiSurfaceLabel))

    # Create numpy data directory if not existing
    if not os.path.exists(str('%s/python/' % inBasePath)):
        pythonDirectory = '%s/python/' % inBasePath
        os.mkdir(pythonDirectory)
        print('Created directory: %s' % pythonDirectory)

    # Check if Numpy file exists
    if (not os.path.isfile(filePathNumpy)) or inUpdateNumpyDataFile:
        tempRunData = dataBdG(filePath, inFermiSurfaceLabel)

        # Save numpy data file
        np.save(filePathNumpy, tempRunData)
        print('Save: %s' % filePathNumpy)
    else:
        # Read numpy data file
        tempRunData = np.load(filePathNumpy).item()
        print('Load: %s' % filePathNumpy)

    return tempRunData