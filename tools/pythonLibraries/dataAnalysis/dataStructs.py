from __future__ import division

import os
import os.path
import sys

import time
import math
import numpy as np
from pythonLibraries.fileIO import readFiles as RF

# Default field names and components
defaultFieldDict = {
#   'observable': ['field1','field2',...]
    'delta':     ['re', 'im', 'abs', 'phase'],
    'current':   ['x', 'y', 'abs'],
    'B':         ['abs'],
    'A':         ['x', 'y', 'abs'],
    'divA':      ['abs'],
    'divJ':      ['abs'],
    'ldos':      ['abs'],
    'jspectral': ['x', 'y', 'abs'],
    'ps': ['x', 'y', 'abs'],
}
defaultFieldList = []
for observable in defaultFieldDict.keys():
    for field in defaultFieldDict[observable]:
        defaultFieldList.append(str('%s.%s' % (observable,field)))

# Class for creating data structure from simulation
class runData:
    def __init__(self, filePath, fields_in=None, fileNames_in=None,
                 getPhase=True, verbose=False, zeroToNaN=True):
        self.filePath = filePath

        # Initialize parameter object
        self.params = RF.readMetaDataFile(self.filePath)

        # All the field and observable names were changed for the first git version.
        # As a result, here is a lot of messy code to try to ensure backwards compability.
        if self.params != None:
            # Make sure grain margin is set
            if not self.params.has_key('grainMargin'):
                if self.params.has_key('outerRadius'):
                    self.params['grainMargin'] = (1.0 - 2.0*self.params['outerRadius'])
                else:
                    print('WARNING! Neither grainMargin nor outerRadius were specified! Assuming a margin of 0.04.')
                    self.params['grainMargin'] = 0.04
                    self.params['outerRadius'] = 0.5*(1.0 - self.params['grainMargin'])
            # Make sure that outerRadius is set, for backwards compability
            if not self.params.has_key('outerRadius'):
                # Due to the above if-statements, it is safe to assume grain margin is set
                self.params['outerRadius'] = 0.5*(1.0 - self.params['grainMargin'])

            # Make sure number of lattice points is set
            if not self.params.has_key('numLatticePoints'):
                if self.params.has_key('Nx'):
                    self.params['numLatticePoints'] = self.params['Nx']
                else:
                    raise Exception('ERROR! Neither Nx nor numLatticePoints were specified!')
            # Make sure that sideLength is set
            if not self.params.has_key('sideLength'):
                if self.params.has_key('R'):
                    self.params['sideLength'] = self.params['R']
                else:
                    raise Exception('ERROR! Neither sideLength nor R were specified!')

            # Set start/end point of physical grain in simulation space
            self.params['minIndex'] = self.getLatticePoint( 0,
                                                            self.params['sideLength'],
                                                            self.params['numLatticePoints'],
                                                            self.params['outerRadius'] )
            self.params['maxIndex'] = self.getLatticePoint( self.params['sideLength'],
                                                            self.params['sideLength'],
                                                            self.params['numLatticePoints'],
                                                            self.params['outerRadius'] )

        # Initialize observables
        if (fields_in == None) or (fields_in == 'all') or (not fields_in):
            self.delta      = observable(self, 'delta',     self.filePath, zeroToNaN=zeroToNaN, verbose=verbose,
                                         getPhase=getPhase)
            self.ps         = observable(self, 'ps',        self.filePath, zeroToNaN=zeroToNaN)
            self.current    = observable(self, 'current',   self.filePath, zeroToNaN=zeroToNaN, verbose=verbose)
            self.B          = observable(self, 'B',         self.filePath, zeroToNaN=zeroToNaN, verbose=verbose)
            self.A          = observable(self, 'A',         self.filePath, zeroToNaN=zeroToNaN)
            self.divA       = observable(self, 'divA',      self.filePath, zeroToNaN=zeroToNaN)
            self.divJ       = observable(self, 'divJ',      self.filePath, zeroToNaN=zeroToNaN)
            self.LDOS       = observable(self, 'ldos',      self.filePath, zeroToNaN=zeroToNaN)
            self.jspectral  = observable(self, 'jspectral', self.filePath, zeroToNaN=zeroToNaN)
        else:
            __fieldDict, __fieldList, __fileNames, __fieldMessage = check_fields_in(fields_in, fileNames_in)

            if __fieldDict == None:
                raise Exception('Unknown fields parameter: %s\nError message: %s' %
                                (fields_in, __fieldMessage))

            for r_i in range(0,len(sorted(__fieldDict))):
                observableName = sorted(__fieldDict)[r_i]
                fileName = __fileNames[observableName][0]

                setattr( self,
                         observableName,
                         observable( self, 
                                     observableName,
                                     self.filePath,
                                     fileName=fileName,
                                     getPhase=getPhase,
                                     verbose=verbose,
                                     zeroToNaN=zeroToNaN) )

            del __fieldDict
            del __fieldList
            del __fileNames
            del __fieldMessage
    
    # Functions
    def getObservable(self, observable_in):
        return getattr(self, observable_in)

    def getObservableProperty(self, observable_in, property_in):
        return getattr(getattr(self, observable_in), property_in)

    def getFieldMax(self, observable_in, field_in):
        return np.nanmax(self.getObservableProperty(observable_in, field_in))

    def getFieldMin(self, observable_in, field_in):
        return np.nanmin(self.getObservableProperty(observable_in, field_in))

    # Todo(patric): replace all dependence outerRadius in favor of margin. This removes backwards-compability though, which is currently useful
    def getLatticePoint( self, coordinate, sideLength, Nx, outerRadius ):
        return int(float(coordinate)/float(sideLength) * float(Nx-1)*2.0*outerRadius +\
                   0.5*float(Nx-Nx*2.0*outerRadius) )


class observable:
    def __init__(self, caller, gridName, filePath, fileName, passNoExist=False, getPhase=False, verbose=False, zeroToNaN=True):
        # Human readable name
        self.name = gridName

        # Add trailing slash to path
        self.filePath = filePath
        if filePath[-1] != '/':
            self.filePath += '/'

        # Create path to grid data file
        if fileName == None:
            self.filePath += str(gridName) + '.grd'
        else:
            self.filePath += str(fileName) + '.grd'

        # Check that directory exists
        if not os.path.exists(self.filePath):
            raise Exception('No such directory:\n%s' % self.filePath)

        # Assign None to members if file does not exist
        if gridName == 'ps':
            if not hasattr(caller, 'delta'):
                raise Exception('ERROR! The superfluid momentum ps cannot be calculated because the order parameter has not been initialized yet.')
            
            # Get grain coverage of simulation grain, to calculate min/max indices in grain
            if caller.params.has_key('outerRadius'):
                outerRadius = caller.params['outerRadius']
            else:
                outerRadius = 0.5*(1.0 - caller.params['grainMargin'])

            # Get min/max indices of grain
            if caller.params.has_key('minIndex'):
                _imin = caller.params['minIndex']
            else:
                _imin = caller.getLatticePoint(coordinate=0,
                                        sideLength=caller.params['sideLength'],
                                        Nx=caller.params['numLatticePoints'],
                                        outerRadius=outerRadius)
            if caller.params.has_key('minIndex'):
                _imax = caller.params['maxIndex']
            else:
                _imax = caller.getLatticePoint(coordinate=caller.params['sideLength'],
                                        sideLength=caller.params['sideLength'],
                                        Nx=caller.params['numLatticePoints'],
                                        outerRadius=outerRadius)

            self.y = np.zeros_like(caller.delta.phase)
            self.x = np.zeros_like(caller.delta.phase)

            # Compute superflow from phase:
            # ps = (hbar/2)*gradient(chi), which in dimensionless units is:
            # ps/(kB*Tc/vF) = gradient(chi) / 2
            #self.y[_imin:_imax+1,_imin:_imax+1], self.x[_imin:_imax+1,_imin:_imax+1] = np.gradient(0.5*caller.delta.phase[_imin:_imax+1,_imin:_imax+1])
            self.y[_imin:_imax+1,_imin:_imax+1], self.x[_imin:_imax+1,_imin:_imax+1] = \
                0.5*( caller.delta.re[_imin:_imax+1,_imin:_imax+1] * \
                        np.gradient(caller.delta.im[_imin:_imax+1,_imin:_imax+1]) - \
                      caller.delta.im[_imin:_imax+1,_imin:_imax+1] * \
                        np.gradient(caller.delta.re[_imin:_imax+1,_imin:_imax+1]) ) / \
                caller.delta.abs[_imin:_imax+1,_imin:_imax+1]**2
            self.abs = np.sqrt(self.x**2 + self.y**2)

            self.grid = np.array((np.zeros_like(self.x),
                                  np.zeros_like(self.x),
                                  np.zeros_like(self.x)))
            self.grid[0,:,:] = self.x
            self.grid[1,:,:] = self.y
            self.grid[2,:,:] = self.abs
        elif not os.path.exists(self.filePath):
            if passNoExist:
                self.filePath = None
                self.dict = None
                self.grid = None
                self.singleGrid = None
            else:
                raise Exception('No such file:\n%s' % self.filePath)
        else:
            # Read field data from binary file
            self.dict, self.grid = RF.readFieldsFromBinaryFile(self.filePath,inVerbose=verbose)

            # Read extra params if LDOS 
            if gridName == 'ldos':
                self.params = RF.readMetaDataFile(filePath + '/'*(filePath[-1]!='/') + 'ldos_params.dat')

            # Read extra params if jspectral
            if gridName == 'jspectral':
                self.params = RF.readMetaDataFile(filePath + '/'*(filePath[-1]!='/') + 'jspectral_params.dat')

            # Put field data in ndarray
            self.grid = RF.gridDataToNDArray(self.dict, self.grid, fieldName=self.name, zeroToNaN=zeroToNaN)

            # Create short-hand aliases
            if self.dict['isComplex'] == 0 and self.dict['numFields'] == 1:
                self.singleGrid = True
                self.abs = self.grid[0,:,:]
            elif self.dict['isComplex'] == 1 and self.dict['numFields'] == 1:
                self.singleGrid = False
                self.re  = self.grid[0,:,:]
                self.im  = self.grid[1,:,:]
                self.abs = self.grid[2,:,:]
                if getPhase:
                    self.phase = self.computePhase()
            elif self.dict['isComplex'] == 0 and self.dict['numFields'] == 2:
                self.singleGrid = False
                self.x   = self.grid[0,:,:]
                self.y   = self.grid[1,:,:]
                self.abs = self.grid[2,:,:]
            elif self.dict['isComplex'] == 0 and self.dict['numFields'] > 2:
                self.abs = self.grid
            elif self.dict['isComplex'] == 1 and self.dict['numFields'] == 2:
                print('WARNING! Strange shape found: %s. Multi-component OP?' % np.shape(self.grid))
                self.x    = self.grid[0,:,:]
                self.y    = self.grid[1,:,:]
                self.abs  = self.grid[2,:,:]
                self.x2   = self.grid[3,:,:]
                self.y2   = self.grid[4,:,:]
                self.abs2 = self.grid[5,:,:]
            else:
                print('WARNING! Strange shape found: %s. Multi-component OP?' % np.shape(self.grid))
                raise Exception('ERROR! Do not know what to do with isComplex=%d and numFields=%d.' %
                                (self.dict['isComplex'], self.dict['numFields']))

    # Compute phase from real and imaginary part, using atan2
    def computePhase(self):
        phase = np.zeros_like(self.abs)

        if self.singleGrid:
            return phase

        for i in range(0,self.dict['dimX']*self.dict['dimY']):
            x_i = i  % self.dict['dimX']
            y_i = i // self.dict['dimX']

            this_re = self.grid[0,y_i,x_i]
            this_im = self.grid[1,y_i,x_i]

            phase[y_i,x_i] = np.arctan2(this_im,this_re)

        return phase

# Check
def check_fields_in(fields_in, fileNames_in):
    # Input is string
    if isinstance(fields_in, str):
        fieldDict_out, fieldList_out, message_out = check_field_string(fields_in)
    # Input is a list
    elif isinstance(fields_in, list):
        fieldDict_out = {}
        fileNamesDict_out = {}
        fieldList_out = []
        message_out = None
        for f_i in range(0,len(fields_in)):
            thisField = fields_in[f_i]
            if (fileNames_in != None) and isinstance(fileNames_in, list):
                thisFileName = fileNames_in[f_i] 
            else:
                thisFileName = None

            checkDict, checkList, checkMessage = check_field_string(thisField)
            if checkDict == None:
                return None, None, None, checkMessage
            else:
                checkDictKeys = checkDict.keys()
                if len(checkDictKeys) != 1:
                    raise Exception('ERROR! Incompatible dict length from check_field_string: %s' %
                                    checkDict)
                thisObservable = checkDictKeys[0]

                # Check for duplicate
                if fieldDict_out.has_key(thisObservable):
                    if fieldDict_out[thisObservable] == checkDict[thisObservable]:
                        raise Exception('ERROR! Duplicate!\nfieldDict_out: %s\ncheckDict: %s' % 
                                        (fieldDict_out, checkDict))
                else:
                    fieldDict_out[thisObservable] = []
                    fileNamesDict_out[thisObservable] = []

                for thisField in checkDict[thisObservable]:
                    thisFullString = str('%s.%s' % (thisObservable,thisField))
                    if not thisField in fieldDict_out[thisObservable]:
                        fieldDict_out[thisObservable].append(thisField)
                        fileNamesDict_out[thisObservable].append(thisFileName)
                    if not thisFullString in fieldList_out:
                        fieldList_out.append(thisFullString)
    # Input is a dictionary
    elif isinstance(fields_in, dict):
        """
        Work in progress, do not delete!

        fieldDict_out = {}
        fieldList_out = []
        message_out = None

        # Loop over keys in dict
        for key in fields_in:
            # Check key has None or empty value, in that case return all fields
            if fields_in[key] == None or not fields_in[key]:
                checkDict, checkList, checkMessage = check_field_string(key)
            elif :

            else:
                raise Exception('Don\'t know what to do with dict key %s in dict: %s' % (key, fields_in))

            if checkDict == None:
                return None, None, checkMessage"""
        raise Exception('Dictionary not implemented for fields_in yet ...')
    else:
    # Input is something else
        fieldDict_out = None
        fieldList_out = None
        message_out = str('Incompatible type of fields_in: %s.' % (type(fields_in)))

    return fieldDict_out, fieldList_out, fileNamesDict_out, message_out

def check_field_string(fields_in):
    fieldDict_out = {}
    fieldList_out = []
    message_out = None
    # Input is string
    if isinstance(fields_in, str):
        fl = fields_in.split('.')
        # Input is just an observable
        if len(fl) == 1:
            observable = fl[0]

            #Check if observable exists
            if defaultFieldDict.has_key(observable) and fl[0]:
                fieldDict_out[observable] = defaultFieldDict[observable]
                for field in defaultFieldDict[observable]:
                    fieldList_out.append(str('%s.%s' % (observable,field)))
            else:
                fieldDict_out = None
                fieldList_out = None
                message_out = str('Observable does not exist: %s.' % (observable))
        # Input is observable.field
        elif len(fl) == 2:
            observable = fl[0]
            field = fl[1]
            # Check if observable exists
            if defaultFieldDict.has_key(observable) and fl[0]:
                # Check if field exists
                if (field in defaultFieldDict[observable]) and fl[1]:
                    fieldDict_out[observable] = [field]
                    fieldList_out.append(str('%s.%s' % (observable,field)))
                else:
                    fieldDict_out = None
                    fieldList_out = None
                    message_out = str('Field does not exist: %s.' % (field))
            else:
                fieldDict_out = None
                fieldList_out = None
                message_out = str('Observable does not exist: %s.' % (observable))
        # Input has incompatible length
        else:
            fieldDict_out = None
            fieldList_out = None
            message_out = str('Incompatible split length of fields_in: %d.' % (len(fl)))
    else:
        fieldDict_out = None
        fieldList_out = None
        message_out = str('Incompatible type of fields_in: %s.' % (type(fields_in)))

    return fieldDict_out, fieldList_out, message_out
