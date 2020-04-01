from __future__ import division

import os
import sys
import re
import numpy as np

# Read metadata (header) and data from binary file
def readFieldsFromBinaryFile(inFile, inVerbose=True):
    if inVerbose:
        print("Reading binary file: %s" % inFile)

    with open(inFile, "rb") as f:

        # Read data header: read numParamsHeader number of i4=32bit integers
        # There header in each .grd file currently has 32 parameters, see Parameters.h
        numParamsHeader = 32 # Number of parameters in header
        bytesPerInteger = 4 # Number of bytes per parameter
        numBytesHeader = numParamsHeader*bytesPerInteger
        intDataType  = np.dtype("i%d" % bytesPerInteger)
        header = np.fromfile(f, dtype=intDataType, count=numParamsHeader)

        # Fetch data from header array
        sizeOfDataType = header[0]
        isComplex = header[1]
        dimX = header[2]
        dimY = header[3]
        dimZ = header[4]
        numFields = header[5]
        storeParameters = header[6]
        parametersVersion = header[7]
        numBytesParameters = header[8]

        # Output the file header contents
        names = ['sz','isComplex','dimX','dimY','dimZ','numFields','storeParameters',
                 'parametersVersion','numBytesParameters']
        if inVerbose:
            for i in range(0,len(names)):
                print('%s: %s' % (names[i],header[i]))
            print('numElements: %s' % numElements)
            print('numBytesData: %s' % numBytesData)
        
        # Calculate some additional useful parameters
        numElements = numFields * dimX * dimY * int(1+isComplex)
        numBytesData = numElements * sizeOfDataType
        gridDataOffset = numBytesHeader

        if storeParameters:
            gridDataOffset += numBytesParameters

        # Save meta data in dictionary format
        headerDict = {}
        headerDict['sizeOfDataType'] = sizeOfDataType
        headerDict['isComplex'] = isComplex
        headerDict['dimX'] = dimX
        headerDict['dimY'] = dimY
        headerDict['dimZ'] = dimZ
        headerDict['numFields'] = numFields
        headerDict['storeParameters'] = storeParameters
        headerDict['parametersVersion'] = parametersVersion
        headerDict['numBytesParameters'] = numBytesParameters
        headerDict['numElements'] = numElements
        headerDict['numBytesData'] = numBytesData
        headerDict['totNumFields'] = numFields * int(1+isComplex)

        # Fetch grid data
        f.seek(gridDataOffset, os.SEEK_SET)
        floatDataType = np.dtype("f%s" % sizeOfDataType)
        data = np.fromfile(f, dtype=floatDataType, count=numElements)

    return headerDict, data

# Take grid data that has been read, and convert to numpy ND array format
def gridDataToNDArray(dict_in, data_in, slice_in=None, fieldName=None, zeroToNaN=True):
    # Arange all fields according to: [x1,y1,abs1, x2,y2,abs2, ...]

    # Check if need to add extra fields for abs()
    if dict_in['isComplex'] or dict_in['numFields'] == 2:
        allFields = dict_in['totNumFields'] + dict_in['totNumFields']//2
        additionalFields = True
    else:
        allFields = dict_in['totNumFields']
        additionalFields = False
    
    if (dict_in['numFields']%2 == 1) and (not dict_in['isComplex']) and dict_in['numFields'] > 1:
        raise Exception('ERROR! Don\'t know what to do with odd number of fields! Dict:\n%s' % dict_in)

    if dict_in['isComplex']:
        numLoops = dict_in['numFields']
    else:
        numLoops = dict_in['numFields']//2

    if additionalFields:
        data_out = np.zeros((allFields,
                             dict_in['dimY'],
                             dict_in['dimX']))
        for f_i in range(0, numLoops):
            fout1_i = f_i*3 + 0
            fout2_i = f_i*3 + 1
            fout3_i = f_i*3 + 2

            for i in range(0,dict_in['dimX']*dict_in['dimY']):
                x_i = i  % dict_in['dimX']
                y_i = i // dict_in['dimX']

                fin1_i = i + (f_i*2 + 0) * (dict_in['dimX']*dict_in['dimY'])
                fin2_i = i + (f_i*2 + 1) * (dict_in['dimX']*dict_in['dimY'])

                f1 = data_in[fin1_i]
                f2 = data_in[fin2_i]

                data_out[fout1_i,y_i,x_i] = f1
                data_out[fout2_i,y_i,x_i] = f2
                data_out[fout3_i,y_i,x_i] = np.sqrt(f1*f1 + f2*f2)
    else:
        if zeroToNaN:
            data_in[data_in == 0] = np.nan
        return data_in.reshape((dict_in['totNumFields'],dict_in['dimY'],dict_in['dimX']))

    if zeroToNaN:
        data_out[data_out == 0] = np.nan

    return data_out

# Read parameter data file
def readMetaDataFile(filePath):
    if filePath[-4:] == '.dat':
        # Filepath points to a .dat
        paramsFile = filePath
    else:
        # Filepath does not point to a .dat file. Attempt to construct full
        # path, and add '/' if path does not end with it.
        paramsFile = filePath + '/'*(filePath[-1]!='/') + 'parameters.dat'

    paramsDict = {}

    if os.path.exists(paramsFile):
        with open(paramsFile) as f:
            lines = 0
            rows = []
            for line in f:
                if line.strip():
                    rows.append(line.strip().split(' '))
                    lines += 1
        if lines > 2:
            raise Exception('Too many lines (%d) in paramsFile: %s' % (lines, paramsFile))

        for i in range(0, len(rows[1])):
            try:
                paramsDict[str(rows[0][i+1])] = float(rows[1][i])
            except ValueError:
                paramsDict[str(rows[0][i+1])] = str(rows[1][i])
        return paramsDict
    else:
        print('Could not find parameters file: %s' % paramsFile)
        return None

def readColumnsFromTextFile(inFile, inVerbose=True):
    if inVerbose:
        print("Reading text file: %s" % inFile)

    data_out = np.loadtxt(inFile, unpack=True)

    return data_out

def readConvergenceFromTextFile(inFile, inVerbose=True):
    iterations = []
    freeEnergy = []
    deltaRes = []
    
    if inVerbose:
        print("Reading text file: %s" % inFile)
    
    with open(inFile) as fin:
        for line in fin:
            thisLine = line.split()
            iterations.append( float(thisLine[0]) )
            freeEnergy.append( float(thisLine[1]) )
            deltaRes.append( float(thisLine[2]) )

    return np.array(iterations), np.array(freeEnergy), np.array(deltaRes)


# Get last free energy, residual etc from free energy file.
# Highly used function in data post-processing scripts.
def getLastFreeEnergy(filePath, fileName=None, inVerbose=False):
    if fileName != None:
        freeEnergyFile = str(filePath + ('/'*(filePath[-1]!='/')) + fileName)
    else:
        freeEnergyFile = filePath

    if inVerbose:
        print("filePath:\n%s" % filePath)
        print("*"*40)

    dict_out = {}
    header = {}

    with open(freeEnergyFile) as f:
        lines = 0
        for line in f:
            if line.strip():
                row = line.strip().split(' ')

                if row[0] == '#' and lines == 0:
                    if inVerbose:
                        print('Header: %s' % line.strip())

                    for i in range(1,len(row)):
                        header[i-1] = str(row[i])
                        dict_out[str(row[i])] = None
                elif lines == 0 and row[0] != '#':
                    raise Exception('No header in free energy file:\n%s' % freeEnergyFile)
                else:
                    lastRow = row
                lines += 1

        for i in range(0,len(lastRow)):
            dict_out[header[i]] = lastRow[i]

    return dict_out

# Parse fullPath and return all free energies in a dictionary
def parseEnergySweeps(fullBasePath, B0_filter='all', inVerbose=False):
    if inVerbose:
        print("parseEnergySweeps(): fullBasePath:\n%s" % fullBasePath)
        print("*"*40)
    
    dict_iterFiles = {}

    runs = 0

    ##################################################
    #         Extract by parsing directories         #
    ##################################################
    for root, dirs, files in os.walk(fullBasePath):
        if not dirs:
            runName = root.replace(fullBasePath, "")
            runPath = root + "/"

            # Extract variables from name
            run_wnum = ""
            run_B0   = ""
            run_BI   = ""
            wnum_search = re.search('^w(.+?)_B', runName)
            B0_search   = re.search('B(.+?)_BI', runName)
            BI_search   = re.search('BI(.+?)$', runName)
            if wnum_search and B0_search and BI_search:
                run_wnum = int(wnum_search.group(1))
                run_B0   = float(B0_search.group(1))
                run_BI   = float(BI_search.group(1))
                run_B    = (float(run_B0) * float(run_wnum)) + float(run_BI)
            else:
                raise Exception('Couldn\'t extract data from: %s' % root)
            
            if B0_filter == 'all' or np.allclose(float(B0_filter), float(run_B0)):
                runs += 1
                
                if not (run_wnum in dict_iterFiles):
                    dict_iterFiles[run_wnum] = {'B0': float(run_B0),
                                                'energies': [],
                                                'bvalues': [],
                                                'residuals': [],
                                                'runs': []}

                # Extract free energy for all iterations
                last_energy = 0.0
                last_residual = 1e-1
                last_run = 0
                with open(runPath + "freeEnergy.dat") as f:
                    lines = 0
                    for line in f:
                        if line.strip():
                            if line[0] != '#':
                                lines += 1
                                row = line.strip().split(' ')
                                last_energy = float(row[2])
                                last_residual = float(row[1])
                                last_run = int(row[0])

                # Append final energies
                dict_iterFiles[run_wnum]['energies'].append(last_energy)
                dict_iterFiles[run_wnum]['bvalues'].append(run_B)
                dict_iterFiles[run_wnum]['residuals'].append(last_residual)
                dict_iterFiles[run_wnum]['runs'].append(last_run)

                if inVerbose:
                    print("%d:\t%s\t" % (runs, runName)),
                    print("w='%s', B0='%s', BI='%s', B='%s'" % (run_wnum, run_B0, run_BI, run_B)),
                    print("\t#%d, res=%s, e=%s" % (last_run, last_residual, last_energy))

    if inVerbose:
        print("*"*40)
        print("parseEnergySweeps(): Found %d runs." % runs)

    return dict_iterFiles

# Read Ozaki energies generated by Ozaki energy script
def readOzakiDatFile(file_in, inVerbose=False):
    dict_out = {}
    header = {}

    with open(file_in) as f:
        lines = 0
        for line in f:
            if line.strip():
                row = line.strip().split(' ')

                if row[0] == '#' and lines == 0:
                    if inVerbose:
                        print('Header: %s' % line.strip())

                    for i in range(1,len(row)):
                        header[i-1] = str(row[i])
                        dict_out[str(row[i])] = []
                elif lines == 0 and row[0] != '#':
                    raise Exception('No header in free energy file:\n%s' % file_in)
                else:
                    for i in range(0, len(row)):
                        dict_out[header[i]].append(float(row[i]))
                lines += 1
    return header, dict_out
