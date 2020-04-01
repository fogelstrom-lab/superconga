# Python Libraries
These Python Libraries contain convenient classes for data post-processing, in particular data analysis, file IO and plotting. Please note that the library is currently only tested for Python2.

## List of modules
The most important modules of the library are the following:
* `fileIO/readFiles` - Read binary "grid files" (for observables, e.g. the order parameter), metadata "parameter files" (parameters used to set up the simulation, so that it can be reproduced or re-run), and per-iteration files (e.g. free energy data files) generated from simulations with the CUDA framework. These data is conveniently stored in the form of Numpy ND-arrays and dicts.
* `dataAnalysis/dataStructs` - Uses the `fileIO/readFiles` module to create a convenient data-structure object for a simulation, where the parameters and data can be easily indexed and accessed. Also contains convenient helper functions that e.g. compute the superflow from the phase, gets the grid coordinates/indices of a coordinate given in coherence lengths. 
* `plotting/latexPlot` - Sets up parameters and runs functions that are typically done in every plotting script, in particular rcParams for latex plotting.

Other modules that might come in handy are:
* `plotting/cmapsExtra` - Custom defined colormaps, e.g. Mikael Håkanssons famous current colormap.
* `plotting/miscPlot` - Miscellaneous utilities for plotting.
* `plotting/old_streamplot` - Python has very poor capabilities of proper streamplots. It used to have better support, and here is a re-implementation of that functionality.
* `plotting/plotEnergySweeps` - Helper functions to plot the free energy while sweeping various parameters.
* `plotting/plotFields` - Helper functions for plotting fields (i.e. imshows, and grids of imshows).

## Including the modules
The modules can be included with simple import statements:
```python
from pythonLibraries.plotting import latexPlot
from pythonLibraries.fileIO import readFiles as RF
from pythonLibraries.plotting import cmapsExtra as cm
from pythonLibraries.dataAnalysis import dataStructs as DS
```
Make sure that the calling python scripts has a path to the libraries, e.g. by using the `sys` Python library
```python
import sys
sys.path.insert(0, '../../relative/path/to/tools/')
```
where `tools` is the parent directory of `pythonLibraries`.

## Example usage
Here are some examples of usage of the modules (once they have been included properly).

### Simulation data: Parameters and data at coordinates
To create a simulation data object, run for example:
```python
import sys
import numpy as np
# Load data structure module
sys.path.insert(0, '../../relative/path/to/tools/')
from pythonLibraries.dataAnalysis import dataStructs as DS
# Read simulation data from file
data = DS.runData('path/to/my/dataFolder',
                  fields_in=['delta','current'],
                  fileNames_in=['order_parameter','current_density'],
                  zeroToNaN=True)
# Print parameters
print(data.params)
currentMax = np.nanmax(data.current.abs)
print('The maximum value of the current is %s' % currentMax)
print('The x-component of the current at grid points (x,y)=(300,200) is %s' % data.current.x[200,300])
# Get order parameter phase at specific coherence-length coordinates
sideLength = data.params['numLatticePoints']
numLatticePoints = data.params['numLatticePoints']
grainPercent = 1.0 - data.params['grainMargin']
x10 = DS.getLatticePoint(10, sideLength, numLatticePoints, grainPercent)
y10 = x10
print('The phase at coherence lengths (x,y)=(10,10) is %s' % data.current.phase[y10,x10])
```

### Simulation data: Read and plot LDOS
Assume that LDOS data wants to be read and plotted, for a square grain, where the LDOS has only been saved for positive energies (under simplified conditions, e.g. particle-hole symmetry etc, negative energies can be obtained from mirroring the data over energy=0).
```python
import sys
import numpy as np
import matplotlib.pyplot as plt
# Load data structure module
sys.path.insert(0, '../../relative/path/to/tools/')
from pythonLibraries.dataAnalysis import dataStructs as DS
# Read LDOS data
data = DS.runData('path/to/my/dataFolder', fields_in=['ldos'])
# Fetch LDOS parameters
minEnergy = data.ldos.params['minEnergy']
maxEnergy = data.ldos.params['maxEnergy']
numEnergies = data.ldos.params['numEnergies']
energies = np.linspace(minEnergy,maxEnergy,numEnergies)
# Get the coordinates to the center of an edge of the grain
sideLength = data.params['numLatticePoints']
numLatticePoints = data.params['numLatticePoints']
grainPercent = 1.0 - data.params['grainMargin']
yEdge = DS.getLatticePoint(0, sideLength, numLatticePoints, grainPercent)
xCenter = DS.getLatticePoint(numLatticePoints//2, sideLength, numLatticePoints, grainPercent)
# Set up figure object
fig, ax = plt.subplots(1,1)
ax.plot(energies, data.ldos.abs[:,yEdge,xCenter],'k-')
# Mirror data over energy=0
if np.allclose(minEnergy) or minEnergy > 0:
    ax.plot(-energies, data.ldos.abs[:,yEdge,xCenter],'k-')
plt.show()
```

### Custom colormaps
To access Mikael Håkanssons characteristic colormaps for e.g. currents, it can be loaded and added into the matplotlib pipeline with the following code:
```python
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
# Load data structure module
sys.path.insert(0, '../../relative/path/to/tools/')
from pythonLibraries.plotting import cmapsExtra as cm
cm.reg_cmaps()
# Set colormap for current and order-parameter phase
currentCmap = mpl.cm.get_cmap('myred')
phaseCmap = plt.cm.get_cmap("RdBu")
# Set color to be gray outside superconducting grain (i.e. the vacuum)
currentCmap.set_bad(color='#c0c0c0',alpha=1.0)
phaseCmap.set_bad(color='#c0c0c0',alpha=1.0)
```

### Latex plotting
To set up typical plotting parameters, including latex plotting, run:
```python
import sys
# Load data structure module
sys.path.insert(0, '../../relative/path/to/tools/')
from pythonLibraries.plotting import latexPlot
# Set up plotting parameters
latexFontSize = latexPlot.latexPlot(usetex = False, fsize_in=14)
```
the variable `latexFontSize` should then be sent to any plotting routines.
