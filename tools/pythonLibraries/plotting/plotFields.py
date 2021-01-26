from __future__ import division

import os
import re
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Cursor, Button
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pythonLibraries.fileIO import readFiles as RF
from pythonLibraries.dataAnalysis import dataStructs as DS

# Set up RC params for latex plotting
from matplotlib import rcParams
plt.rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
plt.rc('text', usetex=True)
#rcParams['font.family'] = 'sans-serif'
#rcParams['font.serif'] = ['Computer Modern Roman']
#rcParams['text.usetex'] = True

# Register new cmaps
import cmapsExtra as cm
cm.reg_cmaps()

# Plotting parameters
fontsize = 12
fillBetweenColor = u'#c0c0c0'

plotParams = {
'font_size':        16,
'tick_size':        15,
'cbar_size':        15,
'cmap_in':          'jet',
'fillBetweenColor': u'#c0c0c0',
'outerRadius':      0.48,
'x_label':          r'$x/\xi_0$',
'y_label':          r'$y/\xi_0$',
'cbar_label':
    {
    'delta.re':     r'$\mathrm{Re}\left(\Delta\right)/k_BT_c$',
    'delta.im':     r'$\mathrm{Im}\left(\Delta\right)/k_BT_c$',
    'delta.abs':    r'$\left|\Delta\right|/k_BT_c$',
    'delta.phase':  r'$\mathrm{Arg}\left(\Delta\right)$',
    'current.x':    r'$j_x/j_d$',
    'current.y':    r'$j_y/j_d$',
    'current.abs':  r'$\left|j\right|/j_d$',
    'B.abs':        r'$\Phi\cdot 10^{-5}/\Phi_0$',
    'ldos.abs':     r'$N/N_{F}$',
    },
'cmaps':
    {
    'delta.re':     'seismic',
    'delta.im':     'seismic',
    'delta.abs':    'jet',
    'delta.phase':  'myredblue',
    'current.x':    'seismic',
    'current.y':    'seismic',
    'current.abs':  'myred',
    'B.abs':        'PuOr',
    'ldos.abs':     'myred4',
    },
'norm_midpoint':
    {
    'delta.re':     None,
    'delta.im':     None,
    'delta.abs':    None,
    'delta.phase':  None,
    'current.x':    0,
    'current.y':    0,
    'current.abs':  None,
    'B.abs':        0,
    'ldos.abs':     None,
    },
'cbar_labelpad': 10,
'dpi': 60,
'wspace': 0,
'hspace': 0,
}

defaultSideLengths = {'3.3': 20.0, '5.0': 30.0, '6.6': 40.0, '8.3': 50.0, '9.9': 60.0, '9.947': 60.0}

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def reverse_colourmap(cmap, name = 'my_cmap_r'):
    reverse = []
    k = []   

    for key in cmap._segmentdata:    
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:                    
            data.append((1-t[0],t[2],t[1]))            
        reverse.append(sorted(data))    

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL) 
    return my_cmap_r

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class TwoInnerPointsNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, up=None, clip=False, ratios=None):
        self.low = low
        self.up = up
        self.ratios = ratios
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.up, self.vmax], self.ratios
        return np.ma.masked_array(np.interp(value, x, y))

class ThreeInnerPointsNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, low=None, mid=None, up=None, clip=False, ratios=None):
        self.low = low
        self.mid = mid
        self.up = up
        self.ratios = ratios
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.low, self.mid, self.up, self.vmax], self.ratios
        return np.ma.masked_array(np.interp(value, x, y))

def plotSingleGrid(grid_in,
                   fieldName_in=None,
                   outerRadius=plotParams['outerRadius'],
                   sideLength=None, scale=1.0,
                   sideLengthOverride=False,
                   norm_in=None,
                   vmin_in=None, vmax_in=None,
                   zeroToNaN=False,
                   fillOutside=True,
                   ax_title=None, x_label=None, y_label=None, y_label2=None,
                   plot_cbar=True, cbar_label=None,
                   cmap_in=plotParams['cmap_in'],
                   f_in=None,
                   ax_in=None,
                   font_size=plotParams['font_size'],
                   tick_size=plotParams['tick_size'],
                   cbar_size=plotParams['cbar_size']):
    # Check that grid_in is actually a grid
    try:
        dim = grid_in.ndim
    except AttributeError:
        dim = 0

    if zeroToNaN:
        grid_in[grid_in == 0] = np.nan

    # Check that grid is actually 2D
    if dim == 0 or dim > 3:
        raise Exception('Don\'t know what to do with dim-%d object' % (dim))
    elif dim == 3:
        if np.shape(grid_in)[0] > 1:
            raise Exception('Don\'t know what to do with object with shape=%s' % str(np.shape(grid_in)))
        else:
            dataMasked = np.ma.array(grid_in[0,:,:]/scale, mask=np.isnan(grid_in[0,:,:]))
    elif dim == 2:
        dataMasked = np.ma.array(grid_in/scale, mask=np.isnan(grid_in))

    if sideLength == None or sideLengthOverride:
        b = 0.0
        xmin = 0
        xmax = np.shape(dataMasked)[0]
        ymin = 0
        ymax = np.shape(dataMasked)[1]
        xLength = np.shape(dataMasked)[0]
        yLength = np.shape(dataMasked)[1]
    else:
        b = (sideLength/(2.0*outerRadius) - sideLength) * 0.5
        xmin = -b
        xmax = sideLength+b
        ymin = -b
        ymax = sideLength+b
        xLength = sideLength
        yLength = sideLength

    # Generate axis
    if ax_in == None:
        f, (ax) = plt.subplots(1,sharex=False,sharey=False,dpi=60)
    else:
        f = f_in
        ax = ax_in

    cmap = mpl.cm.get_cmap(cmap_in)#'myred4')
    cmap.set_bad(plotParams['fillBetweenColor'],1.0)

    if norm_in != None and isinstance(norm_in,(int,long,float)):
        norm = MidpointNormalize(midpoint=norm_in,vmin=vmin_in,vmax=vmax_in)
    else:
        norm = None

    cax = ax.imshow(dataMasked, cmap=cmap,
                    norm=norm,
                    vmin=vmin_in, vmax=vmax_in,
                    extent=(xmin,xmax,ymin,ymax),
                    interpolation='none',#'bilinear',
                    aspect=1.0,
                    origin='lower')#,aspect=1.0)#'auto')

    if ax_title != None:
        ax.set_title(ax_title,fontsize=font_size)
    if x_label != None:
        ax.set_xlabel(x_label,fontsize=font_size)
    if y_label != None:
        ax.set_ylabel(y_label,fontsize=font_size)
    if y_label2 != None:
        ax.annotate(y_label2, xy=(1.075,0.5), rotation=90, fontsize=font_size,
                    ha='center',va='center',xycoords='axes fraction')
    ax.tick_params( axis='both', which='major', labelsize=tick_size )

    if fillOutside:
        ax.fill_between([-b,xLength+b],y1=-b,y2=0,color=plotParams['fillBetweenColor'])
        ax.fill_between([-b,xLength+b],y1=yLength+b,y2=yLength,color=plotParams['fillBetweenColor'])

        ax.fill_between([-b,0],y1=yLength+b,y2=-b,color=plotParams['fillBetweenColor'])
        ax.fill_between([xLength,xLength+b],y1=yLength+b,y2=-b,color=plotParams['fillBetweenColor'])

    if plot_cbar:
        cbar_ax,kw = mpl.colorbar.make_axes(ax)#, "top")
        cbar_out = f.colorbar(cax, ax=ax, cax=cbar_ax)#, orientation='horizontal')
        #cbar_out.ax.xaxis.set_ticks_position('top')
        cbar_out.solids.set_edgecolor("face")
        cbar_out.ax.tick_params( axis='both', which='major', labelsize=cbar_size)
        if cbar_label != None:
            cbar_out.set_label(cbar_label,rotation=90,labelpad=plotParams['cbar_labelpad'],fontsize=font_size)

    #ax.set_xlim([xmin,xmax])
    #ax.set_ylim([ymin,ymax])
    ax.set_adjustable('box-forced')
    ax.autoscale(False)

    #deltaAx.set_xlim([0,deltaDict['dimX']])
    #deltaAx.set_ylim([0,deltaDict['dimY']])

    #f.text(0.5, 0.03, s=r'$x/\xi_0$', ha='center', fontsize=fontsize)
    #f.text(0.02, 0.5, s=r'$y/\xi_0$',  va='center', fontsize=fontsize, rotation=90)
    #f.subplots_adjust(hspace=0,wspace=0)
    #f.tight_layout()

    return f, ax

def plotGrid(pathList,
             fieldList,
             basePath=None,
             x_titles=None,
             y_titles=None,
             sharex_in=True,
             sharey_in=True,
             normalizeGrid=True,
             dpi_in=plotParams['dpi'],
             wspace_in=plotParams['wspace'],
             hspace_in=plotParams['hspace'],
             savePath_in=None,
             figList_in=None,
             axGrids_in=None,
             figSize_in=None,
             *args,
             **kwargs
            ):

    numPlotsY = np.shape(pathList)[0]
    numPlotsX = np.shape(pathList)[1]
    numPlots = numPlotsX*numPlotsY

    runs = {}
    figs = {}
    ax_grids = {}
    maxValues = {}
    minValues = {}

    # Append basePath if set
    if basePath != None:
        for y_i in range(0,numPlotsY):
            for x_i in range(0,numPlotsX):
                pathList[y_i][x_i] = ('%s%s%s' %
                                      (basePath,
                                       str('/'*(basePath[-1]!='/')),
                                       pathList[y_i][x_i]))

    for f_i in range(0,len(fieldList)):
        field = fieldList[f_i]
        if field == 'B' or field == 'b' or field == 'b.abs':
            fieldList[f_i] = 'B.abs'

        if not fieldList[f_i] in plotParams['cbar_label']:
            raise Exception('Unknown field: %s' % field)

    # Get max and min values for each field
    for y_i in range(0,numPlotsY):
        runs[y_i] = {}
        for x_i in range(0,numPlotsX):
            runPath = pathList[y_i][x_i]

            if not os.path.exists(runPath):
                raise Exception('No such path:\n%s' % runPath)

            runs[y_i][x_i] = DS.runData(runPath,fields_in=fieldList)

            """print('%s\t%s' % (np.nanmin(runs[y_i][x_i].current.abs),
                              np.nanmax(runs[y_i][x_i].current.abs) )),

            print('\t%s\t%s' % (np.nanmin(runs[y_i][x_i].delta.im),
                              np.nanmax(runs[y_i][x_i].delta.im) ))"""

            # Find out max value for all fields
            for f_i in range(0, len(fieldList)):
                thisField = fieldList[f_i]
                if normalizeGrid:
                    observable, field = thisField.split('.')
                    thisMax = runs[y_i][x_i].getFieldMax(observable,field)
                    thisMin = runs[y_i][x_i].getFieldMin(observable,field)
                    if not (thisField in maxValues):
                        maxValues[thisField] = thisMax
                    else:
                        maxValues[thisField] = max(thisMax,maxValues[thisField])
                    if not (thisField in minValues):
                        minValues[thisField] = thisMin
                    else:
                        minValues[thisField] = min(thisMin,minValues[thisField])
                else:
                    minValues[thisField] = None
                    maxValues[thisField] = None
    
    """print maxValues['current.abs']
    print minValues['current.abs']
    print maxValues['delta.im']
    print minValues['delta.im']"""

    for f_i in range(0,len(fieldList)):
        figsInner_out = []
        axesInner_out = []

        fieldName = fieldList[f_i]
        observable, field = fieldName.split('.')

        newFig = True

        if figList_in != None and axGrids_in != None:
            if (isinstance(figList_in,list)) and (len(fieldList) == len(figList_in)) and \
               (isinstance(axGrids_in,list)) and (len(fieldList) == len(axGrids_in)):
                f = figList_in[f_i]
                ax_grid = axGrids_in[f_i]
                newFig = False
        
        if newFig:
            f, ax_grid = plt.subplots(numPlotsY,numPlotsX,sharex=sharex_in,sharey=sharey_in,
                                      dpi=dpi_in,squeeze=False, figsize=figSize_in)

        figs[f_i] = f
        ax_grids[f_i] = ax_grid

        for y_i in range(0,numPlotsY):
            for x_i in range(0,numPlotsX):
                if runs[y_i][x_i].params != None:
                    outerRadius = runs[y_i][x_i].params['outerRadius']
                    if 'innerRadius' in runs[y_i][x_i].params:
                        innerRadius = runs[y_i][x_i].params['innerRadius']
                    else:
                        print('Warning, run does not have parameter "innerRadius" set. Setting it to 0.35.')
                        #print('Run params: %s' % (runs[y_i][x_i].params))
                        innerRadius = 0.35
                    sideLength = runs[y_i][x_i].params['R']
                else:
                    searchSideLength = re.search('/R(\d+\.\d+)/', pathList[y_i][x_i])
                    if searchSideLength:
                        extractedSideLenth = searchSideLength.group(1)
                        if extractedSideLenth in defaultSideLengths:
                            sideLength = defaultSideLengths[extractedSideLenth]
                        else:
                            sideLength = None
                    else:
                        sideLength = None
                    outerRadius = plotParams['outerRadius']
                    innerRadius = None
                
                ax = ax_grid[y_i,x_i]

                if x_i == 0 and sideLength != None:
                    y_label = plotParams['y_label']
                else:
                    y_label = None

                if (y_i == numPlotsY-1) and sideLength != None:
                    x_label = plotParams['x_label']
                else:
                    x_label = None

                if (y_i == 0 and x_titles != None):
                    ax_title = x_titles[x_i]
                else:
                    ax_title = None

                if (x_i == numPlotsX-1 and y_titles != None):
                    y_label2 = y_titles[y_i]
                else:
                    y_label2 = None
                #ax.yaxis.set_major_locator(MaxNLocator(prune='upper'))

                cax_out = plotSingleGrid(runs[y_i][x_i].getObservableProperty(observable, field),
                                         fieldName,
                                         outerRadius=outerRadius,
                                         sideLength=sideLength,
                                         x_label=x_label,
                                         y_label=y_label,
                                         ax_title=ax_title,
                                         y_label2=y_label2,
                                         plot_cbar=False,
                                         cbar_label=plotParams['cbar_label'][fieldName],
                                         cmap_in=plotParams['cmaps'][fieldName],
                                         f_in=f,
                                         ax_in=ax,
                                         norm_in = plotParams['norm_midpoint'][fieldName],
                                         vmin_in = minValues[fieldName],
                                         vmax_in = maxValues[fieldName],
                                         *args,
                                         **kwargs
                                        )

        f.subplots_adjust(wspace=wspace_in,hspace=hspace_in,
                          left=0.05, right=1.00, bottom=0.125, top=0.925)
        cbar_ax,kw = mpl.colorbar.make_axes([ax for ax in ax_grid.flat])#[r_i]])
        cbar_out = f.colorbar(cax_out, cax=cbar_ax)#, orientation='horizontal')
        cbar_out.ax.tick_params( axis='both', which='major', labelsize=plotParams['cbar_size'])
        cbar_out.set_label(plotParams['cbar_label'][fieldName],rotation=90,
                           labelpad=plotParams['cbar_labelpad'],
                           fontsize=plotParams['font_size'])
        cbar_out.solids.set_edgecolor("face")


        if savePath_in != None:
            dirName = os.path.dirname(savePath_in)
            if not os.path.isdir(dirName):
                os.makedirs(dirName)
            savePath = str('%s_%s.pdf' % (savePath_in, fieldName.replace('.','_'))) 
            plt.savefig(savePath)#, bbox_inches='tight')
            print('Saved file:\n%s' % savePath)
    return figs, ax_grids
"""
def plotSingleGrid(grid_in,
                   fieldName_in=None,
                   outerRadius=plotParams['outerRadius'],
                   sideLength=None, scale=1.0,
                   vmin_in=None, vmax_in=None,
                   fillOutside=True,
                   ax_title=None, x_label=None, y_label=None,
                   plot_cbar=True, cbar_label=None,
                   cmap_in=plotParams['cmap_in'],
                   f_in=None, ax_in=None,
                   font_size=plotParams['font_size'],
                   tick_size=plotParams['tick_size'],
                   cbar_size=plotParams['cbar_size']):  """


def plotSweepGrid(sweepPath,
                  fieldList,
                  wnums_in='all', bvalues_in='all', savePath_in=None,
                  *args, **kwargs):

    if not os.path.isdir(sweepPath):
        raise Exception('No such directory:\n%s' % sweepPath)

    sweep_dict = RF.parseEnergySweeps(sweepPath)
    sweep_wnums = sorted(sweep_dict.keys())
    sweep_bvalues = np.array(sorted(sweep_dict[sweep_wnums[0]]['bvalues'])) - \
                    sweep_wnums[0]*sweep_dict[sweep_wnums[0]]['B0']

    if wnums_in == 'all':
        wnums = sweep_wnums
    elif isinstance(wnums_in, int):
        wnums = [wnums_in]
    elif isinstance(wnums_in, list):
        wnums = wnums_in
    else:
        raise Exception('Unknown wnums_in: %s' % wnums_in)

    for wnum in wnums:
        if not (wnum in sweep_dict):
            raise Exception('Unknown wnum=%d in sweep_dict, has keys: %s' % (wnum, sweep_wnums))

    if bvalues_in == 'all':
        bvalues = sweep_bvalues
    elif isinstance(bvalues_in, int):
        if bvalues_in == 0:
            bvalues = [0]
        elif bvalues_in == 3 and len(sweep_bvalues) >= 3:
            bvalues = [sweep_bvalues[0],
                       sweep_bvalues[int(np.floor(float(len(sweep_bvalues))/float(2.0)))],
                       sweep_bvalues[-1]]
        else:
            raise Exception('Don\'t know what to do with bvalues_in=%d and bvalues=%s' %
                            (bvalues_in, sweep_bvalues) )
    elif isinstance(bvalues_in, list):
        bvalues = bvalues_in
    else:
        raise Exception('Unknown wnums_in: %s' % wnums_in)

    runPaths = []
    x_titles = {}
    y_titles = {}

    # Fetch run objects
    for w_i in range(0,len(wnums)):
        wnum = wnums[w_i]
        B0 = sweep_dict[wnum]['B0']
        print B0
        innerRunPaths = []
        y_titles[w_i] = r'$\mathrm{w}=%d$' % wnum
        for b_i in range(0,len(bvalues)):
            bvalue = bvalues[b_i]
            runPath = '%s%sw%d_B%2.3f_BI%2.3f' % (sweepPath, str('/'*(sweepPath[-1]!='/')),
                                                  wnum, B0, bvalue)
            innerRunPaths.append(runPath)
            x_titles[b_i] = r'$\delta=%2.2f$' % (bvalue/B0)
        
        runPaths.append(innerRunPaths)

    plotGrid(runPaths, fieldList,
             x_titles=x_titles, y_titles=y_titles, savePath_in=savePath_in,
             *args, **kwargs)


def plotDelta(deltaDict, deltaData, coherenceLengths=None):

    deltaNumFields = deltaDict['totNumFields'] + deltaDict['numFields']
    numGrids = 3 # Re, Im, Abs
    plotNumY = deltaDict['numFields']
    outerRadius = 0.48

    if coherenceLengths == None:
        b = 0.0
        xmin = 0
        xmax = deltaDict['dimX']
    else:
        b = (coherenceLengths/(2.0*outerRadius) - coherenceLengths) * 0.5
        xmin = -b
        xmax = coherenceLengths+b

    deltaGrid = np.zeros(( deltaNumFields,
                       deltaDict['dimX'],
                       deltaDict['dimY']))
    for i in range(0,len(deltaData)):
        # ijk
        # elements in one field: dimX * dimY
        f_i0 = i//(deltaDict['dimX']*deltaDict['dimY'])
        f_i = f_i0%2 + 3*(f_i0//2)
        x_i = i%deltaDict['dimX']
        y_i = (i%(deltaDict['dimX']*deltaDict['dimY']))//deltaDict['dimX']

        #print("%s\t%s\t%s\t%s" % (i,f_i, x_i, y_i))
        deltaGrid[f_i,y_i,x_i] = deltaData[i]

    """if deltaDict['numFields'] == 2:
        delta_PM_grid = np.zeros(( deltaNumFields,
                                   deltaDict['dimX'],
                                   deltaDict['dimY']))
        delta_PM_grid[0,:,:] = deltaGrid[0,:,:]-deltaGrid[4,:,:]
        delta_PM_grid[1,:,:] = deltaGrid[1,:,:]+deltaGrid[3,:,:]
        delta_PM_grid[3,:,:] = deltaGrid[0,:,:]+deltaGrid[4,:,:]
        delta_PM_grid[4,:,:] = deltaGrid[1,:,:]-deltaGrid[3,:,:]"""



    for f_i in range(0,deltaDict['numFields']):
        for i in range(0,deltaDict['dimX']*deltaDict['dimY']):
            x_i = i%deltaDict['dimX']
            y_i = i//deltaDict['dimX']
            deltaGrid[2+f_i*numGrids,y_i,x_i] = np.sqrt(deltaGrid[f_i*numGrids,y_i,x_i]**2 +
                                                        deltaGrid[1+f_i*numGrids,y_i,x_i]**2)
            #delta_z[x_i,y_i] = deltaGrid[2,y_i,x_i]
            """if deltaDict['numFields'] == 2:
                delta_PM_grid[2+f_i*numGrids,y_i,x_i] = np.sqrt(delta_PM_grid[f_i*numGrids,y_i,x_i]**2 +
                                                            delta_PM_grid[1+f_i*numGrids,y_i,x_i]**2)"""

    #cmapDelta = 'jet'
    cmapDelta = mpl.cm.get_cmap('jet')#'myred4')
    cmapDelta.set_bad(plotParams['fillBetweenColor'],1.0)
    deltaGrid[deltaGrid == 0] = np.nan
    f, ax_grid = plt.subplots(plotNumY,numGrids,squeeze=False,sharex=True,sharey=True)
    """
    plt.register_cmap(name='myblue', data=cdictb) # optional lut kwarg
    plt.register_cmap(name='myred', data=cdictr1) # optional lut kwarg
    plt.register_cmap(name='myred2', data=cdictr2) # optional lut kwarg
    plt.register_cmap(name='myred3', data=cdictr2a) # optional lut kwarg
    plt.register_cmap(name='myred4', data=cdictr2b) # optional lut kwarg
    plt.register_cmap(name='myredblue', data=cdictr2c) # optional lut kwarg
    plt.register_cmap(name='myredblue2', data=cdictr4) # optional lut kwarg"""

    """for i in range(0, plotNumY)
        # Plots for real and imaginary parts
        f1, ax1 = plt.subplots(2,1,squeeze=False,sharex=True,sharey=True)
        # Plots for absolute value 
        f2, ax2 = plt.subplots(1,1,squeeze=False,sharex=True,sharey=True)



        deltaAx.imshow(np.ma.array(deltaGrid[2+numGrids*i,:,:],
                                   mask=np.isnan(deltaGrid[2+numGrids*i,:,:])),
                       cmap=cmapDelta,
                       extent=(xmin,xmax,xmin,xmax),
                       interpolation='bilinear',aspect=1.0)

    for i in range(0, plotNumY)
        ax2 = ax_grid2[0,i]
        deltaAx.imshow(np.ma.array(deltaGrid[2+numGrids*i,:,:],
                                   mask=np.isnan(deltaGrid[2+numGrids*i,:,:])),
                       cmap=cmapDelta,
                       extent=(xmin,xmax,xmin,xmax),
                       interpolation='bilinear',aspect=1.0)"""


    for i in range(0,deltaNumFields):
        deltaAx = ax_grid[i//numGrids,i%numGrids]
        dataMasked = np.ma.array (deltaGrid[i,:,:], mask=np.isnan(deltaGrid[i,:,:]))
        #deltaAx.pcolormesh(dataMasked,cmap=cmapDelta)
        deltaAx.imshow(dataMasked,cmap=cmapDelta,
                       extent=(xmin,xmax,xmin,xmax),
                       interpolation='bicubic',aspect=1.0)#,aspect=1.0)#'auto')
        deltaAx.set_adjustable('box-forced')
        deltaAx.autoscale(False)
        #deltaAx.set_xlim([0,deltaDict['dimX']])
        #deltaAx.set_ylim([0,deltaDict['dimY']])

    f.text(0.5, 0.03, s=plotParams['x_label'], ha='center', fontsize=plotParams['font_size'])
    f.text(0.02, 0.5, s=plotParams['y_label'],  va='center', fontsize=plotParams['font_size'], rotation=90)
    #f.subplots_adjust(hspace=0,wspace=0)
    f.tight_layout()
    


def plotCurrent(currentDict, currentData):

    currentNumFields = currentDict['totNumFields'] + 1
    plotNumX = 3
    plotNumY = 1

    currentGrid = np.zeros(( currentNumFields,
                             currentDict['dimX'],
                             currentDict['dimY']))
    for i in range(0,len(currentData)):
        # ijk
        # elements in one field: dimX * dimY
        f_i = i//(currentDict['dimX']*currentDict['dimY'])
        x_i = i%currentDict['dimX']
        y_i = (i%(currentDict['dimX']*currentDict['dimY']))//currentDict['dimX']

        #print("%s\t%s\t%s\t%s" % (i,f_i, x_i, y_i))
        currentGrid[f_i,y_i,x_i] = currentData[i]

    for i in range(0,currentDict['dimX']*currentDict['dimY']):
        x_i = i%currentDict['dimX']
        y_i = i//currentDict['dimX']
        currentGrid[2,y_i,x_i] = np.sqrt(currentGrid[0,y_i,x_i]**2 + currentGrid[1,y_i,x_i]**2)


    f, ax_grid = plt.subplots(plotNumY,plotNumX,squeeze=False,sharex=True,sharey=True)

    cmapCurrent = 'jet'
    cmapCurrent = mpl.cm.get_cmap(cmapCurrent)
    cmapCurrent.set_bad(u'#CFD0D2',1.0)
    currentGrid[currentGrid == 0] = np.nan

    for i in range(0,currentNumFields):
        currentAx = ax_grid[0,i]
        dataMasked = np.ma.array (currentGrid[i,:,:], mask=np.isnan(currentGrid[i,:,:]))
        #currentAx.pcolormesh(dataMasked,cmap=cmapCurrent)
        currentAx.imshow(dataMasked,cmap=cmapCurrent,interpolation='bicubic',aspect=1.0)
        currentAx.set_adjustable('box-forced')
        currentAx.autoscale(False)
        currentAx.set_xlim([0,currentDict['dimX']])
        currentAx.set_ylim([0,currentDict['dimY']])

    f.text(0.5, 0.91, s=plotParams['x_label'], ha='center', fontsize=plotParams['font_size'])
    f.text(0.02, 0.5, s=plotParams['y_label'],  va='center', fontsize=plotParams['font_size'], rotation=90)
    f.subplots_adjust(hspace=0,wspace=0)
    f.tight_layout()

def plotB(BDict, BData):
    plotNumXB = 1
    plotNumYB = 1
    BNumPlots = plotNumXB*plotNumYB
    BGrid = np.zeros(( BNumPlots,
                       BDict['dimX'],
                       BDict['dimY']))
    BFields = range(0,BNumPlots)
    for f_i in range(0,BNumPlots):
        lf_i = BFields[f_i]
        for i in range(0,BDict['dimX']*BDict['dimY']):
            x_i = i%BDict['dimX']
            y_i = i//BDict['dimX']
            BGrid[f_i,y_i,x_i] = BData[i+lf_i*BDict['dimX']*BDict['dimY']]

    cmapB = 'jet'
    cmapB = mpl.cm.get_cmap(cmapB)
    cmapB.set_bad(u'#CFD0D2',1.0)
    #BGrid[BGrid == 0] = np.nan

    f_B, ax_grid_B = plt.subplots(plotNumYB,plotNumXB,sharex=True,sharey=True,squeeze=False)
    for i in range(0,BNumPlots):
        BAx = ax_grid_B[i//plotNumXB,i%plotNumXB]
        dataMasked = np.ma.array (BGrid[i,:,:], mask=np.isnan(BGrid[i,:,:]))
        #BAx.pcolormesh(dataMasked,cmap=cmapB)
        Bim = BAx.imshow(dataMasked,cmap=cmapB,interpolation='bicubic',aspect='auto')

        BAx.set_title(plotParams['cbar_label']['B.abs'])
        BAxdivider = make_axes_locatable(BAx)
        BAxC = BAxdivider.append_axes("right", size="20%", pad=0.05)
        Bcbar = plt.colorbar(Bim, cax=BAxC)

        BAx.set_adjustable('box-forced')
        BAx.autoscale(False)
        BAx.set_xlim([0,BDict['dimX']])
        BAx.set_ylim([0,BDict['dimY']])

    f_B.text(0.5, 0.91, s=plotParams['x_label'], ha='center', fontsize=plotParams['font_size'])
    f_B.text(0.02, 0.5, s=plotParams['y_label'],  va='center', fontsize=plotParams['font_size'], rotation=90)
    f_B.subplots_adjust(hspace=0,wspace=0)
    f_B.tight_layout()

class viewer_2d(object):
    def __init__(self, dataDict, data, field=0, scaling=1.0): #z,x=None, y=None): #
        """
        Shows a given array in a 2d-viewer.
        Input: dataDict, runData dictionary
        data, z field
        """

        if dataDict['totNumFields'] == 2 and field == 2:
            totNumFields = 3
        elif dataDict['totNumFields'] > field:
            totNumFields = dataDict['totNumFields']
        else:
            raise Exception('Problem in viewer_2d: dataDict[totnumFields]=%d,\tfield=%d' %
                             (dataDict['totNumFields'], field))

        dataGrid = np.zeros(( totNumFields,
                              dataDict['dimX'],
                              dataDict['dimY']))

        for i in range(0,len(data)):
            # ijk
            # elements in one field: dimX * dimY
            f_i = i//(dataDict['dimX']*dataDict['dimY'])
            x_i = i%dataDict['dimX']
            y_i = (i%(dataDict['dimX']*dataDict['dimY']))//dataDict['dimX']

            #print("%s\t%s\t%s\t%s" % (i,f_i, x_i, y_i))
            dataGrid[f_i,y_i,x_i] = data[i]/scaling

        if totNumFields > dataDict['totNumFields']:
            for i in range(0,dataDict['dimX']*dataDict['dimY']):
                x_i = i%dataDict['dimX']
                y_i = i//dataDict['dimX']
                dataGrid[2,y_i,x_i] = np.sqrt(dataGrid[0,y_i,x_i]**2 +
                                              dataGrid[1,y_i,x_i]**2)/scaling

        dataGrid[dataGrid == 0] = np.nan

        x = np.arange(dataDict['dimX'])
        y = np.arange(dataDict['dimY'])
        z= np.ma.array(dataGrid[field,:,:], mask=np.isnan(dataGrid[field,:,:]))

        if x==None:
            self.x=np.arange(z.shape[0])
        else:
            self.x=x
        if y==None:
            self.y=np.arange(z.shape[1])
        else:
            self.y=y
        self.z=z
        self.fig=plt.figure()
        #Doing some layout with subplots:
        self.fig.subplots_adjust(0.05,0.05,0.98,0.98,0.1)
        self.overview=plt.subplot2grid((8,4),(0,0),rowspan=7,colspan=2)
        #self.overview.pcolormesh(self.x,self.y,self.z)
        self.overview.imshow(self.z,interpolation='bicubic',aspect='auto',
                             extent=[self.x[0],self.x[-1],
                                     self.y[0],self.y[-1]])
        self.overview.autoscale(1,'both',1)
        self.x_subplot=plt.subplot2grid((8,4),(0,2),rowspan=4,colspan=2)
        self.y_subplot=plt.subplot2grid((8,4),(4,2),rowspan=4,colspan=2)

        #Adding widgets, to not be gc'ed, they are put in a list:
        
        cursor=Cursor(self.overview, useblit=True, color='black', linewidth=2 )
        but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
        reset_button=Button(but_ax,'Reset')
        but_ax2=plt.subplot2grid((8,4),(7,1),colspan=1)
        legend_button=Button(but_ax2,'Legend')
        self._widgets=[cursor,reset_button,legend_button]
        #connect events
        reset_button.on_clicked(self.clear_xy_subplots)
        legend_button.on_clicked(self.show_legend)
        self.fig.canvas.mpl_connect('button_press_event',self.click)
    
    def show_legend(self, event):        
        """Shows legend for the plots"""
        for pl in [self.x_subplot,self.y_subplot]:
            if len(pl.lines) > 0:
                pl.legend()
        plt.draw()

    def clear_xy_subplots(self,event):
        """Clears the subplots."""
        for j in [self.overview,self.x_subplot,self.y_subplot]:
            j.lines=[]
            j.legend_ = None
        plt.draw()


    def click(self,event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        if event.inaxes==self.overview:
            #Get nearest data
            xpos=np.argmin(np.abs(event.xdata-self.x))
            ypos=np.argmin(np.abs(event.ydata-self.y))
            
            #Check which mouse button:
            if event.button==1:
                #Plot it                
                c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
                self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)

            elif event.button==3:
                #Plot it                
                c,=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
                self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.y_subplot:
            ypos=np.argmin(np.abs(event.xdata-self.y))
            c=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
            self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.x_subplot:
            xpos=np.argmin(np.abs(event.xdata-self.x))
            c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
            self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)
        #Show it
        plt.draw()

class new_viewer_2d(object):
    def __init__(self, dataDict, data, field=0, scaling=1.0): #z,x=None, y=None): #
        """
        Shows a given array in a 2d-viewer.
        Input: dataDict, runData dictionary
        data, z field
        """

        if dataDict['totNumFields'] == 2 and field == 2:
            totNumFields = 3
        elif dataDict['totNumFields'] > field:
            totNumFields = dataDict['totNumFields']
        else:
            raise Exception('Problem in viewer_2d: dataDict[totnumFields]=%d,\tfield=%d' %
                             (dataDict['totNumFields'], field))

        dataGrid = np.zeros(( totNumFields,
                              dataDict['dimX'],
                              dataDict['dimY']))

        for i in range(0,len(data)):
            # ijk
            # elements in one field: dimX * dimY
            f_i = i//(dataDict['dimX']*dataDict['dimY'])
            x_i = i%dataDict['dimX']
            y_i = (i%(dataDict['dimX']*dataDict['dimY']))//dataDict['dimX']

            #print("%s\t%s\t%s\t%s" % (i,f_i, x_i, y_i))
            dataGrid[f_i,y_i,x_i] = data[i]/scaling

        if totNumFields > dataDict['totNumFields']:
            for i in range(0,dataDict['dimX']*dataDict['dimY']):
                x_i = i%dataDict['dimX']
                y_i = i//dataDict['dimX']
                dataGrid[2,y_i,x_i] = np.sqrt(dataGrid[0,y_i,x_i]**2 +
                                              dataGrid[1,y_i,x_i]**2)/scaling

        dataGrid[dataGrid == 0] = np.nan

        x = np.arange(dataDict['dimX'])
        y = np.arange(dataDict['dimY'])
        z= np.ma.array(data, mask=np.isnan(data))

        if x==None:
            self.x=np.arange(z.shape[0])
        else:
            self.x=x
        if y==None:
            self.y=np.arange(z.shape[1])
        else:
            self.y=y
        self.z=z
        self.fig=plt.figure()
        #Doing some layout with subplots:
        self.fig.subplots_adjust(0.05,0.05,0.98,0.98,0.1)
        self.overview=plt.subplot2grid((8,4),(0,0),rowspan=7,colspan=2)
        #self.overview.pcolormesh(self.x,self.y,self.z)
        self.overview.imshow(self.z,interpolation='bicubic',aspect='auto',
                             extent=[self.x[0],self.x[-1],
                                     self.y[0],self.y[-1]])
        self.overview.autoscale(1,'both',1)
        self.x_subplot=plt.subplot2grid((8,4),(0,2),rowspan=4,colspan=2)
        self.y_subplot=plt.subplot2grid((8,4),(4,2),rowspan=4,colspan=2)

        #Adding widgets, to not be gc'ed, they are put in a list:
        
        cursor=Cursor(self.overview, useblit=True, color='black', linewidth=2 )
        but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
        reset_button=Button(but_ax,'Reset')
        but_ax2=plt.subplot2grid((8,4),(7,1),colspan=1)
        legend_button=Button(but_ax2,'Legend')
        self._widgets=[cursor,reset_button,legend_button]
        #connect events
        reset_button.on_clicked(self.clear_xy_subplots)
        legend_button.on_clicked(self.show_legend)
        self.fig.canvas.mpl_connect('button_press_event',self.click)
    
    def show_legend(self, event):        
        """Shows legend for the plots"""
        for pl in [self.x_subplot,self.y_subplot]:
            if len(pl.lines) > 0:
                pl.legend()
        plt.draw()

    def clear_xy_subplots(self,event):
        """Clears the subplots."""
        for j in [self.overview,self.x_subplot,self.y_subplot]:
            j.lines=[]
            j.legend_ = None
        plt.draw()


    def click(self,event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        if event.inaxes==self.overview:
            #Get nearest data
            xpos=np.argmin(np.abs(event.xdata-self.x))
            ypos=np.argmin(np.abs(event.ydata-self.y))
            
            #Check which mouse button:
            if event.button==1:
                #Plot it                
                c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
                self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)

            elif event.button==3:
                #Plot it                
                c,=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
                self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.y_subplot:
            ypos=np.argmin(np.abs(event.xdata-self.y))
            c=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
            self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.x_subplot:
            xpos=np.argmin(np.abs(event.xdata-self.x))
            c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
            self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)
        #Show it
        plt.draw()

class newest_viewer_2d(object):
    def __init__(self,z,x=None, y=None): #
        """
        Shows a given array in a 2d-viewer.
        Input: dataDict, runData dictionary
        data, z field
        """

        if x==None:
            self.x=np.arange(z.shape[0])
        else:
            self.x=x
        if y==None:
            self.y=np.arange(z.shape[1])
        else:
            self.y=y
        self.z=z
        self.fig=plt.figure()
        #Doing some layout with subplots:
        self.fig.subplots_adjust(0.05,0.05,0.98,0.98,0.1)
        self.overview=plt.subplot2grid((8,4),(0,0),rowspan=7,colspan=2)
        #self.overview.pcolormesh(self.x,self.y,self.z)
        self.overview.imshow(self.z,interpolation='none',aspect='auto',
                             extent=[self.x[0],self.x[-1],
                                     self.y[0],self.y[-1]])
        self.overview.autoscale(1,'both',1)
        self.x_subplot=plt.subplot2grid((8,4),(0,2),rowspan=4,colspan=2)
        self.y_subplot=plt.subplot2grid((8,4),(4,2),rowspan=4,colspan=2)

        #Adding widgets, to not be gc'ed, they are put in a list:
        
        cursor=Cursor(self.overview, useblit=True, color='black', linewidth=2 )
        but_ax=plt.subplot2grid((8,4),(7,0),colspan=1)
        reset_button=Button(but_ax,'Reset')
        but_ax2=plt.subplot2grid((8,4),(7,1),colspan=1)
        legend_button=Button(but_ax2,'Legend')
        self._widgets=[cursor,reset_button,legend_button]
        #connect events
        reset_button.on_clicked(self.clear_xy_subplots)
        legend_button.on_clicked(self.show_legend)
        self.fig.canvas.mpl_connect('button_press_event',self.click)
    
    def show_legend(self, event):        
        """Shows legend for the plots"""
        for pl in [self.x_subplot,self.y_subplot]:
            if len(pl.lines) > 0:
                pl.legend()
        plt.draw()

    def clear_xy_subplots(self,event):
        """Clears the subplots."""
        for j in [self.overview,self.x_subplot,self.y_subplot]:
            j.lines=[]
            j.legend_ = None
        plt.draw()


    def click(self,event):
        """
        What to do, if a click on the figure happens:
            1. Check which axis
            2. Get data coord's.
            3. Plot resulting data.
            4. Update Figure
        """
        if event.inaxes==self.overview:
            #Get nearest data
            xpos=np.argmin(np.abs(event.xdata-self.x))
            ypos=np.argmin(np.abs(event.ydata-self.y))
            
            #Check which mouse button:
            if event.button==1:
                #Plot it                
                c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
                self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)

            elif event.button==3:
                #Plot it                
                c,=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
                self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.y_subplot:
            ypos=np.argmin(np.abs(event.xdata-self.y))
            c=self.x_subplot.plot(self.x, self.z[ypos,:],label=str(self.y[ypos]))
            self.overview.axhline(self.y[ypos],color=c.get_color(),lw=2)

        if event.inaxes==self.x_subplot:
            xpos=np.argmin(np.abs(event.xdata-self.x))
            c,=self.y_subplot.plot(self.y, self.z[:,xpos],label=str(self.x[xpos]))
            self.overview.axvline(self.x[xpos],color=c.get_color(),lw=2)
        #Show it
        plt.draw()
