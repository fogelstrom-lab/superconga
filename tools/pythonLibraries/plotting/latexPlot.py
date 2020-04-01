from __future__ import division

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt


def latexPlot(width_in=512.0, fsize_in=16, axlab_ratio=1.5, text_ratio=2.0,
              legend_ratio=1.0, xtick_ratio=1.0, ytick_ratio=1.0, font='sans',
              usetex=True, dpi_in=600):
        fig_width_pt = width_in
        fsize = fsize_in
        # Calculate figure size using golden ratio method
        inches_per_pt = 1.0/72.27                    # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0           # Aesthetic ratio
        fig_width = fig_width_pt*inches_per_pt       # width in inches
        fig_height = fig_width*golden_mean           # height in inches
        #fig_height = fig_height + 40 * inches_per_pt # Add space for title
        fig_size =  [fig_width,fig_height]

        # Set the RC LaTeX parameters
        params = {'backend': 'ps',
                 'axes.labelsize': axlab_ratio*fsize,
                 'text.fontsize': text_ratio*fsize,
                 'legend.fontsize': legend_ratio*fsize,
                 'xtick.labelsize': xtick_ratio*fsize,
                 'ytick.labelsize': ytick_ratio*fsize,
                 'text.usetex': usetex,
                 'font.family': font,
                 'font.sans': 'cm',
                 'figure.figsize': fig_size,
                 'figure.dpi': dpi_in,
                 #'figure.autolayout': True,
                 'text.latex.preamble' : [r'\usepackage{sans}', r'\usepackage{amsmath}']
                 }

        plt.rcParams.update(params)

        return axlab_ratio*fsize

def latexPlotPRC(oneColumn=True, fontSize=12, font='sans', usetex=True, dpi_in=600):
        oneColumnWidth = 3.375  # inches
        twoColumnWidth = 7.0    # inches

        if oneColumn:
            fig_width_inch = oneColumnWidth
        else:
            fig_width_inch = twoColumnWidth
        # Calculate figure size using golden ratio method
        golden_mean = (np.sqrt(5.0)-1.0)/2.0          # Aesthetic ratio
        fig_height_inch = fig_width_inch * golden_mean   # height in inches
        fig_size =  [fig_width_inch,fig_height_inch]

        axlab_ratio = 1.5
        text_ratio = 2.0
        legend_ratio = 1.0
        xtick_ratio = 1.0
        ytick_ratio = 1.0

        # Set the RC LaTeX parameters
        params = {
                  'backend': 'ps',
                  'axes.labelsize': axlab_ratio*fontSize,
                  'text.fontsize': text_ratio*fontSize,
                  'legend.fontsize': legend_ratio*fontSize,
                  'xtick.labelsize': xtick_ratio*fontSize,
                  'ytick.labelsize': ytick_ratio*fontSize,
                  'text.usetex': usetex,
                  'font.family': font,
                  'font.sans': 'cm',
                  'figure.figsize': fig_size,
                  'figure.dpi': dpi_in,
                  #'figure.autolayout': True,
                  'text.latex.preamble' : [r'\usepackage{sans}', r'\usepackage{amsmath}']
                 }

        plt.rcParams.update(params)

        return axlab_ratio*fontSize
