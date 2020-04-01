from __future__ import division

import os
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plotEnergySweeps(runDict,
                     label_in=None,
                     marker_in=None,
                     marker_size=None,
                     color_in = None,
                     ax_in=None,
                     normalize=True,
                     plotResidual=True,
                     B_axis_in=None,
                     Res_axis_in=None,
                     title_in=None,
                     interpolate=False,
                     drawVerticalLine=False,
                     lastAx=True,
                     wnums_in=None):
    if marker_in == None:
        marker_in = 'o'
    if marker_size == None:
        marker_size = 8
    if color_in == None:
        color_in = ['k','r','g','b','m','c','y','','','','']

    if ax_in == None:
        if plotResidual:
            f, (ax1, ax2) = plt.subplots(2,sharex=False,sharey=False)
        else:
            f, (ax1) = plt.subplots(1,sharex=False,sharey=False)
    else:
        ax1 = ax_in[0]
        if plotResidual:
            ax2 = ax_in[1]

    labeled = 0

    for wnum in runDict:
        plot_bvalues = np.array(runDict[wnum]['bvalues'])
        plot_energies = np.array(runDict[wnum]['energies'])
        plot_residuals = np.array(runDict[wnum]['residuals'])
        if normalize:
            plot_bvalues /= runDict[wnum]['B0']

        if labeled == 0:
            this_label = label_in
            labeled = 1
        else:
            this_label = None

        if interpolate and len(plot_bvalues) > 3:
            scatter_alpha = 0.25
            ip_z = np.polyfit(plot_bvalues, plot_energies, 2)
            ip_p = np.poly1d(ip_z)
            ip_xp = np.linspace(min(plot_bvalues), max(plot_bvalues), 1000)
            ax1.plot(ip_xp, ip_p(ip_xp), ls='-',c=color_in[wnum],alpha=0.25)

            ip_mi = np.argmin(ip_p(ip_xp))
            #print('%d\tx=%s,y=%s' % (wnum,(ip_xp[ip_mi]-wnum)/float(wnum),(ip_p(ip_xp))[ip_mi]))
            #ax1.scatter(ip_xp[ip_mi],(ip_p(ip_xp))[ip_mi],c=colors[wnum])
            ax1.plot(ip_xp[ip_mi],
                     (ip_p(ip_xp))[ip_mi],
                     label=this_label,
                     ls='', lw=3.0, marker=marker_in,
                     markeredgewidth=1.0, ms=marker_size,
                     markeredgecolor=color_in[wnum], markerfacecolor='none')#color_in[wnum])
            #ax1.axhline(y=(ip_p(ip_xp))[ip_mi],
            #            alpha=0.5,c=color_in[wnum])
        else:
            scatter_alpha = 1.0
            ax1.plot(plot_bvalues,
                     plot_energies,
                     label=this_label,
                     ls='', lw=3.0, marker=marker_in,
                     markeredgewidth=1.0, ms=marker_size,
                     markeredgecolor=color_in[wnum], markerfacecolor='none')


        if plotResidual:
            ax2.plot(plot_bvalues,
                     plot_residuals,
                     label=label_in,
                     ls='', lw=3.0, marker=marker_in,
                     markeredgewidth=1.0, ms=marker_size,
                     markeredgecolor=color_in[wnum], markerfacecolor='none')

    if lastAx:
        ax1.set_ylabel(r'$\delta\Omega \mathrm{(a.u.)}$')
        legend_out = ax1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        if drawVerticalLine:
            for w_i in wnums_in:
                ax1.axvline(w_i, ls='--', color=color_in[w_i], alpha=0.5)

    if B_axis_in != None:
        ax1.axis(B_axis_in)

    if plotResidual:
        ax2.set_yscale('log')
        ax2.set_ylabel(r'$\Delta_{\mathrm{\mathsf{residual}}}$')
        if normalize:
            ax2.set_xlabel(r'$B/B_c$')
        else:
            ax2.set_xlabel(r'$B/B_0$')
        if Res_axis_in != None:
            ax2.axis(Res_axis_in)
    else:
        if lastAx:
            if normalize:
                ax1.set_xlabel(r'$B/B_c$')
            else:
                ax1.set_xlabel(r'$B/B_0$')
            

    if lastAx:
        if title_in != None:
            plt.title(title_in)
        plt.subplots_adjust(left = 0.15,
                            bottom = 0.17,
                            right = 0.9,
                            top = 0.9,
                            wspace = 0.2,
                            hspace = 0.2)
    else:
        legend_out = None

    if plotResidual:
        ax_out = [ax1, ax2]
    else:
        ax_out = [ax1]
    return ax_out, legend_out