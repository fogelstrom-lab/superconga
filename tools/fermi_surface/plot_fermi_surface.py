import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plotFermiSurface(dispersion,
                     fermiVelocity,
                     dos,
                     filename,
                     labelPosition=0.05,
                     minDos=1e-3,
                     colormap='RdBu_r',
                     fontsize=12,
                     colorBarRatio=0.05,
                     plotRatio=4,
                     hspace=0.25,
                     numTicks=5,
                     dpi=100):
    # Setup figure and subplots.
    sizeX = 10
    sizeY = sizeX * (1 + plotRatio + colorBarRatio)
    fig = plt.figure(figsize=(sizeX, sizeY), dpi=dpi)
    ax_cbar, ax_disp, ax_dos = fig.subplots(nrows=3,
                                            ncols=1,
                                            gridspec_kw={
                                                'height_ratios':
                                                [colorBarRatio, plotRatio, 1],
                                                'hspace':
                                                hspace
                                            })

    # ====================== DISPERSION ====================== #
    kx = np.array(dispersion['kx']) / np.pi
    ky = np.array(dispersion['ky']) / np.pi
    ek = np.array(dispersion['dispersion'])

    # Plot the dispersion in colour.
    c = ax_disp.pcolor(kx, ky, ek, cmap=colormap)

    # Colorbar.
    cbar = fig.colorbar(c, cax=ax_cbar, orientation='horizontal')
    cbar.set_label(r'$E / t$', fontsize=fontsize)
    ax_cbar.set_aspect(colorBarRatio / plotRatio, adjustable='box')

    # The contour plot of the Fermi surface, hence levels=0.
    ax_disp.contour(kx, ky, ek, colors='k', levels=[0.0])

    # ====================== FERMI VELOCITY ====================== #
    kx = np.array(fermiVelocity['kx']) / np.pi
    ky = np.array(fermiVelocity['ky']) / np.pi
    vx = np.array(fermiVelocity['vx'])
    vy = np.array(fermiVelocity['vy'])

    # Plot the Fermi velocities.
    ax_disp.quiver(kx, ky, vx, vy, color='k')

    # Set axis title and labels.
    ax_disp.set_xlabel(r'$k_x a / \pi$', fontsize=fontsize)
    ax_disp.set_ylabel(r'$k_y a / \pi$', fontsize=fontsize)

    # Set axis limits.
    xMin = -1
    xMax = 1
    yMin = xMin
    yMax = xMax
    ax_disp.set_xlim(xMin, xMax)
    ax_disp.set_ylim(yMin, yMax)

    # Set ticks.
    ticks = np.linspace(xMin, xMax, numTicks)
    ax_disp.set_xticks(ticks)
    ax_disp.set_yticks(ticks)

    # Set aspect ratio.
    ax_disp.set_aspect('equal', adjustable='box')

    # Add text.
    xRange = xMax - xMin
    yRange = yMax - yMin
    x = xMin + xRange * labelPosition
    y = yMax - yRange * labelPosition
    ax_disp.text(x, y, r'(a)', fontsize=fontsize, color='w')

    # ====================== DOS ====================== #
    energies = np.array(dos['energies'])
    dos = np.array(dos['dos'])

    # Exclude points with DOS \approx 0.
    cond = dos > minDos
    energies = energies[cond]
    dos = dos[cond]

    # Plot the density of states.
    ax_dos.plot(energies, dos, c='k', label=r'$DOS$')

    # Plot the Fermi level.
    ax_dos.axvline(x=0.0, c='r', dashes=[2, 2], label=r'$\epsilon_F$')

    # Set axis title and labels.
    ax_dos.set_xlabel(r'$E / t$', fontsize=fontsize)
    ax_dos.set_ylabel(r'$DOS$', fontsize=fontsize)
    ax_dos.legend()

    # Set axis limits.
    xMin = energies[0]
    xMax = energies[-1]
    yMin = 0.0
    # Should this be fixed or adjusted with max(dos)?
    yMax = 0.7
    ax_dos.set_xlim(xMin, xMax)
    ax_dos.set_ylim(yMin, yMax)

    # Set aspect ratio.
    xRange = xMax - xMin
    yRange = yMax - yMin
    dosAspectRatio = xRange / (yRange * plotRatio)
    ax_dos.set_aspect(dosAspectRatio, adjustable='box')

    # Add text.
    x = xMin + xRange * labelPosition
    # Some magic number to move the text slightly.
    y = yMax - yRange * labelPosition * dosAspectRatio * 1.5
    ax_dos.text(x, y, r'(b)', fontsize=fontsize)

    # Align labels in the subplots.
    fig.align_labels()

    # For some reason this needs to be here, otherwise the saved plots look different.
    # TODO(niclas): Fix this!
    plt.show()

    # Change extension to png.
    filename = os.path.splitext(filename)[0] + '.png'

    # Save.
    fig.savefig(filename, bbox_inches='tight', dpi=fig.dpi)


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(
        description=
        'Plot the Fermi surface for a tight-binding model on a square lattice.'
    )
    parser.add_argument('name', type=str, help='Name of file to load.')
    args = parser.parse_args()

    # Load file.
    filename = args.name
    suffix = '.json'
    if not filename.endswith(suffix):
        filename += '.json'
    file = open(filename, 'r')
    data = json.load(file)

    # Plot data.
    plotFermiSurface(data['dispersion'], data['fermiVelocity'], data['dos'],
                     filename)


if __name__ == '__main__':
    main()
