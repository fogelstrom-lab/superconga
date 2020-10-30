================
 Tutorials
================

Tutorials go here.

Parameter file
==============

A simulation is set up by the use of a JSON parameter file. Complete examples of parameter files are found in the directory ``examples/``, e.g. ``dwave_octagon.json``
::
    {
        "convergence_criterion": 1e-6,
        "crystal_axes_rotation": 0.0,
        "data_path": "data",
        "external_flux_quanta": 0.0,
        "grain_side_length": 30.0,
        "lattice_side_length": 201,
        "num_fermi_momenta": 55,
        "num_iterations_max": 1000,
        "num_iterations_min": 20,
        "num_ozaki_energies": 20,
        "num_polygon_edges": 8,
        "order_parameter": [
            {
                "critical_temperature": 1.0,
                "initial_phase_shift": 0.0,
                "initial_phase_winding": 0.0,
                "initial_random_phase_weight": 0.0,
                "symmetry": "dx2-y2"
            }
        ],
        "save_data": false,
        "save_data_interval": 200,
        "solenoid": false,
        "temperature": 0.5,
        "verbose": true,
        "visualize": true
    }

All parameters are explained below.

Convergence criterion
---------------------
The convergence criterion, :math:`C`, is set by the parameter ``convergence_criterion``. When the relative change of each order-parameter component is smaller than :math:`C` the simulation is considered converged, i.e.

.. math::

    \sqrt{\frac{\sum_{\mathbf{R}}|\Delta_{i}(\mathbf{R}) - \Delta_{i-1}(\mathbf{R})|^2}{\max(\sum_{\mathbf{R}}|\Delta_{i-1}(\mathbf{R})|^2, \epsilon)}} < C

where :math:`\Delta_i(\mathbf{R})` is the order-parameter component at the position :math:`\mathbf{R}` and the :math:`i` th iteration, and :math:`\epsilon > 0` is a small value to avoid division by zero. A value of :math:`C \approx 10^{-6}` is recommended, and it must be larger than zero.

Crystal-axes rotation
---------------------
The crystal-axes rotation, :math:`\theta_{ab}`, is set by the parameter ``crystal_axes_rotation``. It is the rotation of the crystal ab-axes with the respect to the grain coordinate system. The rotation :math:`\theta_{ab}=0` implies that that coordinate systems coincide. Note that :math:`\theta_{ab}` is expressed in terms of a full turn, i.e. :math:`2 \pi` radians, so that a rotation of e.g. :math:`90^\circ` corresponds to ``"crystal_axes_rotation": 0.25``.

Data path
---------
The parameter ``data_path`` is the relative path to the directory where the data will be saved.

External magnetic flux
----------------------
The external magnetic flux, :math:`\Phi`, is set by the parameter ``external_flux_quanta``. It is expressed in terms of the (superconducting) flux-quanta, :math:`\Phi_0 = h/(2e)`, through the grain. The magnetic field is parallel to the :math:`z`-axis, i.e. :math:`\mathbf{B} = B \hat{\mathbf{z}}` , where :math:`B = \Phi/A` , and :math:`A` is the grain area. For non-solenoid the magnetic field is distributed uniformly over the grain.

Grain side length
-----------------
The simulation is done in a square. The parameter ``grain_side_length`` sets the physical side length of the simulation in units of the superconducting coherence length

.. math::
    \xi_0 = \frac{\hbar v_F}{2\pi k_B T_c}


Lattice side length
-------------------
The simulation is done in a square. The parameter ``lattice_side_length`` sets the number of lattice sites per side, :math:`N`, of the square. Thus the total number of lattice sites is :math:`N^2`.

Number of Fermi momenta
-----------------------
The number of Fermi momenta used in the Fermi-surface average is controlled by the parameter ``num_fermi_momenta``.

Maximum number of iterations
----------------------------
The maximum number of iterations is set by ``num_iterations_max``.

Minimum number of iterations
----------------------------
The minimum number of iterations is set by ``num_iterations_min``.

Number of Ozaki energies
------------------------
Ozaki energies are similar to Matsubara enegies, but are unevenly spaced on the imaginary axis, thus making more efficient. The same accuracy can be achieved with fewer Ozaki, compared to Matsubara, energies. The number of Ozaki energies is controlled by ``num_ozaki_energies``.

Number of polygon edges
-----------------------
The geometry of the grain is controlled by ``num_polygon_edges``. It sets the number of edges of the polygon; e.g. ``"num_polygon_edges": 4`` yields a square. The values ``0``, ``1``, and ``2`` are invalid, but negative values yield a disc.

Order parameter
---------------
The order parameter is set by a list of components. Each component is controlled by five different parameters; two relating to its physical properties, ``critical_temperature``, and ``symmetry``, and three to the initial guess, ``initial_phase_shift``, ``initial_phase_winding``, and ``initial_random_phase_weight``. The relative critical temperature, :math:`T_c`, is set by ``critical_temperature``, and is scaled so that the maximum :math:`T_c` of all components is :math:`1`. The component symmetry, :math:`\eta`, is set by ``symmetry``. Four symmetries are supported;

    * ``s``: Yields an :math:`s`-wave component, i.e. :math:`\eta(\mathbf{k}) = 1`.
    * ``dxy``: Yields a :math:`d_{xy}`-wave component, i.e. :math:`\eta(\mathbf{k}) = \sin(k_x) \sin(k_y)`.
    * ``dx2-y2``: Yields a :math:`d_{x^2 - y^2}`-wave component, i.e. :math:`\eta(\mathbf{k}) = \cos(k_x) - \cos(k_y)`.
    * ``g``: Yields a :math:`g_{xy(x^2 - y^2)}`-wave component, i.e. :math:`\eta(\mathbf{k}) = \sin(k_x) \sin(k_y) ( \cos(k_x) - \cos(k_y))`.

The order-parameter component is initialized as

.. math::
    \Delta(\mathbf{R}) = \Delta_0 \exp(i\theta(\mathbf{R})) \\
    \theta(\mathbf{R}) = \theta_{s} + \theta_w(\mathbf{R}) + \theta_r(\mathbf{R})

where :math:`\Delta_0` is the bulk value, :math:`\theta_{s}` is the (global) phase shift, :math:`\theta_w(\mathbf{R})` is the phase winding, and :math:`\theta_r(\mathbf{R})` is a random phase.

The phase shift, :math:`\theta_{s}`, is set by ``initial_phase_shift``, and is expressed in terms of a full turn, i.e. :math:`2 \pi` radians. The phase winding is parameterized as :math:`\theta_w(\mathbf{R}) = C_w \mathrm{atan2}(k_y, k_x)`, and :math:`C_w` is set by the parameter ``initial_phase_winding``. The random phase is parameterized as :math:`\theta_r(\mathbf{R}) = C_r r(\mathbf{R})` where :math:`r(\mathbf{R})` is a random number, :math:`r \in [-\pi, \pi)`, one for every position :math:`\mathbf{R}`. The value of :math:`C_r` is set by the parameter ``initial_random_phase_weight``.

Save data
---------
Whether to save data or not is set by the parameter ``save_data``.

Interval to save data
---------------------
If ``"save_data": true`` then data is saved at every :math:`n` th iteration, with :math:`n` being set by the parameter ``save_data_interval``.

Solenoid
--------
If the parameter ``solenoid`` is set to ``true`` a hole in the center of the grain is made. If ``"num_polygon_edges": 4`` then the hole will be a square. Furthermore, if ``true`` the solenoid guage of the external magnetic field is used.

Temperature
-----------
The temperature, :math:`T`, is set by the parameter ``temperature``. It is expressed in terms of the maximum critical temperature, :math:`T_c`, of the order-parameter components. Note that :math:`T>T_c` yields no superconductivity, and more Ozaki energies are needed the lower the temperature.

Verbose
-------
If ``"verbose": true`` information at every iteration will be printed to the terminal.

Visualize
---------
If ``"visualize": true`` the order-parameter components, and the current, will be plotted at every iteration.

Command-line flags
==================

Geometry setup
==============
Shapes: n-gon, free polygon, disc
Adding and subtracting
Offset and rotation

Order-parameter setup
=====================

Visualization
=============

Post-processing: plotting the results
=====================================
Using shipped tools (Python) to plot the data

Post-processing: reading and parsing the data
=============================================
How to read the data so it can be imported to your favorite language.
