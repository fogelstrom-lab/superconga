# Examples

A collection of parameter files, one per example, demonstrating different physical aspects 
of superconductors that are computable through the SuprConGPU framework. 
See [the documentation](https://suprcongpu.gitlab.io/suprcongpu/) for more information.

## D-Wave Chiral

A simulation of a [d_{xy} + id_{x^2 - y^2}]-wave superconductor, i.e. a chiral d-wave, in a disc-shaped grain.
No external magnetic field is applied. The result is a chiral current around the grain boundary.

Folder:
```
dwave_chiral
```

## D-Wave Octagon

A simulation of a d_{x^2 - y^2}-wave superconductor in an octagon grain, with half of the edges being
maximally pair-breaking, and the other half not pair-breaking. No external magnetic field is applied. 
The result is a suppression of the order parameter close to the edges with normals along the diagonals.

Folder:
```
dwave_octagon
```

## D-Wave Phase Crystal

A simulation of a d_{x^2 - y^2}-wave superconductor in a diamond shaped grain, with a single pair-breaking edge.
The temperature is low, inducing a crystallization of the phase, eliciting spontaneous current loops and magnetic fields.
Raising the temperature to e.g. T = 0.2 T_c destroys the phase crystal.

Folder:
```
dwave_phase_crystal
```

## D-Wave Plus S-Wave: An Order Parameter with Multiple Components

A simulation of a d_{xy}-wave superconductor with a subdominant s-wave component, in a disc-shaped grain. 
No external magnetic field is applied. The result is small s-wave magnitude along the ab-axes diagonals
close to the edges.

Folder:
```
dwave_plus_swave
```

## S-Wave Disc: The Meissner Effect

A simulation of an s-wave superconductor in a disc-shaped grain. 
A weak external magnetic field is applied, resulting in circulating Meissner currents,
along the edge.

Folder:
```
swave_disc_meissner
```

## S-Wave Disc: The Meissner Effect with Vortex

A simulation of an s-wave superconductor in a disc-shaped grain. 
A magnetic field of one and a half flux-quanta is applied, resulting in
circulating Meissner currents, along the edge, and a vortex in the middle.

Folder:
```
swave_disc_vortex
```

## S-Wave Solenoid: Flux Quantization

A simulation of an s-wave superconductor in a square grain, with a hole in the center
(the hole being square as well) making it a solenoid. A magnetic field of N flux quanta
is applied. The lowest free energy is obtained with an order parameter with N number of
phase windings.

Folder:
```
swave_solenoid_flux_quantization
```

## S-Wave Abrikosov Vortex Lattice

A simulation of an s-wave superconductor in a square grain, with high external flux
quanta, inducing an Abrikosov vortex lattice. The lattice illustrates several microscopic
effects, including 1. the vortex shell effect where the vortices enter in concentric shells
rather than one-by-one, and 2. the lattice shape effect where the lattice mimics the
rotational symmetry of the grain (and the OP pairing symmetry if asymmetric).
Note that the result is sensitive to the initial guess of the order parameter.

Folders:
```
swave_abrikosov_vortex_lattice
```
