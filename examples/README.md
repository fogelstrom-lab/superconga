# Examples

A collection of parameter files, one per example, demonstrating different physical aspects 
of superconductors that are computable through the SuprConGPU framework. 
See [the documentation](https://suprcongpu.gitlab.io/suprcongpu/) for more information.

## D-Wave Square

A simulation of a d-wave superconductor in a square grain, with all edges being
maximally pair-breaking. No external magnetic field is applied. The result is
a suppression of the order parameter close to the edges.

File:
```
dwave_square.json
```

## D-Wave Phase Crystal

A simulation of a d-wave superconductor in a square grain, with all edges being
maximally pair-breaking. The temperature is low, inducing a crystallization of the
phase, eliciting spontaneous current loops and magnetic fields.

File:
```
dwave_phase_crystal.json
```

## S-Wave Disc: Meissner Effect

A simulation of an s-wave superconductor in a disc-shaped grain. 
A weak external magnetic field is applied, resulting in circulating Meissner currents,
along the edge.

File:
```
swave_disc_meissner.json
```

## S-Wave Solenoid: Flux Quantization

A simulation of an s-wave superconductor in a square grain, with a hole in the center
(the hole being square as well) making it a solenoid. A magnetic field of N flux quanta
is applied. The lowest free energy is obtained with an order parameter with N number of
phase windings.

File:
```
swave_solenoid_flux_quantization.json
```

## S-Wave Abrikosov Vortex Lattice

A simulation of an s-wave superconductor in a square grain, with high external flux
quanta, inducing an Abrikosov vortex lattice. The lattice illustrates several microscopic
effects, including 1. the vortex shell effect where the vortices enter in concentric shells
rather than one-by-one, and 2. the lattice shape effect where the lattice mimics the
rotational symmetry of the grain (and the OP pairing symmetry if asymmetric).

File:
```
swave_abrikosov_vortex_lattice.json
```
