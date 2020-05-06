#!/bin/bash
COMPUTE_FERMI_SURFACE=../../../tools/fermi_surface/compute_fermi_surface.py
python $COMPUTE_FERMI_SURFACE -t1 -0.25 -mu 0 -v 1.4 -name fs1
python $COMPUTE_FERMI_SURFACE -t1 -0.495 -t2 0.156 -mu -1.267 -v 1.279 -name fs2
