#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data
from adcc.solver.adcman import jacobi_davidson

# Gather precomputed data
data = import_data()

# Setup the matrix
matrix = adcc.AdcMatrix("adc2", adcc.ReferenceState(data, frozen_core=1))

# Solve for 3 singlets and 3 triplets with some default output
singlets, triplets = jacobi_davidson(matrix, print_level=100, n_singlets=3, n_triplets=3)
print(singlets.describe())
print(triplets.describe())
