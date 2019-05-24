#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
from import_data import import_data

import adcc

from adcc.solver.adcman import jacobi_davidson

# Gather preliminary data and import it into an HfData object
data = import_data()
data["restricted"] = False

# Setup the matrix
matrix = adcc.AdcMatrix("adc2", adcc.ReferenceState(data))

# Solve for 6 states with some default output
states = jacobi_davidson(matrix, print_level=100, n_states=10)
print(states[0].describe())
