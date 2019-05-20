#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from import_data import import_data
from adcc.solver.adcman import jacobi_davidson

# Gather preliminary data and import it into an HfData object
data = import_data()
data["restricted"] = False

# Setup the matrix
mp = adcc.LazyMp(adcc.ReferenceState(data))
matrix = adcc.AdcMatrix("adc2", mp)

# Solve for 6 states with some default output
states = jacobi_davidson(matrix, print_level=100, n_states=10)
print(states[0].describe())
