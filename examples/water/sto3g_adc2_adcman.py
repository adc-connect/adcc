#!/usr/bin/env python3

from adcc.solver.adcman import jacobi_davidson
from import_data import import_data
import adcc
import IPython

# Gather preliminary data and import it into an HfData object
data = import_data()

# Initialise the memory (256 MiB)
adcc.memory_pool.initialise(max_memory=256 * 1024 * 1024)

# Run the initial preparation (MP2, intermediates, ...)
res = adcc.tmp_run_prelim(data, "adc2", n_guess_singles=3)

# Setup the matrix
matrix = adcc.AdcMatrix(res.method, res.ground_state)

# Solve for 3 singlets and 3 triplets with some default output
state = jacobi_davidson(matrix, print_level=100, n_singlets=3, n_triplets=3)
IPython.embed()
