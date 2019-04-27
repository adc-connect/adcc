#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

import molsturm

# Run SCF in molsturm
atoms = ["O", "H", "H"]
coords = [[0, 0, 0],
          [0, 0, 1.795239827225189],
          [1.693194615993441, 0, -0.599043184453037]]
system = molsturm.System(atoms, coords)

hfres = molsturm.hartree_fock(system, basis_type="gaussian",
                              basis_set_name="sto-3g",
                              conv_tol=1e-12, print_iterations=True)

# Run an adc2 calculation:
singlets = adcc.adc2(hfres, n_singlets=5, conv_tol=1e-9)
triplets = adcc.adc2(singlets.matrix, n_triplets=3, conv_tol=1e-9)

# Attach state densities
singlets = adcc.attach_state_densities(singlets)
triplets = adcc.attach_state_densities(triplets)

print(singlets.describe())
print(triplets.describe())
