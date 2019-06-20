#!/usr/bin/env julia
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

# An example how to use adcc from julia
import PyCall
import PyPlot
pyscf = PyCall.pyimport("pyscf")
adcc = PyCall.pyimport("adcc")


mol = pyscf.gto.M(
    atom="""
        O 0 0 0;
        H 0 0 1.795239827225189;
        H 1.693194615993441 0 -0.599043184453037
    """,
    basis="6311g**",
    unit="Bohr"
)
scfres = pyscf.scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.conv_tol_grad = 1e-7
scfres.kernel()

# Run an adc2 calculation:
singlets = adcc.adc2(scfres, n_singlets=5)
triplets = adcc.adc2(singlets.matrix, n_triplets=3)

# Attach state densities
println(singlets.describe())
println()
println(triplets.describe())
