#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import numpy as np

from pyscf import gto, scf
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver import IndexSymmetrisation
from adcc.solver.conjugate_gradient import conjugate_gradient, default_print
from adcc.adc_pp.modified_transition_moments import modified_transition_moments


class ShiftedMat(adcc.AdcMatrix):
    def __init__(self, method, mp_results, omega=0.0):
        self.omega = omega
        super().__init__(method, mp_results)
        self.omegamat = adcc.ones_like(self.diagonal()) * omega

    def __matmul__(self, other):
        return super().__matmul__(other) - self.omegamat * other


# Run SCF in pyscf
mol = gto.M(
    atom="""
    Ne
    """,
    basis='aug-cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

refstate = adcc.ReferenceState(scfres)
matrix = ShiftedMat("adc3", refstate, omega=0.0)
rhs = modified_transition_moments("adc2", matrix.ground_state,
                                  refstate.operators.electric_dipole[0])
preconditioner = JacobiPreconditioner(matrix)
freq = 0.0
preconditioner.update_shifts(freq)

explicit_symmetrisation = IndexSymmetrisation(matrix)

x0 = preconditioner.apply(rhs)
res = conjugate_gradient(matrix, rhs=rhs, x0=x0, callback=default_print,
                         Pinv=preconditioner, conv_tol=1e-4,
                         explicit_symmetrisation=explicit_symmetrisation)

alpha_xx = 2.0 * res.solution @ rhs
np.testing.assert_allclose(alpha_xx, 1.994, atol=1e-3)
print("alpha_xx(0) = ", alpha_xx)
