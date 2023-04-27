#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
"""Example computing the TPA cross section for water using ADC
(10.1063/1.3682324)
"""
import adcc
import numpy as np

from pyscf import gto, scf
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc.solver import IndexSymmetrisation
from adcc.solver.conjugate_gradient import conjugate_gradient, default_print
from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.IsrMatrix import IsrMatrix


class ShiftedMat(adcc.AdcMatrix):
    def __init__(self, method, mp_results, omega=0.0):
        self.omega = omega
        super().__init__(method, mp_results)
        self.omegamat = adcc.ones_like(self.diagonal()) * omega

    def __matmul__(self, other):
        return super().__matmul__(other) - self.omegamat * other


# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='aug-cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

# solve for Eigenvalues
state = adcc.adc2(scfres, n_singlets=1, conv_tol=1e-8)

# setup modified transition moments
dips = state.reference_state.operators.electric_dipole
rhss = modified_transition_moments("adc2", state.ground_state, dips)
isrmatrix = IsrMatrix("adc2", state.ground_state, dips)

S = np.zeros((len(state.excitation_energy), 3, 3))
for f, ee in enumerate(state.excitation_energy):
    freq = ee / 2.0
    matrix = ShiftedMat("adc2", state.ground_state, freq)
    preconditioner = JacobiPreconditioner(matrix)
    explicit_symmetrisation = IndexSymmetrisation(matrix)
    preconditioner.update_shifts(freq)
    response = []
    # solve all systems of equations
    for mu in range(3):
        rhs = rhss[mu]
        x0 = preconditioner.apply(rhs)
        res = conjugate_gradient(
            matrix, rhs=rhs, x0=x0, callback=default_print,
            Pinv=preconditioner, conv_tol=1e-6,
            explicit_symmetrisation=explicit_symmetrisation
        )
        response.append(res)
    right_vec = isrmatrix @ state.excitation_vector[f]
    for mu in range(3):
        for nu in range(mu, 3):
            # compute the matrix element
            S[f, mu, nu] = (
                response[mu].solution @ right_vec[nu]
                + response[nu].solution @ right_vec[mu]
            )
            S[f, nu, mu] = S[f, mu, nu]
    print("Two-Photon Matrix for state", f)
    print(S[f])
    delta = 1.0 / 15.0 * (
        np.einsum('mm,vv->', S[f], S[f])
        + np.einsum('mv,mv->', S[f], S[f])
        + np.einsum('mv,vm->', S[f], S[f])
    )
    print("TPA Cross section [a.u.]: {:.4f}".format(delta))
    np.testing.assert_allclose(6.5539, delta, atol=1e-4)
