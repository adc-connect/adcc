#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import os
import tempfile

import h5py
import numpy as np

from libadcc import HartreeFockProvider
from adcc.misc import cached_property
from adcc.gradients import GradientComponents
from adcc.gradients.TwoParticleDensityMatrix import ao_pair_index, ao_pair_indices

from .EriBuilder import EriBuilder
from ..exceptions import InvalidReference
from ..ExcitedStates import EnergyCorrection

from pyscf import ao2mo, gto, scf, grad
from pyscf.solvent import ddcosmo


class PyScfGradientProvider:
    def __init__(self, scfres):
        self.scfres = scfres
        self.mol = self.scfres.mol
        self.backend = "pyscf"

    def _empty_gradient(self):
        natoms = self.mol.natm
        return {
            "N": np.zeros((natoms, 3)),
            "S": np.zeros((natoms, 3)),
            "T+V": np.zeros((natoms, 3)),
            "OEI": np.zeros((natoms, 3)),
            "TEI": np.zeros((natoms, 3)),
            "Total": np.zeros((natoms, 3)),
        }

    def _add_one_electron_terms(self, gradient_components, g1_ao, w_ao,
                                pyscf_gradient):
        hcore_deriv = pyscf_gradient.hcore_generator()
        Sx = -1.0 * self.mol.intor('int1e_ipovlp', aosym='s1')
        ao_slices = self.mol.aoslice_by_atom()
        for ia in range(self.mol.natm):
            k0, k1 = ao_slices[ia, 2:]

            # derivative of the overlap matrix
            Sx_a = np.zeros_like(Sx)
            Sx_a[:, k0:k1] = Sx[:, k0:k1]
            Sx_a += Sx_a.transpose(0, 2, 1)
            gradient_components["S"][ia] += np.einsum(
                "xpq,pq->x", Sx_a, w_ao
            )

            # derivative of the core Hamiltonian
            Hx_a = hcore_deriv(ia)
            gradient_components["T+V"][ia] += np.einsum(
                "xpq,pq->x", Hx_a, g1_ao
            )

    def _final_gradient_components(self, pyscf_gradient, gradient_components):
        return GradientComponents(
            self.mol.natm, pyscf_gradient.grad_nuc(), gradient_components["S"],
            gradient_components["T+V"], gradient_components["TEI"]
        )

    def correlated_gradient(self, g1_ao, w_ao, g2_ao_1, g2_ao_2):
        """Full-AO reference path. Allocates the full derivative ERI tensor."""
        Gradient = self._empty_gradient()

        # TODO: does RHF/UHF matter here?
        gradient = grad.RHF(self.scfres)
        self._add_one_electron_terms(Gradient, g1_ao, w_ao, gradient)
        ERIx = -1.0 * self.mol.intor('int2e_ip1', aosym='s1')

        ao_slices = self.mol.aoslice_by_atom()
        for ia in range(self.mol.natm):
            # TODO: only contract/compute with specific slices
            # of density matrices (especially TPDM)
            # this requires a lot of work however...
            k0, k1 = ao_slices[ia, 2:]

            # derivatives of the ERIs
            ERIx_a = np.zeros_like(ERIx)
            ERIx_a[:, k0:k1] = ERIx[:, k0:k1]
            ERIx_a += (
                + ERIx_a.transpose(0, 2, 1, 4, 3)
                + ERIx_a.transpose(0, 3, 4, 1, 2)
                + ERIx_a.transpose(0, 4, 3, 2, 1)
            )
            Gradient["TEI"][ia] += np.einsum(
                "pqrs,xprqs->x", g2_ao_1, ERIx_a, optimize=True
            )
            Gradient["TEI"][ia] -= np.einsum(
                "pqrs,xpsqr->x", g2_ao_2, ERIx_a, optimize=True
            )
        return self._final_gradient_components(gradient, Gradient)

    def _symmetrized_density_slice(self, pair_density, a0, a1):
        """
        Build the four-center derivative-symmetrized packed density slice.

        ``pair_density`` stores ``D[p,r,q,s]`` with packed ket pair ``q,s``.
        For a derivative integral block ``E[a,b,q,s]`` this returns the packed
        equivalent of

        ``D[a,b,q,s] + D[b,a,s,q] + D[q,s,a,b] + D[s,q,b,a]``.
        """
        nao = pair_density.shape[0]
        qidx, sidx = ao_pair_indices(nao)
        offdiag = qidx != sidx
        density_slice = np.asarray(pair_density[a0:a1, :, :]).copy()
        density_slice += np.asarray(pair_density[:, a0:a1, :]).transpose(1, 0, 2)

        for local_a, a in enumerate(range(a0, a1)):
            for b in range(nao):
                pair_ab = int(ao_pair_index(a, b))
                ket_as_bra = np.asarray(pair_density[:, :, pair_ab])
                term = ket_as_bra[qidx, sidx].copy()
                if np.any(offdiag):
                    term[offdiag] += ket_as_bra[sidx[offdiag], qidx[offdiag]]
                if a == b:
                    term *= 2.0
                density_slice[local_a, b, :] += term
        return density_slice

    def _contract_tei_with_packed_density(self, pair_density,
                                          shell_chunk_size=1):
        """
        Contract derivative ERIs with packed AO-pair effective density.

        PySCF only packs the ket pair for ``aosym=\"s2kl\"``.  Off-diagonal
        packed density entries therefore contain the sum of both ket orders.
        Shell slices are kept atom-local so atom AO and shell ranges cannot be
        mixed accidentally.
        """
        if shell_chunk_size <= 0:
            raise ValueError("shell_chunk_size needs to be positive.")
        tei = np.zeros((self.mol.natm, 3))
        ao_slices = self.mol.aoslice_by_atom()
        ao_loc = self.mol.ao_loc_nr()
        nbas = self.mol.nbas

        for ia in range(self.mol.natm):
            sh0, sh1 = ao_slices[ia, :2]
            for p0 in range(sh0, sh1, shell_chunk_size):
                p1 = min(p0 + shell_chunk_size, sh1)
                a0, a1 = ao_loc[p0], ao_loc[p1]
                shls_slice = (p0, p1, 0, nbas, 0, nbas, 0, nbas)
                erix = -1.0 * self.mol.intor(
                    'int2e_ip1', comp=3, aosym='s2kl', shls_slice=shls_slice
                )
                density_slice = self._symmetrized_density_slice(
                    pair_density, a0, a1
                )
                tei[ia] += np.einsum(
                    "xabk,abk->x", erix, density_slice, optimize=True
                )
        return tei

    def ao_pair_density_from_dense(self, g2_ao_1, g2_ao_2, out=None):
        from adcc.gradients.TwoParticleDensityMatrix import TwoParticleDensityMatrix
        return TwoParticleDensityMatrix.ao_pair_density_from_dense(
            g2_ao_1, g2_ao_2, out=out
        )

    def correlated_gradient_direct(self, g1_ao, w_ao, g2, refstate=None,
                                   shell_chunk_size=1, pair_chunk_size=None,
                                   pair_density_storage="memory",
                                   scratch_directory=None):
        """
        Direct PySCF gradient path from the MO/block-sparse TPDM.

        The TPDM is transformed block-by-block to the packed AO-pair effective
        density.  It never forms ``g2_ao_1``/``g2_ao_2`` or the full derivative
        ERI tensor.  ``pair_density_storage=\"hdf5\"`` stores the packed density
        in a temporary HDF5 file under ``scratch_directory`` (or the system temp
        directory) and removes it after contraction.
        """
        Gradient = self._empty_gradient()
        gradient = grad.RHF(self.scfres)
        self._add_one_electron_terms(Gradient, g1_ao, w_ao, gradient)

        nao = self.mol.nao_nr()
        npair = nao * (nao + 1) // 2
        h5file = None
        h5path = None
        pair_density = None
        try:
            if pair_density_storage == "memory":
                pair_density = np.zeros((nao, nao, npair))
            elif pair_density_storage in ("hdf5", "outcore"):
                scratch_directory = scratch_directory or os.environ.get(
                    "PYSCF_TMPDIR", tempfile.gettempdir()
                )
                tmp = tempfile.NamedTemporaryFile(
                    prefix="adcc_pyscf_gradient_", suffix=".h5",
                    dir=scratch_directory, delete=False
                )
                h5path = tmp.name
                tmp.close()
                h5file = h5py.File(h5path, "w")
                pair_density = h5file.create_dataset(
                    "pair_density", shape=(nao, nao, npair), dtype="f8",
                    chunks=(min(nao, 16), min(nao, 16), min(npair, 256))
                )
            else:
                raise ValueError("pair_density_storage needs to be 'memory' "
                                 "or 'hdf5'.")

            g2.to_ao_pair_density(
                refstate, pair_chunk_size=pair_chunk_size, out=pair_density
            )
            Gradient["TEI"] += self._contract_tei_with_packed_density(
                pair_density, shell_chunk_size=shell_chunk_size
            )
        finally:
            if h5file is not None:
                h5file.close()
            if h5path is not None:
                try:
                    os.unlink(h5path)
                except FileNotFoundError:
                    pass
        return self._final_gradient_components(gradient, Gradient)


class PyScfOperatorIntegralProvider:
    def __init__(self, scfres):
        self.scfres = scfres
        self.backend = "pyscf"

    @cached_property
    def electric_dipole(self):
        """-sum_i r_i"""
        return list(
            -1.0 * self.scfres.mol.intor_symmetric('int1e_r', comp=3)
        )

    @property
    def magnetic_dipole(self):
        """
        The imaginary part of the integral is returned.
        -0.5 * sum_i r_i x p_i
        """
        def g_origin_dep_ints_mag_dip(gauge_origin="origin"):
            gauge_origin = _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)
            with self.scfres.mol.with_common_orig(gauge_origin):
                return list(
                    -0.5 * self.scfres.mol.intor('int1e_cg_irxp', comp=3, hermi=2)
                )
        return g_origin_dep_ints_mag_dip

    @cached_property
    def electric_dipole_velocity(self):
        """
        The imaginary part of the integral is returned.
        -sum_i p_i
        """
        with self.scfres.mol.with_common_orig((0.0, 0.0, 0.0)):
            return list(
                self.scfres.mol.intor('int1e_ipovlp', comp=3, hermi=2)
            )

    @property
    def electric_quadrupole(self):
        """-sum_i r_{i, alpha} r_{i, beta}"""
        def g_origin_dep_ints_el_quad(gauge_origin):
            gauge_origin = _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)
            with self.scfres.mol.with_common_orig(gauge_origin):
                r_r = -1.0 * self.scfres.mol.intor_symmetric('int1e_rr', comp=9)
                return list(r_r)
        return g_origin_dep_ints_el_quad

    @property
    def electric_quadrupole_traceless(self):
        """
        -0.5 * sum_i (3 * r_{i, alpha} r_{i, beta}
        - delta_{alpha, beta} r_{i}^2)
        """
        def g_origin_dep_ints_el_quad_traceless(gauge_origin):
            gauge_origin = _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)
            with self.scfres.mol.with_common_orig(gauge_origin):
                r_r = self.scfres.mol.intor_symmetric('int1e_rr', comp=9)
                r_r = np.reshape(r_r, (3, 3, r_r.shape[1], r_r.shape[1]))
                r_quadr = self.scfres.mol.intor_symmetric('int1e_r2', comp=1)
                r_quadr_matrix = np.zeros_like(r_r)
                for i in range(3):
                    r_quadr_matrix[i][i] = r_quadr
                term = 0.5 * (3 * r_r - r_quadr_matrix)
                term = -1.0 * np.reshape(term, (9, r_quadr.shape[0],
                                                r_quadr.shape[0]))
                return list(term)
        return g_origin_dep_ints_el_quad_traceless

    @property
    def electric_quadrupole_velocity(self):
        """
        The imaginary part of the integral is returned.
        -sum_i (r_{i, beta} p_{i, alpha} - i delta_{alpha, beta}
        + r_{i, alpha} p_{i, beta})
        """
        def g_origin_dep_ints_el_quad_vel(gauge_origin):
            gauge_origin = _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)
            with self.scfres.mol.with_common_orig(gauge_origin):
                r_p = self.scfres.mol.intor('int1e_irp', comp=9, hermi=0)
                r_p = np.reshape(r_p, (3, 3, r_p.shape[1], r_p.shape[1]))
                ovlp = self.scfres.mol.intor_symmetric('int1e_ovlp', comp=1)
                ovlp_matrix = np.zeros_like(r_p)
                for i in range(3):
                    ovlp_matrix[i][i] = ovlp
                term = r_p + r_p.transpose(1, 0, 2, 3) - ovlp_matrix
                term = -1.0 * np.reshape(term, (9, ovlp.shape[0], ovlp.shape[0]))
                return list(term)
        return g_origin_dep_ints_el_quad_vel

    @property
    def diamagnetic_magnetizability(self):
        """
        0.25 * sum_i (r_{i, alpha} r_{i, beta}
        - delta_{alpha, beta} r_{i}^2)
        """
        def g_origin_dep_ints_diamag_magn(gauge_origin):
            gauge_origin = _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)
            with self.scfres.mol.with_common_orig(gauge_origin):
                r_r = self.scfres.mol.intor_symmetric('int1e_rr', comp=9)
                r_r = np.reshape(r_r, (3, 3, r_r.shape[1], r_r.shape[1]))
                r_quadr = self.scfres.mol.intor_symmetric('int1e_r2', comp=1)
                r_quadr_matrix = np.zeros_like(r_r)
                for i in range(3):
                    r_quadr_matrix[i][i] = r_quadr
                term = 0.25 * (r_quadr_matrix - r_r)
                term = np.reshape(term, (9, r_quadr.shape[0], r_quadr.shape[0]))
                return list(term)
        return g_origin_dep_ints_diamag_magn

    @property
    def pe_induction_elec(self):
        try:
            self.scfres.with_solvent.cppe_state
        except AttributeError:
            return None

        def pe_induction_elec_ao(dm):
            return self.scfres.with_solvent._exec_cppe(
                dm.to_ndarray(), elec_only=True
            )[1]
        return pe_induction_elec_ao

    @property
    def pcm_potential_elec(self):
        if not hasattr(self.scfres, "with_solvent"):
            return None
        if not isinstance(self.scfres.with_solvent, ddcosmo.DDCOSMO):
            return None

        def pcm_potential_elec_ao(dm):
            # Since eps (dielectric constant) is the only solvent parameter
            # in pyscf and there is no solvent data available in the
            # program, the user needs to adjust scfres.with_solvent.eps
            # manually to the optical dielectric constant (if non
            # equilibrium solvation is desired).
            return self.scfres.with_solvent._B_dot_x(
                dm.to_ndarray()
            )
        return pcm_potential_elec_ao


# TODO: refactor ERI builder to be more general
# IntegralBuilder would be good
class PyScfEriBuilder(EriBuilder):
    def __init__(self, scfres, n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted):
        self.scfres = scfres
        if restricted:
            self.mo_coeff = (self.scfres.mo_coeff, self.scfres.mo_coeff)
        else:
            self.mo_coeff = self.scfres.mo_coeff
        super().__init__(n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted)

    @property
    def coefficients(self):
        return {
            "Oa": self.mo_coeff[0][:, :self.n_alpha],
            "Ob": self.mo_coeff[1][:, :self.n_beta],
            "Va": self.mo_coeff[0][:, self.n_alpha:],
            "Vb": self.mo_coeff[1][:, self.n_beta:],
        }

    def compute_mo_eri(self, blocks, spins):
        coeffs = tuple(self.coefficients[blocks[i] + spins[i]] for i in range(4))
        # TODO Pyscf uses HDF5 internal to do the AO2MO here we read it all
        #      into memory. This wastes memory and could be avoided if temporary
        #      files were used instead. These could be deleted on the call
        #      to `flush_cache` automatically.
        sizes = [i.shape[1] for i in coeffs]
        return ao2mo.general(self.scfres.mol, coeffs,
                             compact=False).reshape(sizes[0], sizes[1],
                                                    sizes[2], sizes[3])


class PyScfHFProvider(HartreeFockProvider):
    """
        This implementation is only valid
        if no orbital reordering is required.
    """
    def __init__(self, scfres):
        # Do not forget the next line,
        # otherwise weird errors result
        super().__init__()
        self.scfres = scfres
        n_alpha, n_beta = scfres.mol.nelec
        self.eri_builder = PyScfEriBuilder(self.scfres, self.n_orbs,
                                           self.n_orbs_alpha, n_alpha,
                                           n_beta, self.restricted)
        self.operator_integral_provider = PyScfOperatorIntegralProvider(
            self.scfres
        )
        self.gradient_provider = PyScfGradientProvider(self.scfres)
        if not self.restricted:
            assert self.scfres.mo_coeff[0].shape[1] == \
                self.scfres.mo_coeff[1].shape[1]

        self.environment = None
        self.environment_implementation = None
        if hasattr(self.scfres, "with_solvent"):
            if hasattr(self.scfres.with_solvent, "cppe_state"):
                self.environment = "pe"
                self.environment_implementation = "cppe"
            elif isinstance(self.scfres.with_solvent, ddcosmo.DDCOSMO):
                self.environment = "pcm"
                self.environment_implementation = "ddcosmo"

    def pe_energy(self, dm, elec_only=True):
        pe_state = self.scfres.with_solvent
        e_pe, _ = pe_state.kernel(dm.to_ndarray(), elec_only=elec_only)
        return e_pe

    def pcm_energy(self, dm):
        # Since eps (dielectric constant) is the only solvent parameter
        # in pyscf and there is no solvent data available in the
        # program, the user needs to adjust scfres.with_solvent.eps
        # manually to the optical dielectric constant (if non
        # equilibrium solvation is desired).
        V_pcm = self.scfres.with_solvent._B_dot_x(dm.to_ndarray())
        return np.einsum("uv,uv->", dm.to_ndarray(), V_pcm)

    @property
    def excitation_energy_corrections(self):
        ret = []
        if self.environment == "pe":
            ptlr = EnergyCorrection(
                "pe_ptlr_correction",
                lambda view: 2.0 * self.pe_energy(view.transition_dm_ao,
                                                  elec_only=True)
            )
            ptss = EnergyCorrection(
                "pe_ptss_correction",
                lambda view: self.pe_energy(view.state_diffdm_ao,
                                            elec_only=True)
            )
            ret.extend([ptlr, ptss])
        elif self.environment == "pcm":
            ptlr = EnergyCorrection(
                "pcm_ptlr_correction",
                lambda view: self.pcm_energy(view.transition_dm_ao)
            )
            ret.extend([ptlr])
        return {ec.name: ec for ec in ret}

    def get_backend(self):
        return "pyscf"

    def get_conv_tol(self):
        if self.scfres.conv_tol_grad is None:
            conv_tol = self.scfres.conv_tol
        else:
            conv_tol = max(self.scfres.conv_tol, self.scfres.conv_tol_grad**2)
        return conv_tol

    def get_restricted(self):
        if isinstance(self.scfres.mo_occ, list):
            restricted = len(self.scfres.mo_occ) < 2
        elif isinstance(self.scfres.mo_occ, np.ndarray):
            restricted = self.scfres.mo_occ.ndim < 2
        else:
            raise InvalidReference("Unusual pyscf SCF class encountered. Could "
                                   "not determine restricted / unrestricted.")
        return restricted

    def get_energy_scf(self):
        return float(self.scfres.e_tot)

    def get_spin_multiplicity(self):
        # Note: In the pyscf world spin is 2S, so the multiplicity
        #       is spin + 1
        return int(self.scfres.mol.spin) + 1

    def get_n_orbs_alpha(self):
        if self.restricted:
            return self.scfres.mo_coeff.shape[1]
        else:
            return self.scfres.mo_coeff[0].shape[1]

    def get_n_bas(self):
        return int(self.scfres.mol.nao_nr())

    def get_nuclear_multipole(self, order, gauge_origin=(0, 0, 0)):
        charges = self.scfres.mol.atom_charges()
        if order == 0:
            # The function interface needs to be a np.array on return
            return np.array([np.sum(charges)])
        elif order == 1:
            coords = self.scfres.mol.atom_coords()
            return np.einsum("i,ix->x", charges, coords)
        elif order == 2:
            coords = self.scfres.mol.atom_coords() - gauge_origin
            r_r = np.einsum("ij,ik->ijk", coords, coords)
            res = np.einsum("i,ijk->jk", charges, r_r)
            res = [res[0, 0], res[0, 1], res[0, 2], res[1, 1], res[1, 2], res[2, 2]]
            return np.array(res)
        else:
            raise NotImplementedError("get_nuclear_multipole with order > 2")

    def transform_gauge_origin_to_xyz(self, gauge_origin):
        return _transform_gauge_origin_to_xyz(self.scfres, gauge_origin)

    def fill_occupation_f(self, out):
        if self.restricted:
            out[:] = np.hstack((self.scfres.mo_occ / 2,
                                self.scfres.mo_occ / 2))
        else:
            out[:] = np.hstack((self.scfres.mo_occ[0],
                                self.scfres.mo_occ[1]))

    def fill_orbcoeff_fb(self, out):
        if self.restricted:
            mo_coeff = (self.scfres.mo_coeff,
                        self.scfres.mo_coeff)
        else:
            mo_coeff = self.scfres.mo_coeff
        out[:] = np.transpose(
            np.hstack((mo_coeff[0], mo_coeff[1]))
        )

    def fill_orben_f(self, out):
        if self.restricted:
            out[:] = np.hstack((self.scfres.mo_energy,
                                self.scfres.mo_energy))
        else:
            out[:] = np.hstack((self.scfres.mo_energy[0],
                                self.scfres.mo_energy[1]))

    def fill_fock_ff(self, slices, out):
        diagonal = np.empty(self.n_orbs)
        self.fill_orben_f(diagonal)
        out[:] = np.diag(diagonal)[slices]

    def fill_eri_ffff(self, slices, out):
        self.eri_builder.fill_slice_symm(slices, out)

    def fill_eri_phys_asym_ffff(self, slices, out):
        raise NotImplementedError("fill_eri_phys_asym_ffff not implemented.")

    def has_eri_phys_asym_ffff(self):
        return False

    def flush_cache(self):
        self.eri_builder.flush_cache()


def import_scf(scfres):
    # TODO The error messages here could be a bit more verbose

    if not isinstance(scfres, scf.hf.SCF):
        raise InvalidReference("Unsupported type for backends.pyscf.import_scf.")

    if not scfres.converged:
        raise InvalidReference("Cannot start an adc calculation on top of an SCF,"
                               " which is not yet converged. Did you forget to"
                               " run the kernel() or the scf() function of the"
                               " pyscf scf object?")

    # TODO Check for point-group symmetry,
    #      check for density-fitting or choleski

    return PyScfHFProvider(scfres)


def run_hf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-11,
           conv_tol_grad=1e-9, max_iter=150, pe_options=None):
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit="Bohr",
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        charge=charge,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
        dump_input=False,
        verbose=0,
    )
    if pe_options:
        from pyscf.solvent import PE
        mf = PE(scf.HF(mol), pe_options)
    else:
        mf = scf.HF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter
    # since we want super tight convergence for tests,
    # tweak the options for non-RHF systems
    if multiplicity != 1:
        mf.max_cycle += 500
        mf.diis = scf.EDIIS()
        mf.diis_space = 3
        mf = scf.addons.frac_occ(mf)
    mf.kernel()
    return mf


def run_core_hole(xyz, basis, charge=0, multiplicity=1,
                  conv_tol=1e-11, conv_tol_grad=1e-9, max_iter=150):
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit="Bohr",
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        charge=charge,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
        dump_input=False,
        verbose=0,
    )

    # First normal run
    mf = scf.UHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter
    # since we want super tight convergence for tests,
    # tweak the options for non-RHF systems
    if multiplicity != 1:
        mf.max_cycle += 500
        mf.diis = scf.EDIIS()
        mf.diis_space = 3
        mf = scf.addons.frac_occ(mf)
    mf.kernel()

    # make beta core hole
    mo0 = tuple(c.copy() for c in mf.mo_coeff)
    occ0 = tuple(o.copy() for o in mf.mo_occ)
    occ0[1][0] = 0.0
    dm0 = mf.make_rdm1(mo0, occ0)

    # Run second SCF with MOM
    mf_chole = scf.UHF(mol)
    scf.addons.mom_occ_(mf_chole, mo0, occ0)
    mf_chole.conv_tol = conv_tol
    mf_chole.conv_tol_grad = conv_tol_grad
    mf_chole.max_cycle += 500
    mf_chole.diis = scf.EDIIS()
    mf_chole.diis_space = 3
    mf_chole.kernel(dm0)
    return mf_chole


def _transform_gauge_origin_to_xyz(scfres, gauge_origin):
    """
    Determines the gauge origin. If the gauge origin is defined as a tuple
    the coordinates need to be given in atomic units!
    """
    coords = scfres.mol.atom_coords()
    masses = scfres.mol.atom_mass_list(isotope_avg=True)
    charges = scfres.mol.atom_charges()
    if gauge_origin == "mass_center":
        gauge_origin = tuple(np.einsum("i,ij->j", masses, coords) / masses.sum())
    elif gauge_origin == "charge_center":
        gauge_origin = tuple(np.einsum("i,ij->j", charges, coords)
                             / charges.sum())
    elif gauge_origin == "origin":
        gauge_origin = (0.0, 0.0, 0.0)
    elif isinstance(gauge_origin, tuple):
        gauge_origin = gauge_origin
    else:
        raise NotImplementedError("The gauge origin can be defined either by a "
                                  "keyword (origin, mass_center or charge_center) "
                                  "or by a tuple defining the Cartesian components "
                                  "e.g. (x, y, z)."
                                  )
    return gauge_origin
