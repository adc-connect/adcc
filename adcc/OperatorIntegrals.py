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
import numpy as np

from .misc import cached_property, cached_member_function
from .Tensor import Tensor
from .timings import Timer, timed_member_call
from .OneParticleOperator import OneParticleOperator
from .OneParticleDensity import OneParticleDensity
from .NParticleOperator import OperatorSymmetry, NParticleOperator
from .TwoParticleOperator import TwoParticleOperator
from .functions import einsum
from .MoSpaces import split_spaces

import libadcc


def transform_operator_ao2mo(tensor_bb: Tensor, tensor_ff: NParticleOperator,
                             coefficients,
                             conv_tol: float = 1e-14):
    """Take a block-diagonal tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    Parameters
    ----------
    tensor_bb : Tensor
        Block-diagonal tensor in the atomic orbital basis
    tensor_ff : Tensor
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coefficients : callable
        Function providing coefficient blocks
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
    """
    for blk in tensor_ff.canonical_blocks:
        if len(blk) == 4:
            cleft = coefficients(blk[:2] + "b")
            cright = coefficients(blk[2:] + "b")
            temp = cleft @ tensor_bb @ cright.transpose()

            # TODO: once the permutational symmetry is correct:
            # tensor_ff.set_block(blk, tensor_ff)
            tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), conv_tol)

        elif len(blk) == 8:
            cleft_1 = coefficients(blk[:2] + "b")
            cleft_2 = coefficients(blk[2:4] + "b")
            cright_1 = coefficients(blk[4:6] + "b")
            cright_2 = coefficients(blk[6:] + "b")
            temp = einsum("ia,jb,abcd,kc,ld->ijkl",
                          cleft_1, cleft_2, tensor_bb, cright_1, cright_2)

            # TODO: once the permutational symmetry is correct:
            # tensor_ff.set_block(blk, tensor_ff)
            tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), conv_tol)
        else:
            raise NotImplementedError(
                "Only one- and two-particle operators are implemented."
            )


def transform_operator_ao2mo_spin_projected(tensor_bb: Tensor,
                                            tensor_ff: NParticleOperator,
                                            coeff_map: dict[str, Tensor],
                                            spin_map: str = "aa",
                                            conv_tol: float = 1e-14):
    """Take a tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    The transformation is performed block-wise using the provided
    molecular orbital coefficient matrices for the selected
    spin components.

    Parameters
    ----------
    tensor_bb : Tensor
        Tensor in the atomic orbital basis
    tensor_ff : Tensor
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coeff_map : dict
        Dictionary containing molecular orbital coefficient matrices,
        keyed by orbital space and spin label (e.g. "<space>_a", "<space>_b").
    spin_map : str, optional
        Two-character string specifying which spin components are projected
        for the left and right indices (e.g. "aa", "ab"). Default is "aa".
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
    """
    assert len(spin_map) == 2
    spin1, spin2 = list(spin_map)

    for blk in tensor_ff.canonical_blocks:
        if len(blk) == 4:
            s1, s2 = split_spaces(blk)
            cleft = coeff_map[f"{s1}_{spin1}"]
            cright = coeff_map[f"{s2}_{spin2}"]
            temp = cleft @ tensor_bb @ cright.transpose()

            # TODO: once the permutational symmetry is correct:
            # tensor_ff.set_block(blk, tensor_ff)
            tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), conv_tol)
        else:
            raise NotImplementedError


def replicate_ao_block(mospaces, tensor,
                       symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
                       block: str = "ab"):
    """
    transform_operator_ao2mo requires the operator in the AO basis to be
    replicated in a block-diagonal fashion (e.g. for a OneParticleOperator:
    [A  0
     0  A]).
    This is achieved using this function.

    The `block` argument controls which blocks are constructed:
    - block="ab": replicate the operator for both alpha and beta spaces,
      resulting in a full block-diagonal structure.
    - block="a": construct only the corresponding single block.
    """
    assert block in ["ab", "a"]
    zerobk = np.zeros_like(tensor)
    if len(tensor.shape) == 2:
        sym = libadcc.make_symmetry_operator_basis(
            mospaces, tensor.shape[0], symmetry.to_str(), 1, block
        )
        result = Tensor(sym)

        if block == "ab":
            result.set_from_ndarray(np.block([
                [tensor, zerobk],
                [zerobk, tensor],
            ]), 1e-14)
        else:
            result.set_from_ndarray(np.block([
                tensor
            ]), 1e-14)
    elif len(tensor.shape) == 4:
        sym = libadcc.make_symmetry_operator_basis(
            mospaces, tensor.shape[0], symmetry.to_str(), 2, block
        )
        result = Tensor(sym)
        if block == "ab":
            tensor_ex = - tensor.transpose((0, 1, 3, 2))
            tensor_as = tensor + tensor_ex
            full_tensor = np.block([
                [
                    [  # [aaaa, aaab], [aaba aabb]
                        [tensor_as, zerobk],
                        [zerobk, zerobk],
                    ],
                    [  # [abaa, abab], [abba, abbb]
                        [zerobk, tensor],
                        [tensor_ex, zerobk],
                    ],
                ],
                [
                    [  # [baaa, baab], [baba, babb]
                        [zerobk, tensor_ex],
                        [tensor, zerobk],
                    ],
                    [  # [bbaa, bbab], [bbba, bbbb]
                        [zerobk, zerobk],
                        [zerobk, tensor_as],
                    ],
                ],
            ])
            result.set_from_ndarray(full_tensor, 1e-14)
        else:
            raise NotImplementedError(
                f"Invalid block {block}. Only 'ab' supported for 2-particle "
                "operators."
            )
    else:
        raise NotImplementedError("Only 2 and 4 dimensional tensors supported.")
    return result


class OperatorIntegrals:
    def __init__(self, provider, mospaces, coefficients, coefficients_alpha,
                 coefficients_beta, conv_tol):
        self._provider_ao = provider
        self.mospaces = mospaces
        self._coefficients = coefficients
        self._coefficients_alpha = coefficients_alpha
        self._coefficients_beta = coefficients_beta
        self._conv_tol = conv_tol
        self._import_timer = Timer()

    @property
    def provider_ao(self):
        """
        The data structure which provides the integral data in the
        atomic orbital basis from the backend.
        """
        return self._provider_ao

    @property
    def available(self) -> tuple[str, ...]:
        """Which integrals are available in the underlying backend"""
        return self.provider_ao.available

    @cached_property
    @timed_member_call("_import_timer")
    def overlap_ao(self) -> Tensor:
        """Return the overlap in the atomic orbital basis."""
        if "overlap" not in self.available:
            raise NotImplementedError(f"overlap operator not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, "overlap")
        ovlp_bb = replicate_ao_block(self.mospaces, ao_operator,
                                     symmetry=OperatorSymmetry.HERMITIAN,
                                     block="a")
        return ovlp_bb

    @cached_property
    @timed_member_call("_import_timer")
    def ssq_1p(self) -> OneParticleOperator:
        """Returns the one-particle part of the S^2 operator"""
        op = OneParticleOperator(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        # d_ij = 3/4 \delta_ij
        for ss in self.mospaces.subspaces_occupied:
            op[ss + ss].set_mask("ii", 0.75)
        # d_ab = 3/4 \delta_ij
        for ss in self.mospaces.subspaces_virtual:
            op[ss + ss].set_mask("ii", 0.75)
        return op

    @cached_property
    @timed_member_call("_import_timer")
    def ssq_2p(self) -> TwoParticleOperator:
        """Returns the two-particle part of the S^2 operator"""
        # currently not implemented for CVS
        if any(sp not in ["o1", "v1"] for sp in self.mospaces.subspaces):
            raise NotImplementedError("The 2-particle part of the SSq operator is "
                                      "only implemented for the occupied and "
                                      "virtual spaces, i.e., "
                                      "CVS is not supported yet.")
        # Intermediates
        # S^aa, S^ab and S^bb (spin projected overlap matrices)
        ovlp_bb: Tensor = self.overlap_ao
        coeff_map = {}
        for sp in self.mospaces.subspaces:
            coeff_map[sp + "_a"] = self._coefficients_alpha(sp + "b")
            coeff_map[sp + "_b"] = self._coefficients_beta(sp + "b")

        S_aa = OneParticleOperator(self.mospaces,
                                   symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected(ovlp_bb, S_aa, coeff_map, "aa",
                                                self._conv_tol)

        S_ab = OneParticleOperator(self.mospaces,
                                   symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected(ovlp_bb, S_ab, coeff_map, "ab",
                                                self._conv_tol)

        S_bb = OneParticleOperator(self.mospaces,
                                   symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected(ovlp_bb, S_bb, coeff_map, "bb",
                                                self._conv_tol)

        # additional intermediate
        S_aa_minus_bb = S_aa - S_bb

        op = TwoParticleOperator(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        op.oooo = (
            + 1.0 * einsum("ik,jl->ijkl", S_aa_minus_bb.oo,  S_aa_minus_bb.oo)
            # exploit symmetry S_ab.oo = S_ba.oo.T
            + 4.0 * einsum("ik,jl->ijkl", S_ab.oo.T, S_ab.oo)
        ).antisymmetrise(2, 3)
        op.ooov = (
            + 2.0 * einsum("ik,ja->ijka", S_ab.oo.T, S_ab.ov)
            + 2.0 * einsum("ik,ja->ijka", S_ab.oo, S_ab.vo.T)
        ).antisymmetrise(0, 1)
        op.oovv = (
            + 2.0 * einsum("ia,jb->ijab", S_ab.vo.T, S_ab.ov)
            + 2.0 * einsum("ia,jb->ijab", S_ab.ov, S_ab.vo.T)
        ).antisymmetrise(2, 3)
        op.ovov = (
            + 0.5 * einsum("ij,ab->iajb", S_aa_minus_bb.oo, S_aa_minus_bb.vv)
            + 1.0 * einsum("ij,ab->iajb", S_ab.oo.T, S_ab.vv)
            + 1.0 * einsum("ij,ab->iajb", S_ab.oo, S_ab.vv.T)
            - 1.0 * einsum("ib,aj->iajb", S_ab.vo.T, S_ab.vo)
            - 1.0 * einsum("ib,aj->iajb", S_ab.ov, S_ab.ov.T)
        )
        op.ovvv = (
            + 2.0 * einsum("ib,ac->iabc", S_ab.vo.T, S_ab.vv)
            + 2.0 * einsum("ib,ac->iabc", S_ab.ov, S_ab.vv.T)
        ).antisymmetrise(2, 3)
        op.vvvv = (
            + 1.0 * einsum("ac,bd->abcd", S_aa_minus_bb.vv, S_aa_minus_bb.vv)
            # exploit symmetry S_ab.vv = S_ba.vv.T
            + 4.0 * einsum("ac,bd->abcd", S_ab.vv, S_ab.vv.T)
        ).antisymmetrise(2, 3)
        return op

    def _import_dipole_like_operator(
        self, integral: str,
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> tuple[OneParticleOperator, ...]:
        if integral not in self.available:
            raise NotImplementedError(f"{integral.replace('_', ' ')} operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)
        assert len(ao_operator) == 3  # has to have a x, y and z component

        dipoles = []
        for comp in range(3):  # [x, y, z]
            dip_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                        symmetry=symmetry)
            dip_ff = OneParticleOperator(self.mospaces, symmetry=symmetry)
            transform_operator_ao2mo(dip_bb, dip_ff, self._coefficients,
                                     self._conv_tol)
            dipoles.append(dip_ff)
        return tuple(dipoles)

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole(self) -> tuple[OneParticleOperator, ...]:
        """Return the electric dipole integrals in the molecular orbital basis."""
        return self._import_dipole_like_operator(
            "electric_dipole", symmetry=OperatorSymmetry.HERMITIAN)

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole_velocity(self) -> tuple[OneParticleOperator, ...]:
        """
        Return the electric dipole integrals (in the velocity gauge)
        in the molecular orbital basis.
        """
        return self._import_dipole_like_operator(
            "electric_dipole_velocity", symmetry=OperatorSymmetry.ANTIHERMITIAN)

    def _import_g_origin_dep_dip_like_operator(
            self, integral: str, gauge_origin="origin",
            symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> tuple[OneParticleOperator, ...]:
        """
        Imports the operator and transforms it to the molecular orbital basis.

        Parameters
        ----------
        integral : str
            The dipole like gauge dependent integral to import: an integral
            that consists of 3 components (x, y, z) and whose AO import function
            takes the gauge origin as argument.
        gauge_origin: str or tuple[str]
            The gauge origin used for the generation of the AO integrals.
        is_symmetric : bool, optional
            If the imported operator is symmetric, by default True
        """
        if integral not in self.available:
            raise NotImplementedError(f"{integral} operator is not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)(gauge_origin)
        assert len(ao_operator) == 3  # has to have a x, y and z component

        dipoles = []
        for comp in range(3):  # [x, y, z]
            dip_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                        symmetry=symmetry)
            dip_ff = OneParticleOperator(self.mospaces, symmetry=symmetry)
            transform_operator_ao2mo(dip_bb, dip_ff, self._coefficients,
                                     self._conv_tol)
            dipoles.append(dip_ff)
        return tuple(dipoles)

    # separate the timings, so one can easily see in the timings how many different
    # gauge_origins were used throughout the calculation
    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def magnetic_dipole(self, gauge_origin="origin"
                        ) -> tuple[OneParticleOperator, ...]:
        """
        Returns the magnetic dipole intergrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_dip_like_operator(
            integral="magnetic_dipole", gauge_origin=gauge_origin,
            symmetry=OperatorSymmetry.ANTIHERMITIAN
        )

    def _import_g_origin_dep_quad_like_operator(
        self, integral: str, gauge_origin="origin",
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> tuple[tuple[OneParticleOperator, ...], ...]:  # noqa E501
        """
        Imports the operator and transforms it to the molecular orbital basis.

        Parameters
        ----------
        integral : str
            The quadrupole like gauge dependent integral to import: an integral
            that consists of 9 components (xx, xy, xz, ... zz)
            and whose AO import function takes the gauge origin as single argument.
        gauge_origin: str or tuple[str]
            The gauge origin used for the generation of the AO integrals.
        is_symmetric : bool, optional
            if the imported operator is symmetric, by default True
        """
        if integral not in self.available:
            raise NotImplementedError(f"{integral} operator is not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)(gauge_origin)
        assert len(ao_operator) == 9

        flattened = []
        for comp in range(9):  # [xx, xy, xz, yx, yy, yz, zx, zy, zz]
            quad_bb = replicate_ao_block(self.mospaces, ao_operator[comp],
                                         symmetry=symmetry)
            quad_ff = OneParticleOperator(
                self.mospaces, symmetry=symmetry
            )
            transform_operator_ao2mo(quad_bb, quad_ff, self._coefficients,
                                     self._conv_tol)
            flattened.append(quad_ff)
        return (tuple(flattened[:3]), tuple(flattened[3:6]), tuple(flattened[6:]))

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole(self, gauge_origin="origin"
                            ) -> tuple[tuple[OneParticleOperator, ...], ...]:
        """
        Returns the electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole", gauge_origin=gauge_origin,
            symmetry=OperatorSymmetry.HERMITIAN
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_traceless(self, gauge_origin="origin"
                                      ) -> tuple[tuple[OneParticleOperator, ...], ...]:  # noqa E501
        """
        Returns the traceless electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole_traceless", gauge_origin=gauge_origin,
            symmetry=OperatorSymmetry.HERMITIAN
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_velocity(self, gauge_origin="origin"
                                     ) -> tuple[tuple[OneParticleOperator, ...], ...]:  # noqa E501
        """
        Returns the electric quadrupole integrals in velocity gauge
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="electric_quadrupole_velocity", gauge_origin=gauge_origin,
            symmetry=OperatorSymmetry.ANTIHERMITIAN
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def diamagnetic_magnetizability(self, gauge_origin="origin"
                                    ) -> tuple[tuple[OneParticleOperator, ...], ...]:  # noqa E501
        """
        Returns the diamagnetic magnetizability integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_g_origin_dep_quad_like_operator(
            integral="diamagnetic_magnetizability", gauge_origin=gauge_origin,
            symmetry=OperatorSymmetry.HERMITIAN
        )

    def _import_density_dependent_operator(
        self, operator: str, density_mo: OneParticleDensity,
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> OneParticleDensity:
        """
        Import the density-dependent operator and transform it to the
        molecular orbital basis.

        Parameters
        ----------
        integral : str
            The density-dependent operator to import: an operator
            whose AO import function takes a density matrix as single argument.
        density_mo: OneParticleDensity
            The density in the MO basis for which to compute the operator.
        is_symmetric : bool, optional
            if the imported operator is symmetric, by default True
        """
        dm_ao = sum(density_mo.to_ao_basis())
        v_ao = getattr(self.provider_ao, operator)(dm_ao)
        v_bb = replicate_ao_block(
            self.mospaces, v_ao, symmetry=symmetry
        )
        v_ff = OneParticleDensity(self.mospaces, symmetry=symmetry)
        transform_operator_ao2mo(
            v_bb, v_ff, self._coefficients, self._conv_tol
        )
        return v_ff

    def pe_induction_elec(self,
                          density_mo: OneParticleDensity) -> OneParticleDensity:
        """
        Returns the (density-dependent) PE electronic induction operator in the
        molecular orbital basis.
        """
        if "pe_induction_elec" not in self.available:
            raise NotImplementedError("PE electronic induction operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")
        return self._import_density_dependent_operator(
            operator="pe_induction_elec", density_mo=density_mo,
            symmetry=OperatorSymmetry.HERMITIAN
        )

    def pcm_potential_elec(self,
                           density_mo: OneParticleDensity) -> OneParticleDensity:
        """
        Returns the (density-dependent) electronic PCM potential operator in the
        molecular orbital basis
        """
        if "pcm_potential_elec" not in self.available:
            raise NotImplementedError("Electronic PCM potential operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")
        return self._import_density_dependent_operator(
            operator="pcm_potential_elec", density_mo=density_mo,
            symmetry=OperatorSymmetry.HERMITIAN
        )

    def _import_2p_like_operator(
        self, integral: str,
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> TwoParticleOperator:
        if integral not in self.available:
            raise NotImplementedError(f"{integral.replace('_', ' ')} operator "
                                      "not implemented "
                                      f"in {self.provider_ao.backend} backend.")

        ao_operator = getattr(self.provider_ao, integral)

        op_bbbb = replicate_ao_block(self.mospaces, ao_operator,
                                     symmetry=symmetry)

        op_ffff = TwoParticleOperator(self.mospaces, symmetry=symmetry)
        transform_operator_ao2mo(op_bbbb, op_ffff, self._coefficients,
                                 self._conv_tol)
        return op_ffff

    @property
    def timer(self):
        ret = Timer()
        ret.attach(self._import_timer, subtree="import")
        return ret
