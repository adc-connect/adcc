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
from collections.abc import Callable
from typing import Literal

import numpy as np

import libadcc

from .backends import OperatorIntegralProvider
from .functions import einsum
from .misc import cached_member_function, cached_property
from .MoSpaces import MoSpaces, split_spaces
from .NParticleOperator import OperatorSymmetry
from .OneParticleDensity import OneParticleDensity
from .OneParticleOperator import OneParticleOperator
from .Tensor import Tensor
from .timings import Timer, timed_member_call
from .TwoParticleOperator import TwoParticleOperator
from .typing import (
    Array2D,
    Array4D,
    Coordinate,
    DipoleLike,
    DipoleLikeArray,
    QuadrupoleLike,
    QuadrupoleLikeArray,
    is_dipole_like,
    is_quadruple_like,
)


def transform_operator_ao2mo_1p(
    tensor_bb: libadcc.Tensor,
    tensor_ff: OneParticleOperator,
    coefficients: Callable[[str], libadcc.Tensor],
    tolerance: float = 1e-14,
):
    """
    Take a block-diagonal one-particle operator in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    Parameters
    ----------
    tensor_bb : libadcc.Tensor
        Block-diagonal tensor in the atomic orbital basis with
        dimensionality (2 n_bas, 2 n_bas) and two identical diagonal
        blocks wich are repeated for the alpha and beta spin.
    tensor_ff : OneParticleOperator
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coefficients : Callable[[str], libadcc.Tensor]
        Function providing coefficient blocks
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
    """
    for blk in tensor_ff.canonical_blocks:
        sp1, sp2 = split_spaces(blk)
        cleft = coefficients(sp1 + "b")
        cright = coefficients(sp2 + "b")
        temp = cleft @ tensor_bb @ cright.transpose()

        # TODO: once the permutational symmetry is correct:
        # tensor_ff.set_block(blk, tensor_ff)
        tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), tolerance)


def transform_operator_ao2mo_2p(
    tensor_bb: libadcc.Tensor,
    tensor_ff: TwoParticleOperator,
    coefficients: Callable[[str], libadcc.Tensor],
    tolerance: float = 1e-14,
):
    """
    Take a block-diagonal tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    Parameters
    ----------
    tensor_bb : libadcc.Tensor
        Block-diagonal tensor in the atomic orbital basis with
        dimensionality (2 n_bas, 2 n_bas) and two identical diagonal
        blocks wich are repeated for the alpha and beta spin.
    tensor_ff : TwoParticleOperator
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coefficients : Callable[[str], libadcc.Tensor]
        Function providing coefficient blocks
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
    """
    for blk in tensor_ff.canonical_blocks:
        sp1, sp2, sp3, sp4 = split_spaces(blk)
        cleft_1 = coefficients(sp1 + "b")
        cleft_2 = coefficients(sp2 + "b")
        cright_1 = coefficients(sp3 + "b")
        cright_2 = coefficients(sp4 + "b")
        temp = einsum("ia,jb,abcd,kc,ld->ijkl", cleft_1, cleft_2, tensor_bb, cright_1, cright_2)

        # TODO: once the permutational symmetry is correct:
        # tensor_ff.set_block(blk, tensor_ff)
        tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), tolerance)


def transform_operator_ao2mo_spin_projected_1p(
    tensor_bb: libadcc.Tensor,
    tensor_ff: OneParticleOperator,
    coeff_alpha: Callable[[str], libadcc.Tensor],
    coeff_beta: Callable[[str], libadcc.Tensor],
    spin_block: Literal["aa", "ab", "ba", "bb"] = "aa",
    tolerance: float = 1e-14,
):
    """
    Take a tensor in the atomic orbital basis
    and transform it into the molecular orbital basis in the
    convention used by adcc.

    The transformation is performed block-wise using the provided
    molecular orbital coefficient matrices for the selected
    spin components.

    Parameters
    ----------
    tensor_bb : libadcc.Tensor
        Tensor in the atomic orbital basis
    tensor_ff : np.ndarray[tuple[int, ...], np.dtype[np.float64]]
        Output tensor with the symmetry set-up to contain
        the operator in the molecular orbital representation
    coeff_alpha : Callable[[str], libadcc.Tensor]
        Function providing alpha coefficient blocks.
    coeff_beta : Callable[[str], libadcc.Tensor]
        Function providing beta coefficient blocks.
    spin_block : Literal["aa", "ab", "ba", "bb"], optional
        Two-character string specifying which spin components are projected
        for the left and right indices. Default is "aa".
    conv_tol : float, optional
        SCF convergence tolerance, by default 1e-14
    """
    assert len(spin_block) == 2
    spin1, spin2 = spin_block
    assert spin1 in ["a", "b"] and spin2 in ["a", "b"]
    left = coeff_alpha if spin1 == "a" else coeff_beta
    right = coeff_alpha if spin2 == "a" else coeff_beta

    for blk in tensor_ff.canonical_blocks:
        s1, s2 = split_spaces(blk)
        temp = left(s1 + "b") @ tensor_bb @ right(s2 + "b").transpose()

        # TODO: once the permutational symmetry is correct:
        # tensor_ff.set_block(blk, tensor_ff)
        tensor_ff[blk].set_from_ndarray(temp.to_ndarray(), tolerance)


def replicate_ao_block_1p(
    mospaces: MoSpaces,
    tensor: Array2D,
    symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
    block: Literal["ab", "a"] = "ab",
) -> libadcc.Tensor:
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
    sym = libadcc.make_symmetry_operator_basis(
        mospaces, tensor.shape[0], symmetry.to_str(), 1, block
    )
    result = Tensor(sym)

    if block == "ab":
        zerobk = np.zeros_like(tensor)
        result.set_from_ndarray(np.block([
            [tensor, zerobk],
            [zerobk, tensor],
        ]), 1e-14)  # fmt: skip
    else:
        result.set_from_ndarray(np.block([
            tensor
        ]), 1e-14)  # fmt: skip
    return result


def replicate_ao_block_2p(
    mospaces: MoSpaces,
    tensor: Array4D,
    symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
    block: Literal["ab"] = "ab",
) -> libadcc.Tensor:
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
    assert block == "ab"
    sym = libadcc.make_symmetry_operator_basis(
        mospaces, tensor.shape[0], symmetry.to_str(), 2, block
    )
    result = Tensor(sym)

    zerobk = np.zeros_like(tensor)
    tensor_ex = -tensor.transpose((0, 1, 3, 2))
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
    ])  # fmt: skip
    result.set_from_ndarray(full_tensor, 1e-14)
    return result


class OperatorIntegrals:
    def __init__(
        self,
        provider: OperatorIntegralProvider,
        mospaces: MoSpaces,
        coefficients: Callable[[str], libadcc.Tensor],
        coefficients_alpha: Callable[[str], libadcc.Tensor],
        coefficients_beta: Callable[[str], libadcc.Tensor],
        conv_tol: float,
    ):
        self._provider_ao: OperatorIntegralProvider = provider
        self.mospaces: MoSpaces = mospaces
        self._coefficients: Callable[[str], libadcc.Tensor] = coefficients
        self._coefficients_alpha: Callable[[str], libadcc.Tensor] = coefficients_alpha
        self._coefficients_beta: Callable[[str], libadcc.Tensor] = coefficients_beta
        self._conv_tol: float = conv_tol
        self._import_timer: Timer = Timer()

    @property
    def provider_ao(self) -> OperatorIntegralProvider:
        """
        The data structure which provides the integral data in the
        atomic orbital basis from the backend.
        """
        return self._provider_ao

    @property
    def available(self) -> tuple[str, ...]:
        """Which integrals are available in the underlying backend"""
        return self.provider_ao.available

    def _import_operator_1p(
        self, ao_operator: Array2D, symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> OneParticleOperator:
        """
        Imports the given ao_operator to `adcc` by replicating the two-dimensional array
        in a block diagonal fashion and transform the result to the MO basis.
        """
        op_bb = replicate_ao_block_1p(
            mospaces=self.mospaces, tensor=ao_operator, symmetry=symmetry, block="ab"
        )
        op_ff = OneParticleOperator(self.mospaces, symmetry=symmetry)
        transform_operator_ao2mo_1p(
            tensor_bb=op_bb,
            tensor_ff=op_ff,
            coefficients=self._coefficients,
            tolerance=self._conv_tol,
        )
        return op_ff

    def _import_operator_2p(
        self, ao_operator: Array4D, symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> TwoParticleOperator:
        op_bbbb = replicate_ao_block_2p(
            mospaces=self.mospaces, tensor=ao_operator, symmetry=symmetry, block="ab"
        )
        op_ffff = TwoParticleOperator(self.mospaces, symmetry=symmetry)
        transform_operator_ao2mo_2p(
            tensor_bb=op_bbbb,
            tensor_ff=op_ffff,
            coefficients=self._coefficients,
            tolerance=self._conv_tol,
        )
        return op_ffff

    def _import_dipole_like_operator(
        self, ao_operator: DipoleLikeArray, symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN
    ) -> DipoleLike:
        res = tuple(
            self._import_operator_1p(ao_operator=comp, symmetry=symmetry) for comp in ao_operator
        )
        assert is_dipole_like(res)
        return res

    def _import_quadruple_like_operator(
        self,
        ao_operator: QuadrupoleLikeArray,
        symmetry: OperatorSymmetry = OperatorSymmetry.HERMITIAN,
    ) -> QuadrupoleLike:
        flattened = tuple(
            self._import_operator_1p(ao_operator=comp, symmetry=symmetry) for comp in ao_operator
        )
        res = (tuple(flattened[:3]), tuple(flattened[3:6]), tuple(flattened[6:]))
        assert is_quadruple_like(res)
        return res

    @cached_property
    @timed_member_call("_import_timer")
    def overlap_ao(self) -> libadcc.Tensor:
        """Return the overlap in the atomic orbital basis."""
        ao_operator = self.provider_ao.overlap
        ovlp_bb = replicate_ao_block_1p(
            self.mospaces, ao_operator, symmetry=OperatorSymmetry.HERMITIAN, block="a"
        )
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
        # NOTE: the implementation might also work for CVS. But double check
        # once CVS 2p densities are implemented before removing the
        # exception.
        if "o2" in self.mospaces.subspaces:
            raise NotImplementedError(
                "The 2-particle part of the SSq operator is "
                "only implemented for the occupied and "
                "virtual spaces, i.e., "
                "CVS is not supported yet."
            )
        # Intermediates
        # S^aa, S^ab and S^bb (spin projected overlap matrices)
        # NOTE: For UHF, the diagonal spin blocks of the overlap matrix
        # (in the MO basis) are diagonal.
        # Only the off-diagonal spin blocks contain off-diagonal elements!
        # (For RHF, the off-diagonal spin blocks are also diagonal!)
        # -> apply the UHF assumptions below
        ovlp_bb: libadcc.Tensor = self.overlap_ao

        S_aa = OneParticleOperator(self.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected_1p(
            ovlp_bb, S_aa, self._coefficients_alpha, self._coefficients_beta, "aa", self._conv_tol
        )

        S_ab = OneParticleOperator(self.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected_1p(
            ovlp_bb, S_ab, self._coefficients_alpha, self._coefficients_beta, "ab", self._conv_tol
        )

        S_bb = OneParticleOperator(self.mospaces, symmetry=OperatorSymmetry.NOSYMMETRY)
        transform_operator_ao2mo_spin_projected_1p(
            ovlp_bb, S_bb, self._coefficients_alpha, self._coefficients_beta, "bb", self._conv_tol
        )

        # additional intermediate (is diagonal)
        S_aa_minus_bb = S_aa - S_bb

        op = TwoParticleOperator(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        for block in op.canonical_blocks:
            p, q, r, s = split_spaces(block)
            # + S^ba_pr S^ab_qs + S^ba_qs S^ab_pr
            # - S^ba_ps S^ab_qr - S^ba_qr S^ab_ps
            # first generate the 2 terms with a positive sign:
            # since we only have S^ab: S^ba = (S^ab).transpose
            res = einsum("rp,qs->pqrs", S_ab[r + p], S_ab[q + s])
            if p == q and r == s:
                res = 2.0 * res.symmetrise((0, 1), (2, 3))
            else:
                res += 1.0 * einsum("sq,pr->pqrs", S_ab[s + q], S_ab[p + r])
            # + 0.5 * D_pr D_qs - 0.5 * D_ps D_qr
            # D = S^aa - S^bb is diagonal in the MO basis
            # -> only consider when the spaces match
            if p == r and q == s:
                res += 0.5 * einsum("pr,qs->pqrs", S_aa_minus_bb[p + r], S_aa_minus_bb[q + s])
            # then deal with the terms with negative sign:
            # 2 S^ab terms and the single D term
            if p == q and r == s:
                # NOTE: We only need 1 of the antisymmetrisations to get the
                # correct result. The second one is needed for the correct
                # permutational symmetry in the symmetry object.
                res = 2.0 * res.antisymmetrise(0, 1).antisymmetrise(2, 3)
            elif p == q:
                res = 2.0 * res.antisymmetrise(0, 1)
            elif r == s:
                res = 2.0 * res.antisymmetrise(2, 3)
            else:
                # always subtract the S^ab contributions
                res += (
                    - einsum("sp,qr->pqrs", S_ab[s + p], S_ab[q + r])
                    - einsum("rq,ps->pqrs", S_ab[r + q], S_ab[p + s])
                )  # fmt: skip
                # conditionally subtract the D contribution
                if p == s and q == r:
                    res -= 0.5 * einsum("ps,qr->pqrs", S_aa_minus_bb[p + s], S_aa_minus_bb[q + r])
            op[block] = res
        return op

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole(self) -> DipoleLike:
        """Return the electric dipole integrals in the molecular orbital basis."""
        return self._import_dipole_like_operator(
            ao_operator=self.provider_ao.electric_dipole, symmetry=OperatorSymmetry.HERMITIAN
        )

    @cached_property
    @timed_member_call("_import_timer")
    def electric_dipole_velocity(self) -> DipoleLike:
        """
        Return the electric dipole integrals (in the velocity gauge)
        in the molecular orbital basis.
        """
        return self._import_dipole_like_operator(
            ao_operator=self.provider_ao.electric_dipole_velocity,
            symmetry=OperatorSymmetry.ANTIHERMITIAN,
        )

    # separate the timings, so one can easily see in the timings how many different
    # gauge_origins were used throughout the calculation
    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def magnetic_dipole(self, gauge_origin: Coordinate | str = "origin") -> DipoleLike:
        """
        Returns the magnetic dipole intergrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_dipole_like_operator(
            ao_operator=self.provider_ao.magnetic_dipole(gauge_origin=gauge_origin),
            symmetry=OperatorSymmetry.ANTIHERMITIAN,
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole(self, gauge_origin: Coordinate | str = "origin") -> QuadrupoleLike:
        """
        Returns the electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_quadruple_like_operator(
            ao_operator=self.provider_ao.electric_quadrupole(gauge_origin=gauge_origin),
            symmetry=OperatorSymmetry.HERMITIAN,
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_traceless(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLike:
        """
        Returns the traceless electric quadrupole integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_quadruple_like_operator(
            ao_operator=self.provider_ao.electric_quadrupole_traceless(gauge_origin=gauge_origin),
            symmetry=OperatorSymmetry.HERMITIAN,
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def electric_quadrupole_velocity(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLike:
        """
        Returns the electric quadrupole integrals in velocity gauge
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_quadruple_like_operator(
            ao_operator=self.provider_ao.electric_quadrupole_velocity(gauge_origin=gauge_origin),
            symmetry=OperatorSymmetry.ANTIHERMITIAN,
        )

    @cached_member_function(timer="_import_timer", separate_timings_by_args=True)
    def diamagnetic_magnetizability(
        self, gauge_origin: Coordinate | str = "origin"
    ) -> QuadrupoleLike:
        """
        Returns the diamagnetic magnetizability integrals
        in the molecular orbital basis dependent on the selected gauge origin.
        The default gauge origin is set to (0.0, 0.0, 0.0) (= 'origin').
        """
        return self._import_quadruple_like_operator(
            ao_operator=self.provider_ao.diamagnetic_magnetizability(gauge_origin=gauge_origin),
            symmetry=OperatorSymmetry.HERMITIAN,
        )

    def pe_induction_elec(self, density_mo: OneParticleDensity) -> OneParticleOperator:
        """
        Returns the (density-dependent) PE electronic induction operator in the
        molecular orbital basis.
        """
        dm_ao = sum(density_mo.to_ao_basis())
        assert isinstance(dm_ao, libadcc.Tensor)
        return self._import_operator_1p(
            ao_operator=self.provider_ao.pe_induction_elec(dm=dm_ao),
            symmetry=OperatorSymmetry.HERMITIAN,
        )

    def pcm_potential_elec(self, density_mo: OneParticleDensity) -> OneParticleOperator:
        """
        Returns the (density-dependent) electronic PCM potential operator in the
        molecular orbital basis
        """
        dm_ao = sum(density_mo.to_ao_basis())
        assert isinstance(dm_ao, libadcc.Tensor)
        return self._import_operator_1p(
            ao_operator=self.provider_ao.pcm_potential_elec(dm=dm_ao),
            symmetry=OperatorSymmetry.HERMITIAN,
        )

    @property
    def timer(self) -> Timer:
        ret = Timer()
        ret.attach(self._import_timer, subtree="import")
        return ret
