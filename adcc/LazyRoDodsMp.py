## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
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

import libadcc

from . import block as b
from .functions import direct_sum, einsum
from .GroundState import GroundState
from .misc import cached_member_function
from .MoSpaces import split_spaces
from .NParticleOperator import OperatorSymmetry
from .OneParticleDensity import OneParticleDensity
from .TwoParticleDensity import TwoParticleDensity


class LazyRoDodsMp(GroundState):
    @cached_member_function()
    def ts1(self, space: str, apply_cvs: bool = False) -> libadcc.Tensor:
        split = split_spaces(space)
        if len(split) != 2 or split[1] != b.v:
            raise ValueError(f"Invalid space '{space}' for first-order singles amplitudes.")
        # negative sign, because the denominator has to be (e_i - e_a) instead of (e_a - e_i)
        return -self.reference_state.operators.fock[space] / self.df(space)

    @cached_member_function()
    def td1(self, space: str) -> libadcc.Tensor:
        hf = self.reference_state
        sp = split_spaces(space)
        if len(sp) != 4 or sp[2] != b.v or sp[3] != b.v:
            raise ValueError(f"Invalid space '{space}' for first-order doubles amplitudes.")
        # since ia and jb are connected on the df objects, it is sufficient
        # to only symmetrise the virtual orbitals resulting in
        # ia jb + ib ja
        # which is identical to the expression if only the occupied indices
        # would have been symmetrised
        # -> resulting expression has both ij and ab symmetry
        # if we instead would build the denominator from orbital energies
        # occupied and virtual indices both have to be symmetrized!
        return hf.eri(space) / direct_sum(
            "ia,jb->ijab", self.df(sp[0] + sp[2]), self.df(sp[1] + sp[3])
        ).symmetrise((2, 3))

    @cached_member_function()
    def ts2(self, space: str, apply_cvs: bool = False) -> libadcc.Tensor:
        if space != b.ov:
            raise NotImplementedError(
                f"Second-order singles amplitudes not implemented for space '{space}'"
            )
        if self.has_core_occupied_space or apply_cvs:
            raise NotImplementedError(
                "Second-order singles amplitudes not implemented for CVS-RO-DODS-MP"
            )
        # scaling: N^5 -> identical to normal MP
        hf = self.reference_state
        F = hf.operators.fock
        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        # minus sign: orbital energy denominator
        return - (
            + 1 * einsum("ib,ab->ia", t1_1, F.vv)  # N^3: O^1V^2 / N^2: V^2
            + 0.5 * einsum("jkab,jkib->ia", t2_1, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijbc,jabc->ia", t2_1, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ja,ij->ia", t1_1, F.oo)  # N^3: O^2V^1 / N^2: O^1V^1
            - 1 * einsum("jb,ibja->ia", t1_1, hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ijab,jb->ia", t2_1, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * F.ov  # N^2: O^1V^1 / N^2: O^1V^1
        ) / self.df(b.ov)  # fmt: skip

    @cached_member_function()
    def td2(self, space: str) -> libadcc.Tensor:
        if space != b.oovv:
            raise NotImplementedError(
                f"Second-order doubles amplitudes not implemented for space '{space}'"
            )
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "Second-order doubles amplitudes not implemented for CVS-RO-DODS-MP"
            )
        hf = self.reference_state
        F = hf.operators.fock
        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        denom = direct_sum("ia,jb->ijab", self.df(b.ov), self.df(b.ov)).symmetrise(0, 1)
        return (
            # The scaling comment is given as: [comp_scaling] / [mem_scaling]
            # Apply (1 - P_ij - P_ab + P_ijP_ab) to:
            4 * (
                + 1 * einsum("ia,jb->ijab", t1_1, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum("ikac,jckb->ijab", t2_1, hf.ovov)  # N^6: O^3V^3 / N^4: O^2V^2
            )
            # Apply (1 - P_ab) to:
            + 2 * (
                + 1 * einsum("kb,ijka->ijab", t1_1, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("ijbc,ac->ijab", t2_1, F.vv)  # N^5: O^2V^3 / N^4: O^2V^2
            )
            # Apply (1 - P_ij) to:
            + 2 * (
                + 1 * einsum("jc,icab->ijab", t1_1, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
                + 1 * einsum("ikab,jk->ijab", t2_1, F.oo)  # N^5: O^3V^2 / N^4: O^2V^2
            )
            - 0.5 * einsum("klab,ijkl->ijab", t2_1, hf.oooo)  # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijcd,abcd->ijab", t2_1, hf.vvvv)  # N^6: O^2V^4 / N^4: V^4
            + 1 * hf.oovv  # N^4: O^2V^2 / N^4: O^2V^2
        ).antisymmetrise(0, 1).antisymmetrise(2, 3) / denom  # fmt: skip

    @cached_member_function()
    def tt2(self, space: str) -> libadcc.Tensor:
        if space != b.ooovvv:
            raise NotImplementedError(
                f"Second-order triples amplitudes not implemented for space '{space}'"
            )
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "Second-order triples amplitudes not implemented for CVS-RO-DODS-MP"
            )
        hf = self.reference_state
        F = hf.operators.fock
        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        df = self.df(b.ov)
        denom = (
            direct_sum("ia,jkbc->ijkabc", df, direct_sum("jb,kc->jkbc", df, df))
            .symmetrise(0, 1, 2)
            .symmetrise(3, 4, 5)
        )
        # minus sign: orbital energy denominator
        return - 9 * (
            # The scaling comment is given as: [comp_scaling] / [mem_scaling]
            # Apply (1 - P_ij - P_ik - P_ab - P_ac + P_ijP_ab + P_ijP_ac + P_ikP_ab + P_ikP_ac) to:
            + 1 * einsum("ia,jkbc->ijkabc", t1_1, hf.oovv)  # N^6: O^3V^3 / N^6: O^3V^3
            # Apply (1 - P_ij - P_jk - P_ac - P_bc + P_ijP_ac + P_ijP_bc + P_jkP_ac + P_jkP_bc) to:
            + 1 * einsum("ikab,jc->ijkabc", t2_1, F.ov)  # N^6: O^3V^3 / N^6: O^3V^3
            # Apply (1 - P_ij - P_ik - P_ac - P_bc + P_ijP_ac + P_ijP_bc + P_ikP_ac + P_ikP_bc) to:
            + 1 * einsum("ilab,jklc->ijkabc", t2_1, hf.ooov)  # N^7: O^4V^3 / N^6: O^3V^3
            # Apply (1 - P_ik - P_jk - P_ab - P_ac + P_ikP_ab + P_ikP_ac + P_jkP_ab + P_jkP_ac) to:
            + 1 * einsum("ijad,kdbc->ijkabc", t2_1, hf.ovvv)  # N^7: O^3V^4 / N^6: O^3V^3
        ).antisymmetrise(0, 1, 2).antisymmetrise(3, 4, 5) / denom  # fmt: skip

    @cached_member_function()
    def ts3(self, space: str) -> libadcc.Tensor:
        if space != b.ov:
            raise NotImplementedError(
                f"Third-order singles amplitudes not implemented for space '{space}'"
            )
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "Third-order singles amplitudes not implemented for CVS-RO-DODS-MP"
            )
        hf = self.reference_state
        F = hf.operators.fock
        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        t1_2 = self.ts2(b.ov)
        t2_2 = self.td2(b.oovv)
        t3_2 = self.tt2(b.ooovvv)
        # minus sign: orbital energy denominator
        return - (
            + 1 * einsum("ib,ab->ia", t1_1, F.vv)  # N^3: O^1V^2 / N^2: V^2
            + 1 * einsum("ib,ab->ia", t1_2, F.vv)  # N^3: O^1V^2 / N^2: V^2
            + 0.5 * einsum("jkab,jkib->ia", t2_1, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijbc,jabc->ia", t2_1, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            + 0.5 * einsum("jkab,jkib->ia", t2_2, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
            + 0.5 * einsum("ijbc,jabc->ia", t2_2, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
            - 1 * einsum("ja,ij->ia", t1_1, F.oo)  # N^3: O^2V^1 / N^2: O^1V^1
            - 1 * einsum("jb,ibja->ia", t1_1, hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ijab,jb->ia", t2_1, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ja,ij->ia", t1_2, F.oo)  # N^3: O^2V^1 / N^2: O^1V^1
            - 1 * einsum("jb,ibja->ia", t1_2, hf.ovov)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ijab,jb->ia", t2_2, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkabc,jkbc->ia", t3_2, hf.oovv)  # N^6: O^3V^3 / N^6: O^3V^3
            - 1 * einsum("jb,jb->", t1_1, F.ov) * t1_1  # N^2: O^1V^1 / N^2: O^1V^1
            + 0.25 * einsum("jkbc,jkbc->", t2_1, hf.oovv) * t1_1  # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * F.ov  # N^2: O^1V^1 / N^2: O^1V^1
        ) / self.df(b.ov)  # fmt: skip

    @cached_member_function()
    def td3(self, space: str) -> libadcc.Tensor:
        if space != b.oovv:
            raise NotImplementedError(
                f"Third-order doubles amplitudes not implemented for space '{space}'"
            )
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "Third-order doubles amplitudes not implemented for CVS-RO-DODS-MP"
            )
        hf = self.reference_state
        F = hf.operators.fock
        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        t1_2 = self.ts2(b.ov)
        t2_2 = self.td2(b.oovv)
        t3_2 = self.tt2(b.ooovvv)
        denom = direct_sum("ia,jb->ijab", self.df(b.ov), self.df(b.ov)).symmetrise(0, 1)
        return (
            # The scaling comment is given as: [comp_scaling] / [mem_scaling]
            # Apply (1 - P_ij - P_ab + P_ijP_ab) to:
            4 * (
                + 1 * einsum("ia,jb->ijab", t1_1, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum("ikac,jckb->ijab", t2_1, hf.ovov)  # N^6: O^3V^3 / N^4: O^2V^2
                + 1 * einsum("ia,jb->ijab", t1_2, F.ov)  # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum("ikac,jckb->ijab", t2_2, hf.ovov)  # N^6: O^3V^3 / N^4: O^2V^2
            )
            # Apply (1 - P_ab) to:
            + 2 * (
                + 1 * einsum("kb,ijka->ijab", t1_1, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("ijbc,ac->ijab", t2_1, F.vv)  # N^5: O^2V^3 / N^4: O^2V^2
                + 1 * einsum("kb,ijka->ijab", t1_2, hf.ooov)  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("ijbc,ac->ijab", t2_2, F.vv)  # N^5: O^2V^3 / N^4: O^2V^2
                + 0.5 * einsum("ijkbcd,kacd->ijab", t3_2, hf.ovvv)  # N^7: O^3V^4 / N^6: O^3V^3
                # N^5: O^2V^3 / N^4: O^2V^2
                - 0.5 * einsum("bc,ijac->ijab", einsum("klbd,klcd->bc", t2_1, hf.oovv), t2_1)
            )
            # Apply (1 - P_ij) to:
            + 2 * (
                + 1 * einsum("jc,icab->ijab", t1_1, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
                + 1 * einsum("ikab,jk->ijab", t2_1, F.oo)  # N^5: O^3V^2 / N^4: O^2V^2
                + 1 * einsum("jc,icab->ijab", t1_2, hf.ovvv)  # N^5: O^2V^3 / N^4: O^1V^3
                + 1 * einsum("ikab,jk->ijab", t2_2, F.oo)  # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("jklabc,klic->ijab", t3_2, hf.ooov)  # N^7: O^4V^3 / N^6: O^3V^3
                # N^6: O^3V^3 / N^4: O^2V^2
                - 1 * einsum("ikad,jkbd->ijab", einsum("ilac,klcd->ikad", t2_1, hf.oovv), t2_1)
                # N^5: O^3V^2 / N^4: O^2V^2
                - 0.5 * einsum("jk,ikab->ijab", einsum("jlcd,klcd->jk", t2_1, hf.oovv), t2_1)
            )
            # Apply 1 to:
            + 1 * einsum("ijkabc,kc->ijab", t3_2, F.ov)  # N^6: O^3V^3 / N^6: O^3V^3
            - 0.5 * einsum("klab,ijkl->ijab", t2_1, hf.oooo)  # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijcd,abcd->ijab", t2_1, hf.vvvv)  # N^6: O^2V^4 / N^4: V^4
            - 0.5 * einsum("klab,ijkl->ijab", t2_2, hf.oooo)  # N^6: O^4V^2 / N^4: O^2V^2
            - 0.5 * einsum("ijcd,abcd->ijab", t2_2, hf.vvvv)  # N^6: O^2V^4 / N^4: V^4
            + 1 * einsum("kc,kc->", t1_1, F.ov) * t2_1  # N^4: O^2V^2 / N^4: O^2V^2
            # N^6: O^4V^2 / N^4: O^2V^2
            + 0.25 * einsum("ijkl,klab->ijab", einsum("ijcd,klcd->ijkl", t2_1, hf.oovv), t2_1)
            + 1 * hf.oovv  # N^4: O^2V^2 / N^4: O^2V^2
        ).antisymmetrise(0, 1).antisymmetrise(2, 3) / denom  # fmt: skip

    @cached_member_function()
    def energy_correction(self, level: int = 2) -> float:
        if level <= 0:
            raise ValueError(f"Level has to be a positive non-zero integer, got {level}.")
        if self.has_core_occupied_space:
            raise NotImplementedError("CVS-RO-DODS-MP energy corrections not implemented.")

        if level == 2:
            ts = self.ts1
            td = self.td1
        elif level == 3:
            ts = self.ts2
            td = self.td2
        elif level == 4:
            ts = self.ts3
            td = self.td3
        else:
            raise NotImplementedError(
                f"RO-DODS-MP energy correction not implemented for level {level}"
            )
        return ts(b.ov).dot(self.reference_state.operators.fock.ov) - 0.25 * td(b.oovv).dot(
            self.reference_state.oovv
        )

    def energy(self, level: int = 2) -> float:
        if level < 0:
            raise ValueError(f"Level has to be a positive integer, got {level}.")
        elif level == 0:
            # sum of orbital energies
            # TODO: core orbitals and frozen core
            return self.reference_state.nuclear_repulsion_energy + np.einsum(
                "i->", self.reference_state.foo.diagonal().to_ndarray()
            )
        # collect all the energy corrections
        res = self.reference_state.energy_scf
        for lv in range(2, level + 1):
            res += self.energy_correction(lv)
        return res

    def diffdm(self, level: int = 2, apply_cvs: bool = False) -> OneParticleDensity:
        corrections = (
            self.first_order_dm_correction,
            self.second_order_dm_correction,
            self.third_order_dm_correction,
        )
        if level < 1:
            raise ValueError(f"Difference density of order {level} vanishes.")
        elif level > len(corrections):
            raise NotImplementedError(f"Difference density of order {level} is not implemented.")
        # copy to avoid cache corruption
        ret = corrections[0](apply_cvs).copy()
        for correction in corrections[1:level]:
            ret += correction(apply_cvs)
        assert isinstance(ret, OneParticleDensity)
        return ret

    def density(self, level: int = 2, apply_cvs: bool = False) -> OneParticleDensity:
        density = self.reference_state.density
        if level == 0:
            return density
        return density + self.diffdm(level, apply_cvs=apply_cvs)

    def diffdm_2p(self, level: int = 2, apply_cvs: bool = False) -> TwoParticleDensity:
        corrections = (
            self.first_order_dm_correction_2p,
            self.second_order_dm_correction_2p,
        )
        if level < 1:
            raise ValueError(f"2-particle difference density of order {level} vanishes.")
        elif level > len(corrections):
            raise NotImplementedError(
                f"2-particle difference density of order {level} is not implemented."
            )
        # copy to avoid cache corruption
        ret = corrections[0](apply_cvs).copy()
        for correction in corrections[1:level]:
            ret += correction(apply_cvs)
        assert isinstance(ret, TwoParticleDensity)
        return ret

    def density_2p(self, level: int = 2, apply_cvs: bool = False) -> TwoParticleDensity:
        density = self.reference_state.density_2p
        if level == 0:
            return density
        return density + self.diffdm_2p(level, apply_cvs=apply_cvs)

    @cached_member_function()
    def first_order_dm_correction(self, apply_cvs: bool = False) -> OneParticleDensity:
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "First-order difference density correction not implemented for CVS-RO-DODS-MP"
            )
        assert not apply_cvs
        ret = OneParticleDensity(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        ret.ov = self.ts1(b.ov)
        ret.reference_state = self.reference_state
        return ret

    @cached_member_function()
    def second_order_dm_correction(self, apply_cvs: bool = False) -> OneParticleDensity:
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "First-order difference density correction not implemented for CVS-RO-DODS-MP"
            )
        assert not apply_cvs

        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        t1_2 = self.ts2(b.ov)

        ret = OneParticleDensity(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        ret.oo = (
            - 1 * einsum("ia,ja->ij", t1_1, t1_1)  # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ikab,jkab->ij", t2_1, t2_1)  # N^5: O^3V^2 / N^4: O^2V^2
        )  # fmt: skip
        ret.ov = (
            - 1 * einsum("jb,ijab->ia", t1_1, t2_1)  # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * t1_2  # N^2: O^1V^1 / N^2: O^1V^1
        )  # fmt: skip
        ret.vv = (
            + 1 * einsum("ia,ib->ab", t1_1, t1_1)  # N^3: O^1V^2 / N^2: V^2
            + 0.5 * einsum("ijac,ijbc->ab", t2_1, t2_1)  # N^5: O^2V^3 / N^4: O^2V^2
        )  # fmt: skip
        ret.reference_state = self.reference_state
        return ret

    @cached_member_function()
    def third_order_dm_correction(self, apply_cvs: bool = False) -> OneParticleDensity:
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "First-order difference density correction not implemented for CVS-RO-DODS-MP"
            )
        assert not apply_cvs

        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        t1_2 = self.ts2(b.ov)
        t2_2 = self.td2(b.oovv)
        t3_2 = self.tt2(b.ooovvv)
        t1_3 = self.ts3(b.ov)

        ret = OneParticleDensity(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        ret.oo = 2 * (
            # Apply (1 + P_ij) to:
            - 1 * einsum("ia,ja->ij", t1_1, t1_2)  # N^3: O^2V^1 / N^2: O^1V^1
            - 0.5 * einsum("ikab,jkab->ij", t2_1, t2_2)  # N^5: O^3V^2 / N^4: O^2V^2
        ).symmetrise()  # fmt: skip
        ret.ov = (
            - 1 * einsum("jb,jb->", t1_1, t1_1) * t1_1  # N^2: O^1V^1 / N^2: O^1V^1
            - 1 * einsum("jb,ijab->ia", t1_1, t2_2)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("ijab,jb->ia", t2_1, t1_2)  # N^4: O^2V^2 / N^4: O^2V^2
            - 0.25 * einsum("jkbc,jkbc->", t2_1, t2_1) * t1_1  # N^4: O^2V^2 / N^4: O^2V^2
            - 0.25 * einsum("jkbc,ijkabc->ia", t2_1, t3_2)  # N^6: O^3V^3 / N^6: O^3V^3
            + 1 * t1_3  # N^2: O^1V^1 / N^2: O^1V^1
        )  # fmt: skip
        ret.vv = 2 * (
            # Apply (1 + P_ab) to:
            + 1 * einsum("ia,ib->ab", t1_1, t1_2)  # N^3: O^1V^2 / N^2: V^2
            + 0.5 * einsum("ijac,ijbc->ab", t2_1, t2_2)  # N^5: O^2V^3 / N^4: O^2V^2
        ).symmetrise()  # fmt: skip
        ret.reference_state = self.reference_state
        return ret

    @cached_member_function()
    def first_order_dm_correction_2p(self, apply_cvs: bool = False) -> TwoParticleDensity:
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "First-order 2-particle difference density correction not implemented "
                "for CVS-RO-DODS-MP"
            )
        assert not apply_cvs

        # Kronecker delta
        d_oo = self.reference_state.foo.zeros_like()
        d_oo.set_mask("ii", 1.0)

        ret = TwoParticleDensity(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        ret.ooov = 2 * (
            # Apply (1 - P_ij) to:
            + 1 * einsum("ja,ik->ijka", self.ts1(b.ov), d_oo)  # N^4: O^3V^1 / N^4: O^3V^1
        ).antisymmetrise(0, 1)  # fmt: skip
        ret.oovv = -1 * self.td1(b.oovv)
        ret.reference_state = self.reference_state
        return ret

    @cached_member_function()
    def second_order_dm_correction_2p(self, apply_cvs: bool = False) -> TwoParticleDensity:
        if self.has_core_occupied_space:
            raise NotImplementedError(
                "Second-order 2-particle difference density correction not implemented "
                "for CVS-RO-DODS-MP"
            )
        assert not apply_cvs

        t1_1 = self.ts1(b.ov)
        t2_1 = self.td1(b.oovv)
        t1_2 = self.ts2(b.ov)

        # Kronecker delta
        d_oo = self.reference_state.foo.zeros_like()
        d_oo.set_mask("ii", 1.0)

        ret = TwoParticleDensity(self.mospaces, symmetry=OperatorSymmetry.HERMITIAN)
        ret.oooo = (
            # Apply 1 to:
            + 0.5 * einsum("ijab,klab->ijkl", t2_1, t2_1)  # N^6: O^4V^2 / N^4: O^2V^2
            # Apply (1 - P_ij - P_kl + P_ijP_kl) to:
            + 4 * (
                # N^4: O^4 / N^4: O^4
                + 1 * einsum("il,jk->ijkl", einsum("ia,la->il", t1_1, t1_1), d_oo)
                # N^5: O^3V^2 / N^4: O^2V^2
                + 0.5 * einsum("il,jk->ijkl", einsum("imab,lmab->il", t2_1, t2_1), d_oo)
            )
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)  # fmt: skip
        ret.ooov = (
            # Apply (1 - P_ij) to:
            2 * (
                + 1 * einsum("ja,ik->ijka", t1_2, d_oo)  # N^4: O^3V^1 / N^4: O^3V^1
                # N^4: O^2V^2 / N^4: O^2V^2
                + 1 * einsum("ia,jk->ijka", einsum("lb,ilab->ia", t1_1, t2_1), d_oo)
            )
            # Apply 1 to:
            - 1 * einsum("kb,ijab->ijka", t1_1, t2_1)  # N^5: O^3V^2 / N^4: O^2V^2
        ).antisymmetrise(0, 1)  # fmt: skip
        ret.oovv = -1 * self.td2(b.oovv)
        ret.ovov = (
            # Apply 1 to:
            - 1 * einsum("ja,ib->iajb", t1_1, t1_1)  # N^4: O^2V^2 / N^4: O^2V^2
            - 1 * einsum("jkac,ikbc->iajb", t2_1, t2_1)  # N^6: O^3V^3 / N^4: O^2V^2
            # N^4: O^2V^2 / N^4: O^2V^2
            + 1 * einsum("ab,ij->iajb", einsum("ka,kb->ab", t1_1, t1_1), d_oo)
            # N^5: O^2V^3 / N^4: O^2V^2
            + 0.5 * einsum("ab,ij->iajb", einsum("klac,klbc->ab", t2_1, t2_1), d_oo)
        )  # fmt: skip
        ret.ovvv = (
            # Apply 1 to:
            - 1 * einsum("ja,ijbc->iabc", t1_1, t2_1)  # N^5: O^2V^3 / N^4: O^1V^3
        ).antisymmetrise(2, 3)  # fmt: skip
        ret.vvvv = (
            # Apply 1 to:
            + 0.5 * einsum("ijab,ijcd->abcd", t2_1, t2_1)  # N^6: O^2V^4 / N^4: V^4
        ).antisymmetrise(0, 1).antisymmetrise(2, 3)  # fmt: skip
        ret.reference_state = self.reference_state
        return ret

    def to_qcvars(self, properties: bool = False, recurse: bool = False, maxlevel: int = 2) -> dict:
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        return self._to_qcvars(
            gs_type="RO-DODS-MP", properties=properties, recurse=recurse, maxlevel=maxlevel
        )
