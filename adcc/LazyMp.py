#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import libadcc
import numpy as np

from .mp import compute_mp2_diffdm  # noqa: F401
from .functions import direct_sum, evaluate, einsum
from .misc import cached_property
from .ReferenceState import ReferenceState
from .OneParticleOperator import OneParticleOperator, product_trace
from .timings import Timer
from . import block as b


def compute_t2(eri, df1, df2):
    return evaluate(
        eri / direct_sum("ia+jb->ijab", df1, df2).symmetrise((2, 3))
    )


def compute_energy_correction(eri, t2):
    return -0.25 * einsum('ijab,ijab->', eri, t2)


class LazyMp:
    def __init__(self, hf):
        """
        Initialise the class dealing with the M/oller-Plesset ground state.
        """
        if isinstance(hf, libadcc.HartreeFockSolution_i):
            hf = ReferenceState(hf)
        if not isinstance(hf, ReferenceState):
            raise TypeError("hf needs to be a ReferenceState "
                            "or a HartreeFockSolution_i")
        self.reference_state = hf
        self.mospaces = hf.mospaces
        self.timer = Timer()
        self.has_core_occupied_space = hf.has_core_occupied_space
        # FIXME: just a hack to let tests pass
        self.t2_set = {}
        # TODO: implement cache

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        else:
            raise AttributeError

    def df(self, space):
        """Delta Fock matrix"""
        hf = self.reference_state
        s1 = space[:2]
        s2 = space[2:]
        fC = hf.fock(s1 + s1).diagonal()
        fv = hf.fock(s2 + s2).diagonal()
        return direct_sum("-i+a->ia", fC, fv).evaluate()

    def t2(self, space):
        """T2 amplitudes"""
        # FIXME: just a hack to let tests pass
        if len(self.t2_set):
            return self.t2_set[space]
        hf = self.reference_state
        if space == b.oovv:
            eia = self.df(b.ov)
            return compute_t2(hf.oovv, eia, eia)
        elif space == b.ocvv:
            eia = self.df(b.ov)
            eca = self.df(b.cv)
            return compute_t2(hf.ocvv, eia, eca)
        elif space == b.ccvv:
            eca = self.df(b.cv)
            return compute_t2(hf.ccvv, eca, eca)
        else:
            raise NotImplementedError("T2 amplitudes not "
                                      f"implemented for space {space}.")

    def set_t2(self, space, tensor):
        """
        Set the T2 amplitudes tensor. This invalidates all data depending on the T2
        amplitudes in this class. Note, that potential other caches, such as
        computed ADC intermediates are *not* automatically invalidated.
        """
        # FIXME: just a temporary hack
        self.t2_set[space] = tensor

    def td2(self, space):
        """Return the T^D_2 term"""
        if space != b.oovv:
            raise NotImplementedError()
        t2erit = self.t2eri(b.oovv, b.ov).transpose((1, 0, 2, 3))
        denom = direct_sum(
            'ia,jb->ijab', self.df(b.ov), self.df(b.ov)
        ).symmetrise(0, 1)
        return evaluate(
            (+ 4.0 * t2erit.antisymmetrise(2, 3).antisymmetrise(0, 1)
             - 0.5 * self.t2eri(b.oovv, b.vv)
             - 0.5 * self.t2eri(b.oovv,  b.oo)) / denom
        )

    def t2eri(self, space, contraction):
        """
        Return the T2 tensor with ERI tensor contraction intermediates.
        These are called pi1 to pi7 in libadc.
        """
        hf = self.reference_state
        expressions = {
            # space + contraction
            b.ooov + b.vv: einsum('ijbc,kabc->ijka', self.t2oo, hf.ovvv),
            b.ooov + b.ov: einsum('ilab,lkjb->ijka', self.t2oo, hf.ooov),
            b.oovv + b.oo: einsum('klab,ijkl->ijab', self.t2oo, hf.oooo),
            b.oovv + b.ov: einsum('jkac,kbic->ijab', self.t2oo, hf.ovov),
            b.oovv + b.vv: einsum('ijcd,abcd->ijab', self.t2oo, hf.vvvv),
            b.ovvv + b.oo: einsum('jkbc,jkia->iabc', self.t2oo, hf.ooov),
            b.ovvv + b.ov: einsum('ijbd,jcad->iabc', self.t2oo, hf.ovvv),
        }
        return expressions[space + contraction].evaluate()

    @property
    def mp2_diffdm(self):
        """
        Return the MP2 differensce density in the MO basis.
        """
        hf = self.reference_state
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        blocks = ["oo", "ov", "vv"]
        if self.has_core_occupied_space:
            blocks += ["cc", "co", "cv"]
        for bl in blocks:
            ba = getattr(b, bl)
            ret[ba] = compute_mp2_diffdm(hf, self, bl, apply_cvs=False)
        ret.reference_state = self.reference_state
        return ret

    def density(self, level=2):
        """
        Return the MP density in the MO basis with all corrections
        up to the specified order of perturbation theory
        """
        if level == 1:
            return self.reference_state.density
        elif level == 2:
            return self.reference_state.density + self.mp2_diffdm
        else:
            raise NotImplementedError("Only densities for level 1 and 2"
                                      " are implemented.")

    def dipole_moment(self, level=2):
        """
        Return the MP dipole moment at the specified level of
        perturbation theory.
        """
        if level == 1:
            return self.reference_state.dipole_moment
        elif level == 2:
            return self.mp2_dipole_moment
        else:
            raise NotImplementedError("Only dipole moments for level 1 and 2"
                                      " are implemented.")

    def energy_correction(self, level=2):
        """Obtain the MP energy correction at a particular level"""
        if level > 3:
            raise NotImplementedError(f"MP({level}) energy correction "
                                      "not implemented.")
        if level < 2:
            return 0.0
        hf = self.reference_state
        if level == 2 and not self.has_core_occupied_space:
            return compute_energy_correction(hf.oovv, self.t2oo)
        elif level == 2 and self.has_core_occupied_space:
            pairs = [(self.t2oo, hf.oovv),
                     (2.0 * self.t2oc, hf.ocvv),
                     (self.t2cc, hf.ccvv)]
            return sum(compute_energy_correction(*p) for p in pairs)
        elif level == 3:
            return compute_energy_correction(hf.oovv, self.td2(b.oovv))

    def energy(self, level=2):
        """
        Obtain the total energy (SCF energy plus all corrections)
        at a particular level of perturbation theory.
        """
        if level == 0:
            # Sum of orbital energies ...
            raise NotImplementedError("Total MP(0) energy not implemented.")

        # Accumulator for all energy terms
        energies = [self.reference_state.energy_scf]
        for il in range(2, level + 1):
            energies.append(self.energy_correction(il))
        return sum(energies)

    def to_qcvars(self, properties=False, recurse=False, maxlevel=2):
        """
        Return a dictionary with property keys compatible to a Psi4 wavefunction
        or a QCEngine Atomicresults object.
        """
        qcvars = {}
        for level in range(2, maxlevel + 1):
            try:
                mpcorr = self.energy_correction(level)
                qcvars[f"MP{level} CORRELATION ENERGY"] = mpcorr
                qcvars[f"MP{level} TOTAL ENERGY"] = self.energy(level)
            except NotImplementedError:
                pass
            except ValueError:
                pass

        if properties:
            for level in range(2, maxlevel + 1):
                try:
                    qcvars["MP2 DIPOLE"] = self.dipole_moment(level)
                except NotImplementedError:
                    pass

        if recurse:
            qcvars.update(self.reference_state.to_qcvars(properties, recurse))
        return qcvars

    @property
    def mp2_density(self):
        return self.density(2)

    @cached_property
    def mp2_dipole_moment(self):
        refstate = self.reference_state
        dipole_integrals = refstate.operators.electric_dipole
        mp2corr = -np.array([product_trace(comp, self.mp2_diffdm)
                             for comp in dipole_integrals])
        return refstate.dipole_moment + mp2corr
