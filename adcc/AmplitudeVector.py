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
import warnings

import numpy as np


class AmplitudeVector(dict):
    def __init__(self, *args, **kwargs):
        """
        Construct an AmplitudeVector. Typical use cases are
        ``AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles)``.
        """
        if args:
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles) "
                          "instead.")
            if len(args) == 1:
                super().__init__(ph=args[0])
            elif len(args) == 2:
                super().__init__(ph=args[0], pphh=args[1])
        else:
            super().__init__(**kwargs)

    def __getattr__(self, key):
        if self.__contains__(key):
            return self.__getitem__(key)
        raise AttributeError

    def __setattr__(self, key, item):
        if self.__contains__(key):
            return self.__setitem__(key, item)
        raise AttributeError

    @property
    def blocks(self):
        warnings.warn("The blocks attribute will change behaviour in 0.16.0.")
        if sorted(self.blocks_ph) == ["ph", "pphh"]:
            return ["s", "d"]
        if sorted(self.blocks_ph) == ["pphh"]:
            return ["d"]
        elif sorted(self.blocks_ph) == ["ph"]:
            return ["s"]
        elif sorted(self.blocks_ph) == []:
            return []
        else:
            raise NotImplementedError(self.blocks_ph)

    @property
    def blocks_ph(self):
        """
        Return the blocks which are used inside the vector.
        Note: This is a temporary name. The attribute will be removed in 0.16.0.
        """
        return sorted(self.keys())

    def __getitem__(self, index):
        if index in (0, 1, "s", "d"):
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "block labels like 'ph', 'pphh' instead.")
            if index in (0, "s"):
                return self.__getitem__("ph")
            elif index in (1, "d"):
                return self.__getitem__("pphh")
            else:
                raise KeyError(index)
        else:
            return super().__getitem__(index)

    def __setitem__(self, index, item):
        if index in (0, 1, "s", "d"):
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "block labels like 'ph', 'pphh' instead.")
            if index in (0, "s"):
                return self.__setitem__("ph", item)
            elif index in (1, "d"):
                return self.__setitem__("pphh", item)
            else:
                raise KeyError(index)
        else:
            return super().__setitem__(index, item)

    def copy(self):
        """Return a copy of the AmplitudeVector"""
        return AmplitudeVector(**{k: t.copy() for k, t in self.items()})

    def evaluate(self):
        for t in self.values():
            t.evaluate()
        return self

    @property
    def needs_evaluation(self):
        return any(t.needs_evaluation for k, t in self.items())

    def ones_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.ones_like() for k, t in self.items()})

    def empty_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.empty_like() for k, t in self.items()})

    def nosym_like(self):
        """Return an empty AmplitudeVector of the same shape and symmetry"""
        return AmplitudeVector(**{k: t.nosym_like() for k, t in self.items()})

    def zeros_like(self):
        """Return an AmplitudeVector of the same shape and symmetry with
           all elements set to zero"""
        return AmplitudeVector(**{k: t.zeros_like() for k, t in self.items()})

    def set_random(self):
        for t in self.values():
            t.set_random()
        return self

    def dot(self, other):
        """Return the dot product with another AmplitudeVector
        or the dot products with a list of AmplitudeVectors.
        In the latter case a np.ndarray is returned.
        """
        if isinstance(other, list):
            # Make a list where the first index is all singles parts,
            # the second is all doubles parts and so on
            return sum(self[b].dot([av[b] for av in other]) for b in self.keys())
        else:
            return sum(self[b].dot(other[b]) for b in self.keys())

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.dot(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return self.dot(other)
        return NotImplemented

    def __forward_to_blocks(self, fname, other):
        if isinstance(other, AmplitudeVector):
            if sorted(other.blocks_ph) != sorted(self.blocks_ph):
                raise ValueError("Blocks of both AmplitudeVector objects "
                                 f"need to agree to perform {fname}")
            ret = {k: getattr(tensor, fname)(other[k])
                   for k, tensor in self.items()}
        else:
            ret = {k: getattr(tensor, fname)(other) for k, tensor in self.items()}
        if any(r == NotImplemented for r in ret.values()):
            return NotImplemented
        else:
            return AmplitudeVector(**ret)

    def __mul__(self, other):
        return self.__forward_to_blocks("__mul__", other)

    def __rmul__(self, other):
        return self.__forward_to_blocks("__rmul__", other)

    def __sub__(self, other):
        return self.__forward_to_blocks("__sub__", other)

    def __rsub__(self, other):
        return self.__forward_to_blocks("__rsub__", other)

    def __truediv__(self, other):
        return self.__forward_to_blocks("__truediv__", other)

    def __imul__(self, other):
        return self.__forward_to_blocks("__imul__", other)

    def __iadd__(self, other):
        return self.__forward_to_blocks("__iadd__", other)

    def __isub__(self, other):
        return self.__forward_to_blocks("__isub__", other)

    def __itruediv__(self, other):
        return self.__forward_to_blocks("__itruediv__", other)

    def __repr__(self):
        return "AmplitudeVector(" + "=..., ".join(self.blocks_ph) + "=...)"

    # __add__ is special because we want to be able to add AmplitudeVectors
    # with missing blocks
    def __add__(self, other):
        if isinstance(other, AmplitudeVector):
            allblocks = sorted(set(self.blocks_ph).union(other.blocks_ph))
            ret = {k: self.get(k, 0) + other.get(k, 0) for k in allblocks}
            ret = {k: v for k, v in ret.items() if v != 0}
        else:
            ret = {k: tensor + other for k, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __radd__(self, other):
        if isinstance(other, AmplitudeVector):
            return other.__add__(self)
        else:
            ret = {k: other + tensor for k, tensor in self.items()}
            return AmplitudeVector(**ret)


class QED_AmplitudeVector:

    # TODO: initialize this class with **kwargs, and think of a functionality, which then eases up
    # e.g. the matvec function in the AdcMatrix.py for arbitrarily large QED vectors. However, this is
    # not necessarily required, since e.g. QED-ADC(3) is very complicated to derive and QED-ADC(1) (with
    # just the single dispersion mode) is purely for academic purposes and hence not required to provide
    # optimum performance
    def __init__(self, ph=None, pphh=None, gs1=None, ph1=None, pphh1=None, gs2=None, ph2=None, pphh2=None):

        self.gs1 = gs1
        self.gs2 = gs2

        if pphh != None:
            self.elec = AmplitudeVector(ph=ph, pphh=pphh)
            self.phot = AmplitudeVector(ph=ph1, pphh=pphh1)
            self.phot2 = AmplitudeVector(ph=ph2, pphh=pphh2)
        else:
            self.elec = AmplitudeVector(ph=ph)
            self.phot = AmplitudeVector(ph=ph1)
            self.phot2 = AmplitudeVector(ph=ph2)
        try:
            self.ph = ph
            self.ph1 = ph1
            self.ph2 = ph2
        except AttributeError:
            pass
        try:
            self.pphh = pphh
            self.pphh1 = pphh1
            self.pphh2 = pphh2
        except AttributeError:
            # there are no doubles terms --> ADC(0) or ADC(1)
            pass
    
    def dot(self, invec):
        def dot_(self, invec):
            if "pphh" in self.elec.blocks_ph:
                return (self.elec.ph.dot(invec.elec.ph) + self.elec.pphh.dot(invec.elec.pphh)
                        + self.gs1 * invec.gs1 + self.phot.ph.dot(invec.phot.ph) + self.phot.pphh.dot(invec.phot.pphh)
                        + self.gs2 * invec.gs2 + self.phot2.ph.dot(invec.phot2.ph) + self.phot2.pphh.dot(invec.phot2.pphh))
            else:
                return (self.elec.ph.dot(invec.elec.ph) + self.gs1 * invec.gs1 + self.phot.ph.dot(invec.phot.ph)
                        + self.gs2 * invec.gs2 + self.phot2.ph.dot(invec.phot2.ph) )
        if isinstance(invec, list):
            return np.array([dot_(self, elem) for elem in invec])
        else:
           return dot_(self, invec)

    def __matmul__(self, other):
        if isinstance(other, QED_AmplitudeVector):
            return self.dot(other)
        if isinstance(other, list):
            #print("list given in QED matmul")
            if all(isinstance(elem, QED_AmplitudeVector) for elem in other):
                return self.dot(other)
        return NotImplemented
    

    def __sub__(self, invec):
        if isinstance(invec, (float, int)): # for diagonal - shift in preconditioner.py
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(ph=self.elec.ph.__sub__(invec), pphh=self.elec.pphh.__sub__(invec),
                                             gs1=self.gs1 - invec, ph1=self.phot.ph.__sub__(invec), pphh1=self.phot.pphh.__sub__(invec),
                                             gs2=self.gs2 - invec, ph2=self.phot2.ph.__sub__(invec), pphh2=self.phot2.pphh.__sub__(invec))
            else:
                return QED_AmplitudeVector(ph=self.elec.ph.__sub__(invec), gs1=self.gs1 - invec, ph1=self.phot.ph.__sub__(invec),
                                            gs2=self.gs2 - invec, ph2=self.phot2.ph.__sub__(invec))


    def __truediv__(self, other):
        if isinstance(other, QED_AmplitudeVector):
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(ph=self.elec.ph.__truediv__(other.elec.ph), pphh=self.elec.pphh.__truediv__(other.elec.pphh),
                                            gs1=self.gs1 / other.gs1, ph1=self.phot.ph.__truediv__(other.phot.ph), pphh1=self.phot.pphh.__truediv__(other.phot.pphh),
                                            gs2=self.gs2 / other.gs2, ph2=self.phot2.ph.__truediv__(other.phot2.ph), pphh2=self.phot2.pphh.__truediv__(other.phot2.pphh))
            else:
                return QED_AmplitudeVector(ph=self.elec.ph.__truediv__(other.elec.ph), 
                                            gs1=self.gs1 / other.gs1, ph1=self.phot.ph.__truediv__(other.phot.ph),
                                            gs2=self.gs2 / other.gs2, ph2=self.phot2.ph.__truediv__(other.phot2.ph))
        elif isinstance(other, (float, int)):
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(ph=self.elec.ph.__truediv__(other), pphh=self.elec.pphh.__truediv__(other),
                                            gs1=self.gs1 / other, ph1=self.phot.ph.__truediv__(other), pphh1=self.phot.pphh.__truediv__(other),
                                            gs2=self.gs2 / other, ph2=self.phot2.ph.__truediv__(other), pphh2=self.phot2.pphh.__truediv__(other))
            else:
                return QED_AmplitudeVector(ph=self.elec.ph.__truediv__(other), 
                                            gs1=self.gs1 / other, ph1=self.phot.ph.__truediv__(other),
                                            gs2=self.gs2 / other, ph2=self.phot2.ph.__truediv__(other))

    def zeros_like(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector(ph=self.elec.zeros_like().ph, pphh=self.elec.zeros_like().pphh, gs1=0,
                                        ph1=self.phot.zeros_like().ph, pphh1=self.phot.zeros_like().pphh, gs2=0, 
                                        ph2=self.phot2.zeros_like().ph, pphh2=self.phot2.zeros_like().pphh)
        else:
            return QED_AmplitudeVector(ph=self.elec.zeros_like().ph, pphh=None, gs1=0,
                                        ph1=self.phot.zeros_like().ph, pphh1=None, gs2=0, 
                                        ph2=self.phot2.zeros_like().ph, pphh2=None)

    def empty_like(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector(ph=self.elec.empty_like().ph, pphh=self.elec.empty_like().pphh, gs1=0,
                                        ph1=self.phot.empty_like().ph, pphh1=self.phot.empty_like().pphh, gs2=0,
                                        ph2=self.phot2.empty_like().ph, pphh2=self.phot2.empty_like().pphh)
        else:
            QED_AmplitudeVector(ph=self.elec.empty_like().ph, pphh=None, gs1=0,
                                        ph1=self.phot.empty_like().ph, pphh1=None, gs2=0,
                                        ph2=self.phot2.empty_like().ph, pphh2=None)

    def copy(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector(ph=self.elec.copy().ph, pphh=self.elec.copy().pphh, gs1=self.gs1,
                                        ph1=self.phot.copy().ph, pphh1=self.phot.copy().pphh, gs2=self.gs2,
                                        ph2=self.phot2.copy().ph, pphh2=self.phot2.copy().pphh)
        else:
            QED_AmplitudeVector(ph=self.elec.copy().ph, pphh=None, gs1=self.gs1,
                                        ph1=self.phot.copy().ph, pphh1=None, gs2=self.gs2,
                                        ph2=self.phot2.copy().ph, pphh2=None)

    def evaluate(self):
        self.elec.evaluate()
        self.phot.evaluate()
        self.phot2.evaluate()
        self.gs1
        self.gs2
        return self