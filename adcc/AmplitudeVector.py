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

from numpy.lib.function_base import blackman
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
                #print(self.blocks_ph, other.blocks_ph)
                #print(other.pphh)
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
            #print("__add__")
            allblocks = sorted(set(self.blocks_ph).union(other.blocks_ph))
            #print(allblocks)
            #for k in allblocks:
                #print(self.get(k, 0), other.get(k, 0))
            ret = {k: self.get(k, 0) + other.get(k, 0) for k in allblocks}
            #print(ret.items())
            ret = {k: v for k, v in ret.items() if v != 0}
        else:
            #print("__add__ else")
            ret = {k: tensor + other for k, tensor in self.items()}
        return AmplitudeVector(**ret)

    def __radd__(self, other):
        if isinstance(other, AmplitudeVector):
            return other.__add__(self)
        else:
            #print("__radd__ else")
            ret = {k: other + tensor for k, tensor in self.items()}
            return AmplitudeVector(**ret)


class QED_AmplitudeVector: # it seems all operations, without further specification of pphh part, are unused and can be omitted

    #def __init__(self, gs=None, elec=None, gs1=None, phot=None):
    def __init__(self, gs=None, ph=None, pphh=None, gs1=None, ph1=None, pphh1=None, gs2=None, ph2=None, pphh2=None): # also write this class with *args, **kwargs
        # maybe do this via a list e.g. ph=[ph, ph1, ph2]

        self.gs = gs_vec(gs)
        self.gs1 = gs_vec(gs1)   

        self.gs2 = gs_vec(gs2)     

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

        #self.elec0 = self.gs # so the rest of the class doesn't have to be rewritten
        #self.phot0 = self.gs1 # so the rest of the class doesn't have to be rewritten

    """
        if len(args) != 0:
            if len(args) != 3:
                raise AttributeError
            else:
                self.phot2 = AmplitudeVector(ph=args[1], pphh=args[2])
                self.gs2 = gs_vec(args[0])
                try:
                    self.ph2 = args[1]
                    self.pphh2 = args[2]
                except AttributeError:
                    pass
    """
    
                

        #if self.phot is None:
        #    self.phot = self.elec.zeros_like() #zeros_like(self.elec)
        #    self.phot0 = 0
        #    if self.elec0 is None:
        #        self.elec0 = 0
        #if self.elec is None:
        #    self.elec = self.phot.zeros_like() #zeros_like(self.phot)
        #    self.elec0 = 0
        #    if self.phot0 is None:
        #        self.phot0 = 0

    #def blocks_ph(self):
    #    return sorted(self.keys())



    def dot(self, invec):
        def dot_(self, invec):
            if "pphh" in self.elec.blocks_ph:
                return (self.gs * invec.gs + self.elec.ph.dot(invec.elec.ph) + self.elec.pphh.dot(invec.elec.pphh)
                        + self.gs1 * invec.gs1 + self.phot.ph.dot(invec.phot.ph) + self.phot.pphh.dot(invec.phot.pphh)
                        + self.gs2 * invec.gs2 + self.phot2.ph.dot(invec.phot2.ph) + self.phot2.pphh.dot(invec.phot2.pphh))
            else:
                return (self.gs * invec.gs + self.elec.ph.dot(invec.elec.ph) + self.gs1 * invec.gs1 + self.phot.ph.dot(invec.phot.ph)
                        + self.gs2 * invec.gs2 + self.phot2.ph.dot(invec.phot2.ph) )
        if isinstance(invec, list):
            list_temp = [] # to return a np.array with different dimensions (gs,ph,pphh), we have to fill them into a list and then convert the list ???
            # even though this list should only be appended by floats, the lower approach didnt work out
            #ret = np.array([])
            #(np.append(ret, dot_(self, elem)) for elem in invec)
            for elem in invec:
                list_temp.append(dot_(self, elem))
            return np.array(list_temp)
        else:
           return dot_(self, invec)

    #def __matmul__(self, invec):
    #    return self.dot(invec)

    def __matmul__(self, other):
        if isinstance(other, QED_AmplitudeVector):
            return self.dot(other)
        if isinstance(other, list):
            #print("list given in QED matmul")
            if all(isinstance(elem, QED_AmplitudeVector) for elem in other):
                return self.dot(other)
        return NotImplemented

    #def __add__(self, invec): # special add function, see AmplitudeVector __add__
    #    # this is only used for the summation of gs and gs1 blocks in matvec from AdcMatrix, because the other blocks are covered by AmplitudeVector
    #    if isinstance(invec, QED_AmplitudeVector):
    #        
    #    else:
    #        return NotImplementedError("__add__ in QED_AmplitudeVector only with QED_AmplitudeVector!")

    #def __add__(self, invec):
    #    return QED_AmplitudeVector(self.elec0 + invec.elec0, self.elec.__add__(invec.elec), self.phot0 + invec.phot0, self.phot.__add__(invec.phot))

    #def __radd__(self, invec):
    #    return QED_AmplitudeVector(self.elec0 + invec.elec0, self.elec.__add__(invec.elec), self.phot0 + invec.phot0, self.phot.__add__(invec.phot))

    def __sub__(self, invec):
        if isinstance(invec, QED_AmplitudeVector):
            return QED_AmplitudeVector(self.gs - invec.gs, self.elec.__sub__(invec.elec), self.gs1 - invec.gs1, self.phot.__sub__(invec.phot))
        elif isinstance(invec, (float, int)): # for diagonal - shift in preconditioner.py
            # this results in a scalar in block pphh, if pphh is originally None
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(gs=self.gs - invec, ph=self.elec.ph.__sub__(invec), pphh=self.elec.pphh.__sub__(invec),
                                             gs1=self.gs1 - invec, ph1=self.phot.ph.__sub__(invec), pphh1=self.phot.pphh.__sub__(invec),
                                             gs2=self.gs2 - invec, ph2=self.phot2.ph.__sub__(invec), pphh2=self.phot2.pphh.__sub__(invec))
            else:
                return QED_AmplitudeVector(gs=self.gs - invec, ph=self.elec.ph.__sub__(invec), gs1=self.gs1 - invec, ph1=self.phot.ph.__sub__(invec),
                                            gs2=self.gs2 - invec, ph2=self.phot2.ph.__sub__(invec))

    def __mul__(self, scalar):
        return QED_AmplitudeVector(scalar * self.gs, self.elec.__mul__(scalar), scalar * self.gs1, self.phot.__mul__(scalar))

    def __rmul__(self, scalar):
        return QED_AmplitudeVector(scalar * self.gs, self.elec.__rmul__(scalar), scalar * self.gs1, self.phot.__rmul__(scalar))

    def __truediv__(self, other):
        #print(type(self.elec), self.elec.ph)
        #print(type(other.elec), other.elec.ph)
        #if "pphh" in self.elec.blocks_ph:
        #    return QED_AmplitudeVector(gs=self.elec0 / other.elec0, ph=self.elec.__truediv__(other.elec),
        #                            gs1=self.phot0 / other.phot0, ph1=self.phot.__truediv__(other.phot))
        #else:
        if isinstance(other, QED_AmplitudeVector):
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(gs=self.gs / other.gs, ph=self.elec.ph.__truediv__(other.elec.ph), pphh=self.elec.pphh.__truediv__(other.elec.pphh),
                                            gs1=self.gs1 / other.gs1, ph1=self.phot.ph.__truediv__(other.phot.ph), pphh1=self.phot.pphh.__truediv__(other.phot.pphh),
                                            gs2=self.gs2 / other.gs2, ph2=self.phot2.ph.__truediv__(other.phot2.ph), pphh2=self.phot2.pphh.__truediv__(other.phot2.pphh))
            else:
                return QED_AmplitudeVector(gs=self.gs / other.gs, ph=self.elec.ph.__truediv__(other.elec.ph), 
                                            gs1=self.gs1 / other.gs1, ph1=self.phot.ph.__truediv__(other.phot.ph),
                                            gs2=self.gs2 / other.gs2, ph2=self.phot2.ph.__truediv__(other.phot2.ph))
        elif isinstance(other, (float, int)):
            #return QED_AmplitudeVector(gs=self.elec0 / other, ph=self.elec.ph.__truediv__(other), 
            #                            gs1=self.phot0 / other, ph1=self.phot.ph.__truediv__(other))
            if "pphh" in self.elec.blocks_ph:
                return QED_AmplitudeVector(gs=self.gs / other, ph=self.elec.ph.__truediv__(other), pphh=self.elec.pphh.__truediv__(other),
                                            gs1=self.gs1 / other, ph1=self.phot.ph.__truediv__(other), pphh1=self.phot.pphh.__truediv__(other),
                                            gs2=self.gs2 / other, ph2=self.phot2.ph.__truediv__(other), pphh2=self.phot2.pphh.__truediv__(other))
            else:
                return QED_AmplitudeVector(gs=self.gs / other, ph=self.elec.ph.__truediv__(other), 
                                            gs1=self.gs1 / other, ph1=self.phot.ph.__truediv__(other),
                                            gs2=self.gs2 / other, ph2=self.phot2.ph.__truediv__(other))

    def zeros_like(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector(0, self.elec.zeros_like(), 0, self.phot.zeros_like(), 0, self.phot2.zeros_like())
        else:
            return QED_AmplitudeVector(0, self.elec.zeros_like(), 0, self.phot.zeros_like())

    def empty_like(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector([], self.elec.empty_like(), [], self.phot.empty_like(), [], self.phot2.empty_like())
        else:
            return QED_AmplitudeVector([], self.elec.empty_like(), [], self.phot.empty_like())

    def copy(self):
        if "pphh" in self.elec.blocks_ph:
            return QED_AmplitudeVector(self.gs, self.elec.copy(), self.gs1, self.phot.copy(), self.gs2, self.phot2.copy())
        else:
            return QED_AmplitudeVector(self.gs, self.elec.copy(), self.gs1, self.phot.copy())

    def evaluate(self):
        #print(self.elec0, self.elec, self.phot0, self.phot)
        self.elec.evaluate()
        self.phot.evaluate()
        self.phot2.evaluate()
        try:
            self.gs.evaluate()
            self.gs1.evaluate()
            self.gs2.evaluate()
        except:
            self.gs
            self.gs1
            self.gs2
        #if "pphh" in self.elec.blocks_ph:
        #    self.phot2.evaluate()
        #    try:
        #        self.gs2.evaluate()
        #    except:
        #        self.gs2
        return self





class gs_vec:
    # this class only exists, to forward gs and gs1 to the relevant operators
    def __init__(self, val):
        #print("uses gs_vec init")
        self.val = val
        #print(type(self.val))
        if isinstance(self.val, AmplitudeVector):
            TypeError("gs_vec got invoked with AmplitudeVector type")
        #try:
        #    self.val = float(arg)
        #except:
        #    raise NotImplementedError("gs_vec needs to be given just one number,"
        #                                "which has to be float convertible")
    def __mul__(self, invec):
        if isinstance(invec, (float, int)):
            #print("uses gs_vec __mul__")
            return self.val * invec
        elif isinstance(invec, AmplitudeVector):
            return invec.__mul__(self.val)
        elif isinstance(invec, gs_vec):
            return self.val * invec.val

    def __rmul__(self, scalar):
        #print("uses gs_vec __rmul__")
        return self.val * scalar

    def __add__(self, scalar):
        if self.val == None or scalar == None:
            return self.val
        else:
            #print("uses add gs_vec", self.val, scalar)
            #print(type(self.val))
            return self.val + scalar

    def __radd__(self, scalar):
        return self.__add__(scalar)
        #if self.val == None:
        #    return scalar
        #else:
        #    #print("uses radd gs_vec", self.val, scalar)
        #    return self.val + scalar

    def __sub__(self, scalar):
        if isinstance(scalar, gs_vec):
            return self.val - scalar.val
        elif isinstance(scalar, (float, int)):
            return self.val - scalar 

    def __truediv__(self, scalar):
        if isinstance(scalar, gs_vec):
            return self.val / scalar.val
        elif isinstance(scalar, (float, int)):
            return self.val / scalar

    def as_float(self):
        return self.val

    





"""
class QED_AmplitudeVector(dict):
    # This vector contains the structure (ph, ph1) or (ph, pphh, ph1, pphh1), where the
    # zero excitation space is included at the beginning of each ph (see matrix.py).
    # The upper and the lower half of the vector have photonic intermediate states 0 and 1,
    # respectively, and we abbreviate ph0 and pphh0 by ph and pphh.

    # we first implement the <0|0> block, but with the groundstate terms, gs = grounstate
    def __init__(self, *args, **kwargs):
        """
        #Construct an AmplitudeVector. Typical use cases are
        #``AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles)``.
"""
        if args:
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "AmplitudeVector(ph=tensor_singles, pphh=tensor_doubles) "
                          "instead.")
            if len(args) == 2:
                super().__init__(gs=args[0], ph=args[1])
            elif len(args) == 4:
                super().__init__(gs=args[0], ph=args[1], pphh1=args[2])
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
        if sorted(self.blocks_ph) == ["gs", "ph", "pphh"]:
            return ["g", "s", "d"]
        if sorted(self.blocks_ph) == ["gs", "ph"]:
            return ["g", "s"]
        if sorted(self.blocks_ph) == ["pphh"]:
            return ["d"]
        elif sorted(self.blocks_ph) == ["ph"]:
            return ["s"]
        elif sorted(self.blocks_ph) == ["gs"]:
            return ["g"]
        elif sorted(self.blocks_ph) == []:
            return []
        else:
            raise NotImplementedError(self.blocks_ph)

    @property
    def blocks_ph(self): # despite the name, this function can be applied to all blocks
        """
        #Return the blocks which are used inside the vector.
        #Note: This is a temporary name. The attribute will be removed in 0.16.0.
"""
        return sorted(self.keys())

    def __getitem__(self, index):
        if index in (0, 1, 2, "g", "s", "d"):
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "block labels like 'ph', 'pphh' instead.")
            if index in (0, "g"):
                return self.__getitem__("gs")
            elif index in (1, "s"):
                return self.__getitem__("ph")
            elif index in (2, "d"):
                return self.__getitem__("pphh")
            #elif index in (3, "d1"):
            #    return self.__getitem__("pphh1")
            else:
                raise KeyError(index)
        else:
            return super().__getitem__(index)

    def __setitem__(self, index, item):
        if index in (0, 1, 2, "g", "s", "d"):
            warnings.warn("Using the list interface of AmplitudeVector is "
                          "deprecated and will be removed in version 0.16.0. Use "
                          "block labels like 'ph', 'pphh' instead.")
            if index in (0, "g"):
                return self.__setitem__("gs", item)
            elif index in (1, "s"):
                return self.__setitem__("ph", item)
            elif index in (2, "d"):
                return self.__setitem__("pphh", item)
            #elif index in (3, "d1"):
            #    return self.__setitem__("pphh1", item)
            else:
                raise KeyError(index)
        else:
            return super().__setitem__(index, item)

    def copy(self):
        """#Return a copy of the AmplitudeVector
"""
        return AmplitudeVector(**{k: t.copy() for k, t in self.items()})

    def evaluate(self):
        for t in self.values():
            t.evaluate()
        return self

    def ones_like(self):
        """#Return an empty AmplitudeVector of the same shape and symmetry
"""
        return AmplitudeVector(**{k: t.ones_like() for k, t in self.items()})

    def empty_like(self):
        """#Return an empty AmplitudeVector of the same shape and symmetry
"""
        return AmplitudeVector(**{k: t.empty_like() for k, t in self.items()})

    def nosym_like(self):
        """#Return an empty AmplitudeVector of the same shape and symmetry
"""
        return AmplitudeVector(**{k: t.nosym_like() for k, t in self.items()})

    def zeros_like(self):
        """#Return an AmplitudeVector of the same shape and symmetry with
           #all elements set to zero
"""
        return AmplitudeVector(**{k: t.zeros_like() for k, t in self.items()})

    def set_random(self):
        for t in self.values():
            t.set_random()
        return self

    def dot(self, other):
        """#Return the dot product with another AmplitudeVector
        #or the dot products with a list of AmplitudeVectors.
        #In the latter case a np.ndarray is returned.
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

"""