#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
from itertools import product
from collections import namedtuple


def range_in(inner, full):
    if inner.start is None:
        inner = slice(0, inner.stop, 1)
    if full.start is None:
        full = slice(0, full.stop, 1)
    return all(r in range(full.start, full.stop)
               for r in range(inner.start, inner.stop))


# Helper namedtuple for slices of spin blocks
SpinBlockSlice = namedtuple('SpinBlockSlice',
                            ['block', 'spin', 'fromslice', 'toslice'])


class EriBuilder:
    """
    Parent class for building ERIs with different backends

    Implementation of the following functions in a derived class
    is necessary:
        - ``compute_mo_eri``: compute a block of integrals (Chemists' notation)
          Gets passed the block as a string like 'OOVV' and the spin block as
          as string like 'abab'.
    """
    def __init__(self, n_orbs, n_orbs_alpha, n_alpha, n_beta, restricted):
        self.n_orbs = n_orbs
        self.n_orbs_alpha = n_orbs_alpha
        self.n_alpha = n_alpha
        self.n_beta = n_beta
        self.eri_cache = {}
        self.restricted = restricted
        self.block2slice = {
            "oa": slice(0, self.n_alpha, 1),
            "va": slice(self.n_alpha, self.n_orbs_alpha, 1),
            "ob": slice(self.n_orbs_alpha, self.n_orbs_alpha + self.n_beta, 1),
            "vb": slice(self.n_orbs_alpha + self.n_beta, self.n_orbs, 1),
        }

    def compute_mo_eri(self, blocks, spins):
        """
        Compute block of the ERI tensor in chemists' indexing
        """
        raise NotImplementedError("Implement compute_mo_eri")

    def split_4d_slice(self, slices):
        """
        Split tuple of four slices into the block spin slices
        and their mapping to where elements are to be placed
        """
        return [SpinBlockSlice(tpl[0][0] + tpl[1][0] + tpl[2][0] + tpl[3][0],
                               tpl[0][1] + tpl[1][1] + tpl[2][1] + tpl[3][1],
                               (tpl[0][2], tpl[1][2], tpl[2][2], tpl[3][2]),
                               (tpl[0][3], tpl[1][3], tpl[2][3], tpl[3][3]))
                for tpl in product(*(self.split_1d_slice(sl) for sl in slices))]

    def split_1d_slice(self, sl):
        """
        Split slice into block-slices or multiple block-slices
        """
        if sl.start is None:
            sl = slice(0, sl.stop, 1)
        if sl.step is None:
            sl = slice(sl.start, sl.stop, 1)

        ret = []
        for (block, bslice) in self.block2slice.items():
            fromslice = toslice = None
            if range_in(sl, bslice):
                fromslice = (sl.start - bslice.start, sl.stop - bslice.start)
                toslice = (0, sl.stop - sl.start)
            elif range_in(bslice, sl):
                fromslice = (0, bslice.stop - bslice.start)
                toslice = (bslice.start - sl.start, bslice.stop - sl.start)
            elif sl.start in range(bslice.start, bslice.stop, 1):
                # Because the previous if failed, it cannot be the full range
                fromslice = (sl.start - bslice.start, bslice.stop - bslice.start)
                toslice = (0, bslice.stop - sl.start)
            elif (sl.stop - 1) in range(bslice.start, bslice.stop, 1):
                # Because the previous ifs failed, it cannot be the full range
                fromslice = (0, sl.stop - bslice.start)
                toslice = (bslice.start - sl.start, sl.stop - sl.start)
            if fromslice is None or toslice is None:
                continue   # Not found
            ret.append(SpinBlockSlice(block[0].upper(), block[1],
                                      slice(*fromslice), slice(*toslice)))
        assert len(ret) > 0
        return ret

    def fill_slice_symm(self, slices, out):
        for sbslices in self.split_4d_slice(slices):
            blocks, spins, fromslices, toslices = sbslices
            if spins not in ["aaaa", "aabb", "bbaa", "bbbb"]:
                out[toslices] = 0  # Zero by symmetry
                continue
            if self.restricted:
                # For restricted spins in chem eri do not matter
                spins = "aaaa"

            cache_key = "".join(blocks) + "".join(spins)
            if cache_key in self.eri_cache:
                eri = self.eri_cache[cache_key]
            else:
                eri = self.compute_mo_eri(blocks, spins)
                self.eri_cache[cache_key] = eri

            out[toslices] = eri[fromslices]

    def flush_cache(self):
        self.eri_cache = {}
