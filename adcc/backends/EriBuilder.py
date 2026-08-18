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
from dataclasses import dataclass
from itertools import product
from typing import Literal, TypeAlias, TypeGuard
import numpy as np

IntSlice: TypeAlias = "slice[int, int, int]"
IntSlice4D = tuple[IntSlice, IntSlice, IntSlice, IntSlice]
Block = Literal["O", "V"]
Block4D = tuple[Block, Block, Block, Block]
Spin = Literal["a", "b"]
Spin4D = tuple[Spin, Spin, Spin, Spin]
Array4D = np.ndarray[tuple[int, int, int, int], np.dtype[np.float64]]


def is_int_slice(sl: slice) -> TypeGuard[IntSlice]:
    return (
        isinstance(sl.start, int)
        and isinstance(sl.stop, int)
        and isinstance(sl.step, int)
    )


@dataclass(frozen=True, slots=True)
class SpinBlockSlice:
    block: Block
    spin: Spin
    fromslice: IntSlice
    toslice: IntSlice


@dataclass(frozen=True, slots=True)
class SpinBlockSlice4D:
    block: Block4D
    spin: Spin4D
    fromslice: IntSlice4D
    toslice: IntSlice4D


def range_in(inner: IntSlice, full: IntSlice) -> bool:
    return all(r in range(full.start, full.stop)
               for r in range(inner.start, inner.stop))


class EriBuilder:
    """
    Parent class for building ERIs with different backends

    Implementation of the following functions in a derived class
    is necessary:
        - ``compute_mo_eri``: compute a block of integrals (Chemists' notation)
          Gets passed the block as a string like 'OOVV' and the spin block as
          as string like 'abab'.
    """
    def __init__(self, n_orbs: int, n_orbs_alpha: int, n_alpha: int, n_beta: int,
                 restricted: bool):
        self.n_orbs: int = n_orbs
        self.n_orbs_alpha: int = n_orbs_alpha
        self.n_alpha: int = n_alpha
        self.n_beta: int = n_beta
        self.eri_cache: dict[str, Array4D] = {}
        self.restricted: bool = restricted
        self.block2slice: dict[tuple[Block, Spin], IntSlice] = {
            ("O", "a"): slice(0, self.n_alpha, 1),
            ("V", "a"): slice(self.n_alpha, self.n_orbs_alpha, 1),
            ("O", "b"): slice(self.n_orbs_alpha,
                              self.n_orbs_alpha + self.n_beta, 1),
            ("V", "b"): slice(self.n_orbs_alpha + self.n_beta, self.n_orbs, 1),
        }

    def compute_mo_eri(self, blocks: Block4D, spins: Spin4D) -> Array4D:
        """
        Compute block of the ERI tensor in chemists' indexing
        """
        raise NotImplementedError("Implement compute_mo_eri")

    def split_4d_slice(self, slices: IntSlice4D) -> list[SpinBlockSlice4D]:
        """
        Split tuple of four slices into the block spin slices
        and their mapping to where elements are to be placed
        """
        splitted = (self.split_1d_slice(sl) for sl in slices)
        return [SpinBlockSlice4D(
            (sl1.block, sl2.block, sl3.block, sl4.block),
            (sl1.spin, sl2.spin, sl3.spin, sl4.spin),
            (sl1.fromslice, sl2.fromslice, sl3.fromslice, sl4.fromslice),
            (sl1.toslice, sl2.toslice, sl3.toslice, sl4.toslice)
        ) for sl1, sl2, sl3, sl4 in product(*splitted)]

    def split_1d_slice(
        self, sl: "slice[int | None, int, int | None]"
    ) -> list[SpinBlockSlice]:
        """
        Split slice into block-slices or multiple block-slices
        """
        if sl.start is None:
            sl = slice(0, sl.stop, 1)
        if sl.step is None:
            sl = slice(sl.start, sl.stop, 1)
        assert is_int_slice(sl)

        ret: list[SpinBlockSlice] = []
        for (block, bslice) in self.block2slice.items():
            fromslice: tuple[int, int] | None = None
            toslice: tuple[int, int] | None = None
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
            ret.append(SpinBlockSlice(
                block[0], block[1], slice(*fromslice, 1), slice(*toslice, 1)
            ))
        assert len(ret) > 0
        return ret

    def fill_slice_symm(self, slices: IntSlice4D, out: Array4D) -> None:
        non_zero_spin_blocks: list[Spin4D] = [  # chemist notation
            ("a", "a", "a", "a"),
            ("a", "a", "b", "b"),
            ("b", "b", "a", "a"),
            ("b", "b", "b", "b"),
        ]
        for sbslices in self.split_4d_slice(slices):
            blocks: Block4D = sbslices.block
            spins: Spin4D = sbslices.spin
            fromslices: IntSlice4D = sbslices.fromslice
            toslices: IntSlice4D = sbslices.toslice
            if spins not in non_zero_spin_blocks:
                out[toslices] = 0  # Zero by symmetry
                continue
            if self.restricted:
                # For restricted spins in chem eri do not matter
                spins = ("a", "a", "a", "a")

            cache_key = "".join(blocks) + "".join(spins)
            if cache_key in self.eri_cache:
                eri = self.eri_cache[cache_key]
            else:
                eri = self.compute_mo_eri(blocks, spins)
                self.eri_cache[cache_key] = eri

            out[toslices] = eri[fromslices]

    def flush_cache(self) -> None:
        self.eri_cache = {}
