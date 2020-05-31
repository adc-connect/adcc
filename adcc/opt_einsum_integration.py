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


__all__ = ["register_with_opt_einsum"]


def register_with_opt_einsum():
    import libadcc

    from opt_einsum.backends.dispatch import EVAL_CONSTS_BACKENDS

    def libadcc_evaluate_constants(const_arrays, expr):
        # Compute the partial expression tree of the inputs
        new_ops, new_contraction_list = expr(*const_arrays, backend='libadcc',
                                             evaluate_constants=True)

        # Evaluate as much as possible and return
        new_ops = [None if x is None else libadcc.evaluate(x) for x in new_ops]
        return new_ops, new_contraction_list
    EVAL_CONSTS_BACKENDS["libadcc"] = libadcc_evaluate_constants
