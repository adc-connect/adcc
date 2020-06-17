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


def _dispatch_diagonal(subscript, outstring, operand):
    # Do the diagonal call one index at a time.
    char = None
    for c in subscript:
        if subscript.count(c) > 1:
            char = c
            break
    assert char is not None
    indices = [i for (i, c) in enumerate(subscript) if c == char]

    # Diagonal pushes the diagonalised index back
    newoperand = operand.diagonal(*indices)
    newsubscript = "".join(c for c in subscript if c != char) + char
    if len(newsubscript) > len(outstring):
        # Recurse if more letters for which diagonals are to be extracted
        return _dispatch_diagonal(newsubscript, outstring, newoperand)
    elif newsubscript == outstring:
        return newoperand
    else:
        # Transpose newsubscript -> outstring
        permutation = tuple(map(newsubscript.index, outstring))
        return newoperand.transpose(permutation)


def _fallback_einsum(einsum_str, *operands, **kwargs):
    # A fallback implementation of einsum in adcc,
    # which deals with a few cases opt_einsum cannot deal with
    from .functions import einsum
    from opt_einsum.parser import gen_unused_symbols

    operands = list(operands)
    subscripts = einsum_str.split("->")[0].split(",")
    outstr = einsum_str.split("->")[1]

    # If there are any diagonal extractions, which can be done in the operands,
    # do them first.
    for i in range(len(subscripts)):
        sub = subscripts[i]
        cdiagonal = set(c for c in sub if sub.count(c) > 1 and c in outstr)
        ctrace = set(c for c in sub if sub.count(c) > 1 and c not in outstr)
        if ctrace:
            raise NotImplementedError("Partial traces (e.g. contractions "
                                      "'iaib->ab') are not yet supported "
                                      "in adcc.einsum.")
        if cdiagonal:
            # Do any possible diagonal extraction first
            outstring = "".join(c for c in sub if c not in cdiagonal)
            outstring += "".join(cdiagonal)
            operands[i] = _dispatch_diagonal(subscripts[i], outstring,
                                             operands[i])
            subscripts[i] = outstring

    if len(subscripts) == 1:
        # At this point all which is left should be a permutation.
        assert all(c in outstr for c in subscripts[0])
        permutation = tuple(map(subscripts[0].index, outstr))
        return operands[0].transpose(permutation)
    elif len(subscripts) == 2:
        # Should the diagonal of a contraction be extracted, e.g. il,laib->aib
        diagonal_chars = [a for a in subscripts[0]
                          if a in subscripts[1] and a in outstr]

        if not diagonal_chars:
            # Try another round of einsum
            return einsum(",".join(subscripts) + "->" + outstr, *operands)

        # Replace one of the duplicate characters in the input
        # and prepend it to output
        replacers = list(gen_unused_symbols(outstr + "".join(subscripts),
                                            len(diagonal_chars)))
        newoutstr = "".join(replacers) + outstr
        for (old, new) in zip(diagonal_chars, replacers):
            subscripts[0] = subscripts[0].replace(old, new)

        # Check we are not creating an infinite loop:
        assert ",".join(subscripts) + "->" + newoutstr != einsum_str

        # Run einsum doing the partial contraction
        # (well actually we should directly do tensordot)
        res = einsum(",".join(subscripts) + "->" + newoutstr, *operands)

        # Run _dispatch_diagonal with the result to form the requested diagonal
        return _dispatch_diagonal("".join(diagonal_chars) + outstr, outstr, res)
    else:
        raise NotImplementedError("Fallback einsum not implemented for more than "
                                  "two operators")


def register_with_opt_einsum():
    import libadcc

    from opt_einsum.backends.dispatch import (EVAL_CONSTS_BACKENDS,
                                              _cached_funcs, _has_einsum)

    def libadcc_evaluate_constants(const_arrays, expr):
        # Compute the partial expression tree of the inputs
        new_ops, new_contraction_list = expr(*const_arrays, backend='libadcc',
                                             evaluate_constants=True)

        # Evaluate as much as possible and return
        new_ops = [None if x is None else libadcc.evaluate(x) for x in new_ops]
        return new_ops, new_contraction_list
    EVAL_CONSTS_BACKENDS["libadcc"] = libadcc_evaluate_constants
    _has_einsum["libadcc"] = False
    _cached_funcs[('einsum', 'libadcc')] = _fallback_einsum
