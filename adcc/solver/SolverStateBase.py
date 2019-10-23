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
from adcc.timings import Timer


class EigenSolverStateBase:
    def __init__(self, matrix):
        """Initialise an EigenSolverStateBase.

        Parameters
        ----------
        matrix
            Matrix to be diagonalised.
        """
        self.matrix = matrix
        self.eigenvalues = None           # Current eigenvalues
        self.eigenvectors = None          # Current eigenvectors
        self.residual_norms = None        # Current residual norms
        self.converged = False            # Flag whether iteration is converged
        self.n_iter = 0                   # Number of iterations
        self.n_applies = 0                # Number of applies
        self.timer = Timer()              # Construct a new timer

    def describe(self):
        text = ""

        problem = str(self.matrix)
        algorithm = getattr(self, "algorithm", "")

        if self.converged:
            conv = "converged"
        else:
            conv = "NOT CONVERGED"

        text += "+" + 60 * "-" + "+\n"
        text += "| {0:<41s}  {1:>15s} |\n".format(algorithm, conv)
        text += ("| {0:30s} n_iter={1:<3d}  n_applies={2:<5d} |\n"
                 "".format(problem[:30], self.n_iter, self.n_applies))
        text += "+" + 60 * "-" + "+\n"
        text += ("|  #     eigenvalue  res. norm       "
                 "dominant elements       |\n")

        body = "| {0:2d} {1:14.7g}  {2:9.4g}               TODO            |\n"
        for i, vec in enumerate(self.eigenvectors):
            text += body.format(i, self.eigenvalues[i], self.residual_norms[i])
        text += "+" + 60 * "-" + "+"
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("SolverState(...)")
        else:
            pp.text(self.describe())
