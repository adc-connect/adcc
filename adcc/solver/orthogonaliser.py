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
import numpy as np

from adcc import evaluate, lincomb


class GramSchmidtOrthogonaliser:
    def __init__(self, explicit_symmetrisation=None, n_rounds=1):
        """
        Initialise the GramSchmidtOrthogonaliser

        Parameters
        ----------
        explicit_symmetrisation
            The explicit symmetrisation to use on each orthogonalised vector
        n_rounds : int
            The number of times to apply the (regular) Gram-Schmidt
        """
        self.explicit_symmetrisation = explicit_symmetrisation
        self.n_rounds = n_rounds

    def qr(self, vectors):
        """
        A simple (and inefficient / inaccurate) QR decomposition based
        on Gram-Schmidt. Use only if no alternatives.

        vectors : list
            List of vectors representing the input matrix to decompose.
        """
        if len(vectors) == 0:
            return []
        elif len(vectors) == 1:
            norm_v = np.sqrt(vectors[0] @ vectors[0])
            return [evaluate(vectors[0] / norm_v)], np.array([[norm_v]])
        else:
            n_vec = len(vectors)
            Q = self.orthogonalise(vectors)
            R = np.zeros((n_vec, n_vec))
            for i in range(n_vec):
                for j in range(i, n_vec):
                    R[i, j] = Q[i] @ vectors[j]
            return Q, R

    def orthogonalise(self, vectors):
        """
        Orthogonalise the passed vectors with each other and return
        orthonormal vectors.
        """
        if len(vectors) == 0:
            return []
        subspace = [evaluate(vectors[0] / np.sqrt(vectors[0] @ vectors[0]))]
        for v in vectors[1:]:
            w = self.orthogonalise_against(v, subspace)
            subspace.append(evaluate(w / np.sqrt(w @ w)))
        return subspace

    def orthogonalise_against(self, vector, subspace):
        """
        Orthogonalise the passed vector against a subspace. The latter is assumed
        to only consist of orthonormal vectors. Effectively computes
        ``(1 - SS * SS^T) * vector`.

        vector
            Vector to make orthogonal to the subspace
        subspace : list
            Subspace of orthonormal vectors.
        """
        # Project out the components of the current subspace
        # That is form (1 - SS * SS^T) * vector = vector + SS * (-SS^T * vector)
        for _ in range(self.n_rounds):
            coefficients = np.hstack(([1], -(vector @ subspace)))
            vector = lincomb(coefficients, [vector] + subspace, evaluate=True)
            if self.explicit_symmetrisation is not None:
                self.explicit_symmetrisation.symmetrise(vector)
        return vector
