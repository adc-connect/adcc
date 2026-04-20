import numpy as np
from itertools import product


class DIIS:
    # adapted from
    # https://github.com/edeprince3/pdaggerq/blob/master/examples/full_cc_codes/diis.py
    def __init__(self, max_subspace=10, start_iter=5):
        self.nvecs = max_subspace
        self.error_vecs = []
        self.prev_vecs = []
        self.start_iter = start_iter
        self.iter = 0

    def extrapolate(self, iterate, error):
        if self.iter < self.start_iter:
            self.iter += 1
            return iterate

        self.prev_vecs.append(iterate)
        self.error_vecs.append(error)
        self.iter += 1

        if len(self.prev_vecs) > self.nvecs:
            self.prev_vecs.pop(0)
            self.error_vecs.pop(0)

        b_mat, rhs = self.get_bmatrix()
        c = np.linalg.solve(b_mat, rhs)
        c = c.flatten()

        new_iterate = self.prev_vecs[0].zeros_like()
        for ii in range(len(self.prev_vecs)):
            new_iterate += c[ii] * self.prev_vecs[ii]
        return new_iterate.evaluate()

    def get_bmatrix(self):
        dim = len(self.prev_vecs)
        b = np.zeros((dim, dim))
        for i, j in product(range(dim), repeat=2):
            if i <= j:
                b[i, j] = self.error_vecs[i].dot(self.error_vecs[j])
                b[j, i] = b[i, j]
        b = np.hstack((b, -1 * np.ones((dim, 1))))
        b = np.vstack((b, -1 * np.ones((1, dim + 1))))
        b[-1, -1] = 0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1, 0] = -1
        return b, rhs