import numpy as np
import pytest

from adcc.solver.fixed_point_diis import diis, SubspaceError, default_print
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc import AdcMatrix, AmplitudeVector, guess_zero, guesses_triplet

from ..testdata_cache import testdata_cache


class TestDIIS:
    def adc_updater(self, matrix: AdcMatrix):
        precond = JacobiPreconditioner(matrix)

        def updater(vec):
            # assume that the input vector is normalized
            mvp = (matrix @ vec).evaluate()
            new_vec = (vec - precond.apply(mvp - vec.dot(mvp) * vec)).evaluate()
            # normalize the new vector
            new_vec /= np.sqrt(new_vec.dot(new_vec))
            return new_vec

        return updater

    def test_adc1(self):
        # try to converge the lowest excited state
        conv_tol = 1e-9
        matrix = AdcMatrix(
            "adc1", testdata_cache.refstate("h2o_sto3g", case="gen")
        )
        # triplet is lower in energy than singlet and numpy just computes
        # all states independent of their spin
        guess = guesses_triplet(matrix, 1, block="ph")[0]
        assert isinstance(guess, AmplitudeVector)
        solution = diis(updater=self.adc_updater(matrix), guess_vector=guess,
                        conv_tol=conv_tol,
                        callback=default_print)
        # ensure the solution is normalized
        assert pytest.approx(np.sqrt(solution @ solution)) == 1
        # ensure that we indeed converged by computing the residual:
        # MY - wY = 0
        mvp = matrix @ solution
        omega = solution.dot(mvp)
        residual = mvp - omega * solution
        assert isinstance(residual, AmplitudeVector)
        assert np.sqrt(residual.dot(residual)) < conv_tol
        # ensure that we converged on the correct energy
        # could either load the result from the reference data or
        # diagonalize the full matrix with numpy
        np_matrix = matrix.to_ndarray()
        ref_eigvals, _ = np.linalg.eigh(np_matrix)
        assert pytest.approx(omega) == ref_eigvals[0]

    def test_adc1_random(self):
        # try to converge the lowest excited state
        conv_tol = 1e-9
        matrix = AdcMatrix(
            "adc1", testdata_cache.refstate("h2o_sto3g", case="gen")
        )
        # I guess with a random guess it is VERY likely that we converge onto
        # the lowest state, but not guaranteed if the random guess is very
        # close to another eigenstate -> only verify convergence in this test
        guess = guess_zero(matrix)
        guess.set_random()
        solution = diis(updater=self.adc_updater(matrix), guess_vector=guess,
                        conv_tol=conv_tol,
                        callback=default_print)
        # ensure the solution is normalized
        assert pytest.approx(np.sqrt(solution @ solution)) == 1
        # ensure that we indeed converged by computing the residual:
        # MY - wY = 0
        mvp = matrix @ solution
        omega = solution.dot(mvp)
        residual = mvp - omega * solution
        assert isinstance(residual, AmplitudeVector)
        assert np.sqrt(residual.dot(residual)) < conv_tol
