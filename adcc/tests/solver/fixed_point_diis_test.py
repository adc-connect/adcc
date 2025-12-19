import numpy as np
import pytest

from adcc.solver.fixed_point_diis import diis, SubspaceError, DIISError, default_print
from adcc.solver.preconditioner import JacobiPreconditioner
from adcc import AdcMatrix, AmplitudeVector, guess_zero, guesses_triplet

from ..testdata_cache import testdata_cache


# wrapper around a numpy array to enable the DIIS with numpy arrays
class DummyVec:
    def __init__(self, array: np.ndarray):
        self.array = array

    def dot(self, other: "DummyVec") -> float:
        return self.array.dot(other.array)

    def zeros_like(self) -> "DummyVec":
        return DummyVec(np.zeros(self.array.shape))

    def __sub__(self, other: "DummyVec") -> "DummyVec":
        return DummyVec(self.array - other.array)

    def __mul__(self, other: float) -> "DummyVec":
        return DummyVec(self.array * other)

    def __iadd__(self, other: "DummyVec") -> "DummyVec":
        self.array += other.array
        return self


class TestDIIS:
    def test_subspace(self):
        guess = DummyVec(np.ones((2, 2)))
        # bad DIIS parameters:
        # start > subspace_size
        with pytest.raises(SubspaceError):
            diis(lambda x: x, guess_vector=guess,
                 diis_start_size=3, max_subspace_size=2)
        # diis_start_size < 2
        with pytest.raises(SubspaceError):
            diis(lambda x: x, guess_vector=guess,
                 diis_start_size=1, max_subspace_size=2)
        # max_subspace_size < 2
        with pytest.raises(SubspaceError):
            diis(lambda x: x, guess_vector=guess,
                 diis_start_size=1, max_subspace_size=1)

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

    def test_conv_failure(self):
        conv_tol = 1e-9
        matrix = AdcMatrix(
            "adc1", testdata_cache.refstate("h2o_sto3g", case="gen")
        )
        # start with a one guess and set n_max_iterations to a low value
        # such that it is not possible to converge
        guess = guess_zero(matrix)
        guess = guess.ones_like()
        guess /= np.sqrt(guess.dot(guess))
        # 0 iterations are not valid
        with pytest.raises(DIISError):
            diis(
                updater=self.adc_updater(matrix), guess_vector=guess,
                conv_tol=conv_tol, callback=default_print, n_max_iterations=0
            )
        # no convergence after initial linear step
        with pytest.raises(DIISError):
            diis(
                updater=self.adc_updater(matrix), guess_vector=guess,
                conv_tol=conv_tol, callback=default_print, n_max_iterations=1
            )
        # no convergence after 2 linear and 1 DIIS step
        with pytest.raises(DIISError):
            diis(
                updater=self.adc_updater(matrix), guess_vector=guess,
                diis_start_size=2, conv_tol=conv_tol, callback=default_print,
                n_max_iterations=3
            )

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
        # if we now restart the diis with the converged solution we should
        # converge instantly (after performing a single linear step)
        other_solution = diis(
            updater=self.adc_updater(matrix), guess_vector=solution,
            conv_tol=conv_tol, callback=default_print,
            n_max_iterations=1
        )
        diff = other_solution - solution
        assert np.sqrt(diff.dot(diff)) < conv_tol
