import numpy as np
import pytest

from adcc.solver.fixed_point_diis import (
    diis, SubspaceError, DIISError, DIISSubspace, default_print
)
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
        # if we now restart the diis with the converged solution we should
        # converge instantly (after performing a single linear step)
        other_solution = diis(
            updater=self.adc_updater(matrix), guess_vector=solution,
            conv_tol=conv_tol, callback=default_print,
            n_max_iterations=1
        )
        diff = other_solution - solution
        assert np.sqrt(diff.dot(diff)) < conv_tol


class TestDIISSubspace:
    def test_update_overlap(self):
        # manually add some simple error vectors to the subspace and
        # test that the overlap is correctly computed
        subspace = DIISSubspace(max_size=5, start_size=5)

        assert subspace.overlap.size == 0
        with pytest.raises(SubspaceError):
            subspace._update_overlap()

        e1 = DummyVec(np.array([1, 0, 0]))
        e2 = DummyVec(np.array([0, 1, 0]))
        e3 = DummyVec(np.array([1, 1, 0]))

        subspace.error_vectors.append(e1)
        subspace._update_overlap()
        assert subspace.overlap.shape == (1, 1)
        assert subspace.overlap[(0, 0)] == 1

        subspace.error_vectors.append(e2)
        subspace._update_overlap()
        assert subspace.overlap.shape == (2, 2)
        assert (subspace.overlap == np.array([[1, 0], [0, 1]])).all()

        subspace.error_vectors.append(e3)
        subspace._update_overlap()
        assert subspace.overlap.shape == (3, 3)
        assert (subspace.overlap == np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 2]
        ])).all()

    def test_add(self):
        # ensure that the vectors are correctly added, n_iter is inscremented
        # and the error overlap matrix is updated
        subspace = DIISSubspace(3, 3)
        v1 = DummyVec(np.array([1, 0, 0]))
        e1 = DummyVec(np.array([0, 1, 1]))
        # everything should start out empty
        assert not subspace.error_vectors and not subspace.subspace_vectors
        assert not subspace.n_iter
        assert not subspace.overlap.size

        subspace.add(subspace_vector=v1, error_vector=e1)
        assert len(subspace.error_vectors) == 1
        assert (subspace.error_vectors[0].array == e1.array).all()
        assert len(subspace.subspace_vectors) == 1
        assert (subspace.subspace_vectors[0].array == v1.array).all()
        assert subspace.n_iter == 1
        assert subspace.overlap.shape == (1, 1)
        assert subspace.overlap[(0, 0)] == 2

    def test_compute_guess(self):
        # ensure that the DIIS extrapolation works correctly
        # solution: [1, 1, 1, 1]
        subspace = DIISSubspace(max_size=4, start_size=2)

        assert not subspace.size and not subspace.n_iter
        with pytest.raises(SubspaceError):
            subspace.compute_guess()

        v1 = DummyVec(np.array([1, 0, 0, 0]))
        e1 = DummyVec(np.array([0, 1, 1, 1]))
        v2 = DummyVec(np.array([0, 1, 0, 0]))
        e2 = DummyVec(np.array([1, 0, 1, 1]))
        v3 = DummyVec(np.array([0, 0, 2, 0]))
        e3 = DummyVec(np.array([1, 1, -1, 1]))

        # add the vectors to the subspace and update the overlap
        subspace.add(subspace_vector=v1, error_vector=e1)
        assert subspace.size == 1 and subspace.n_iter == 1
        # can not omit or only vector
        with pytest.raises(SubspaceError):
            subspace.compute_guess(n_omit_vectors=1)
        # 1 vector -> below start_size -> no extrapolation (v1 returned)
        g1 = subspace.compute_guess()
        assert (v1.array == g1.array).all()

        subspace.add(subspace_vector=v2, error_vector=e2)
        assert subspace.size == 2 and subspace.n_iter == 2
        # if we omit 1 vector -> no extrapolation
        g2 = subspace.compute_guess(n_omit_vectors=1)
        assert (v2.array == g2.array).all()
        # considering both vectors we can now extrapolate a new vector
        assert (subspace.overlap == np.array([[3, 2], [2, 3]])).all()
        # overlap is rescaled leading to the system of linear equations Ax = b
        #     ( 3  2 -1)    ( 1    2/3 -1)        ( 0)
        # A = ( 2  3 -1) -> ( 2/3  1   -1)    b = ( 0)
        #     (-1 -1  0)    (-1   -1    0)        (-1)
        # solution: [1/2, 1/2, 5/6]
        g2 = subspace.compute_guess()
        ref = v1 * 0.5
        ref += v2 * 0.5
        # due to some numerical problems it is not possible to directly compare
        np.testing.assert_allclose(g2.array, ref.array, atol=1e-14)

        subspace.add(subspace_vector=v3, error_vector=e3)
        assert subspace.size == 3 and subspace.n_iter == 3
        # if we omit 2 vectors -> no extrapolation
        g3 = subspace.compute_guess(n_omit_vectors=2)
        assert (v3.array == g3.array).all()
        # if we omit 1 vector
        # -> extrapolation but without v3, because it has the largest error norm
        # -> should end up with g2 again
        g3 = subspace.compute_guess(n_omit_vectors=1)
        assert (g3.array == g2.array).all()
        # solving another system of linear equations
        g3 = subspace.compute_guess()
        #     ( 3  2  1 -1)    ( 1    2/3  1/3  -1)        ( 0)
        # A = ( 2  3  1 -1) -> ( 2/3  1    1/3  -1)    b = ( 0)
        #     ( 1  1  4 -1)    ( 1/3  1/3  4/3  -1)        ( 0)
        #     (-1 -1 -1  0)    (-1   -1   -1     0)        (-1)
        # solution = [1/3, 1/3, 1/3, 2/3]
        ref = v1 * (1 / 3)
        ref += v2 * (1 / 3)
        ref += v3 * (1 / 3)
        # due to some numerical problems it is not possible to directly compare
        np.testing.assert_allclose(g3.array, ref.array, atol=1e-14)
