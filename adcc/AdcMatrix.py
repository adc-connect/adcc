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
import libadcc
import warnings
import numpy as np

from .LazyMp import LazyMp
from .adc_pp import matrix as ppmatrix
from .timings import Timer, timed_member_call
from .AdcMethod import AdcMethod
from .Intermediates import Intermediates
from .AmplitudeVector import AmplitudeVector


class AdcExtraTerm:
    def __init__(self, matrix, blocks):
        """Initialise an AdcExtraTerm.
        This class can be used to add customs terms
        to an existing :py:class:`AdcMatrix`

        Parameters
        ----------
        matrix : AdcMatrix
            The matrix for which the extra term
            should be created.
        blocks : dict
            A dictionary where the key labels the matrix block
            and the item denotes a callable to construct
            an :py:class:`AdcBlock`
        """
        self.ground_state = matrix.ground_state
        self.reference_state = matrix.reference_state
        self.intermediates = matrix.intermediates
        self.blocks = {}
        if not isinstance(blocks, dict):
            raise TypeError("blocks needs to be a dict.")
        for space in blocks:
            block_fun = blocks[space]
            if not callable(block_fun):
                raise TypeError("Items in additional_blocks must be callable.")
            block = block_fun(
                self.reference_state, self.ground_state, self.intermediates
            )
            self.blocks[space] = block


class AdcMatrixlike:
    """
    Base class marker for all objects like ADC matrices.
    """
    pass


class AdcMatrix(AdcMatrixlike):
    # Default perturbation-theory orders for the matrix blocks (== standard ADC-PP).
    default_block_orders = {
        #             ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),
        "adc0":  dict(ph_ph=0, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc1":  dict(ph_ph=1, ph_pphh=None, pphh_ph=None, pphh_pphh=None),  # noqa: E501
        "adc2":  dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=0),     # noqa: E501
        "adc2x": dict(ph_ph=2, ph_pphh=1,    pphh_ph=1,    pphh_pphh=1),     # noqa: E501
        "adc3":  dict(ph_ph=3, ph_pphh=2,    pphh_ph=2,    pphh_pphh=1),     # noqa: E501
    }

    def __init__(self, method, hf_or_mp, block_orders=None, intermediates=None,
                 diagonal_precomputed=None):
        """
        Initialise an ADC matrix.

        Parameters
        ----------
        method : str or AdcMethod
            Method to use.
        hf_or_mp : adcc.ReferenceState or adcc.LazyMp
            HF reference or MP ground state
        block_orders : optional
            The order of perturbation theory to employ for each matrix block.
            If not set, defaults according to the selected ADC method are chosen.
        intermediates : adcc.Intermediates or NoneType
            Allows to pass intermediates to re-use to this class.
        diagonal_precomputed: adcc.AmplitudeVector
            Allows to pass a pre-computed diagonal, for internal use only.
        """
        if isinstance(hf_or_mp, (libadcc.ReferenceState,
                                 libadcc.HartreeFockSolution_i)):
            hf_or_mp = LazyMp(hf_or_mp)
        if not isinstance(hf_or_mp, LazyMp):
            raise TypeError("hf_or_mp is not a valid object. It needs to be "
                            "either a LazyMp, a ReferenceState or a "
                            "HartreeFockSolution_i.")

        if not isinstance(method, AdcMethod):
            method = AdcMethod(method)

        if diagonal_precomputed:
            if not isinstance(diagonal_precomputed, AmplitudeVector):
                raise TypeError("diagonal_precomputed needs to be"
                                " an AmplitudeVector.")
            if diagonal_precomputed.needs_evaluation:
                raise ValueError("diagonal_precomputed must already"
                                 " be evaluated.")

        self.timer = Timer()
        self.method = method
        self.ground_state = hf_or_mp
        self.reference_state = hf_or_mp.reference_state
        self.mospaces = hf_or_mp.reference_state.mospaces
        self.is_core_valence_separated = method.is_core_valence_separated
        self.ndim = 2
        self.extra_terms = []

        self.intermediates = intermediates
        if self.intermediates is None:
            self.intermediates = Intermediates(self.ground_state)

        # Determine orders of PT in the blocks
        if block_orders is None:
            block_orders = self.default_block_orders[method.base_method.name]
        else:
            tmp_orders = self.default_block_orders[method.base_method.name].copy()
            tmp_orders.update(block_orders)
            block_orders = tmp_orders

        # Sanity checks on block_orders
        for block in block_orders.keys():
            if block not in ("ph_ph", "ph_pphh", "pphh_ph", "pphh_pphh"):
                raise ValueError(f"Invalid block order key: {block}")
        if block_orders["ph_pphh"] != block_orders["pphh_ph"]:
            raise ValueError("ph_pphh and pphh_ph should always have "
                             "the same order")
        if block_orders["ph_pphh"] is not None \
           and block_orders["pphh_pphh"] is None:
            raise ValueError("pphh_pphh cannot be None if ph_pphh isn't.")
        self.block_orders = block_orders

        # Build the blocks and diagonals
        with self.timer.record("build"):
            variant = None
            if self.is_core_valence_separated:
                variant = "cvs"
            blocks = {
                block: ppmatrix.block(self.ground_state, block.split("_"),
                                      order=order, intermediates=self.intermediates,
                                      variant=variant)
                for block, order in self.block_orders.items() if order is not None
            }
            # TODO Rename to self.block in 0.16.0
            self.blocks_ph = {bl: blocks[bl].apply for bl in blocks}
            if diagonal_precomputed:
                self.__diagonal = diagonal_precomputed
            else:
                self.__diagonal = sum(bl.diagonal for bl in blocks.values()
                                      if bl.diagonal)
                self.__diagonal.evaluate()
            self.__init_space_data(self.__diagonal)

    def __iadd__(self, other):
        """In-place addition of an :py:class:`AdcExtraTerm`

        Parameters
        ----------
        other : AdcExtraTerm
            the extra term to be added
        """
        if not isinstance(other, AdcExtraTerm):
            return NotImplemented
        if not all(k in self.blocks_ph for k in other.blocks):
            raise ValueError("Can only add to blocks of"
                             " AdcMatrix that already exist.")
        for sp in other.blocks:
            orig_app = self.blocks_ph[sp]
            other_app = other.blocks[sp].apply

            def patched_apply(ampl, original=orig_app, other=other_app):
                return sum(app(ampl) for app in (original, other))
            self.blocks_ph[sp] = patched_apply
        other_diagonal = sum(bl.diagonal for bl in other.blocks.values()
                             if bl.diagonal)
        self.__diagonal = self.__diagonal + other_diagonal
        self.__diagonal.evaluate()
        self.extra_terms.append(other)
        return self

    def __add__(self, other):
        """Addition of an :py:class:`AdcExtraTerm`, creating
        a copy of self and adding the term to the new matrix

        Parameters
        ----------
        other : AdcExtraTerm
            the extra term to be added

        Returns
        -------
        AdcMatrix
            a copy of the AdcMatrix with the extra term added
        """
        if not isinstance(other, AdcExtraTerm):
            return NotImplemented
        ret = AdcMatrix(self.method, self.ground_state,
                        block_orders=self.block_orders,
                        intermediates=self.intermediates,
                        diagonal_precomputed=self.diagonal())
        ret += other
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __init_space_data(self, diagonal):
        """Update the cached data regarding the spaces of the ADC matrix"""
        self.axis_spaces = {}
        self.axis_lengths = {}
        for block in diagonal.blocks_ph:
            self.axis_spaces[block] = getattr(diagonal, block).subspaces
            self.axis_lengths[block] = np.prod([
                self.mospaces.n_orbs(sp) for sp in self.axis_spaces[block]
            ])
        self.shape = (sum(self.axis_lengths.values()),
                      sum(self.axis_lengths.values()))

    def __repr__(self):
        ret = f"AdcMatrix({self.method.name}, "
        for b, o in self.block_orders.items():
            ret += f"{b}={o}, "
        return ret + ")"

    def __len__(self):
        return self.shape[0]

    @property
    def blocks(self):
        # TODO Remove in 0.16.0
        return self.__diagonal.blocks

    def has_block(self, block):
        warnings.warn("The has_block function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `in matrix.axis_blocks` in the future.")
        return self.block_spaces(block) is not None

    def block_spaces(self, block):
        warnings.warn("The block_spaces function is deprecated and "
                      "will be removed in 0.16.0. "
                      "Use `matrix.axis_spaces[block]` in the future.")
        return {
            "s": self.axis_spaces.get("ph", None),
            "d": self.axis_spaces.get("pphh", None),
            "t": self.axis_spaces.get("ppphhh", None),
        }[block]

    @property
    def axis_blocks(self):
        """
        Return the blocks used along one of the axes of the ADC matrix
        (e.g. ['ph', 'pphh']).
        """
        return list(self.axis_spaces.keys())

    def diagonal(self, block=None):
        """Return the diagonal of the ADC matrix"""
        if block is not None:
            warnings.warn("Support for the block argument will be dropped "
                          "in 0.16.0.")
            if block == "s":
                return self.__diagonal.ph
            if block == "d":
                return self.__diagonal.pphh
        return self.__diagonal

    def compute_apply(self, block, tensor):
        warnings.warn("The compute_apply function is deprecated and "
                      "will be removed in 0.16.0.")
        if block in ("ss", "sd", "ds", "dd"):
            warnings.warn("The singles-doubles interface is deprecated and "
                          "will be removed in 0.16.0.")
            block = {"ss": "ph_ph", "sd": "ph_pphh",
                     "ds": "pphh_ph", "dd": "pphh_pphh"}[block]
        return self.block_apply(block, tensor)

    def block_apply(self, block, tensor):
        """
        Compute the application of a block of the ADC matrix
        with another AmplitudeVector or Tensor. Non-matching blocks
        in the AmplitudeVector will be ignored.
        """
        if not isinstance(tensor, libadcc.Tensor):
            raise TypeError("tensor should be an adcc.Tensor")

        with self.timer.record(f"apply/{block}"):
            outblock, inblock = block.split("_")
            ampl = AmplitudeVector(**{inblock: tensor})
            ret = self.blocks_ph[block](ampl)
            return getattr(ret, outblock)

    @timed_member_call()
    def matvec(self, v):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        return sum(block(v) for block in self.blocks_ph.values())

    def rmatvec(self, v):
        # ADC matrix is symmetric
        return self.matvec(v)

    def compute_matvec(self, ampl):
        """
        Compute the matrix-vector product of the ADC matrix
        with an excitation amplitude and return the result.
        """
        warnings.warn("The compute_matvec function is deprecated and "
                      "will be removed in 0.16.0.")
        return self.matvec(ampl)

    def __matmul__(self, other):
        if isinstance(other, AmplitudeVector):
            return self.matvec(other)
        if isinstance(other, list):
            if all(isinstance(elem, AmplitudeVector) for elem in other):
                return [self.matvec(ov) for ov in other]
        return NotImplemented

    def block_view(self, block):
        """
        Return a view into the AdcMatrix that represents a single
        block of the matrix. Currently only diagonal blocks are supported.
        """
        b1, b2 = block.split("_")
        if b1 != b2:
            raise NotImplementedError("Off-diagonal block views not yet "
                                      "implemented.")
            # TODO For off-diagonal blocks we probably need a different
            #      data structure as the AdcMatrix class as these block
            #      are inherently different than an AdcMatrix (no Hermiticity
            #      for example) and basically they only need to support some
            #      form of matrix-vector product and some stastics like
            #      spaces and sizes etc.
        block_orders = {bl: None for bl in self.block_orders.keys()}
        block_orders[block] = self.block_orders[block]
        return AdcMatrix(self.method, self.ground_state,
                         block_orders=block_orders,
                         intermediates=self.intermediates)

    def construct_symmetrisation_for_blocks(self):
        """
        Construct the symmetrisation functions, which need to be
        applied to relevant blocks of an AmplitudeVector in order
        to symmetrise it to the right symmetry in order to be used
        with the various matrix-vector-products of this function.

        Most importantly the returned functions antisymmetrise
        the occupied and virtual parts of the doubles parts
        if this is sensible for the method behind this adcmatrix.

        Returns a dictionary block identifier -> function
        """
        ret = {}
        if self.is_core_valence_separated:
            # CVS doubles part is antisymmetric wrt. (i,K,a,b) <-> (i,K,b,a)
            ret["pphh"] = lambda v: v.antisymmetrise([(2, 3)])
        else:
            def symmetrise_generic_adc_doubles(invec):
                # doubles part is antisymmetric wrt. (i,j,a,b) <-> (i,j,b,a)
                scratch = invec.antisymmetrise([(2, 3)])
                # doubles part is symmetric wrt. (i,j,a,b) <-> (j,i,b,a)
                return scratch.symmetrise([(0, 1), (2, 3)])
            ret["pphh"] = symmetrise_generic_adc_doubles
        return ret

    def dense_basis(self, axis_blocks=None, ordering="adcc"):
        """
        Return the list of indices and their values
        of the dense basis representation

        ordering: adcc, spin, spatial
        """
        ret = []
        if axis_blocks is None:
            axis_blocks = self.axis_blocks
        if not isinstance(axis_blocks, list):
            axis_blocks = [axis_blocks]

        # Define function to impose the order in the basis
        if ordering == "adcc":
            def reduce_index(n_orbsa, idx):
                return idx, idx
        elif ordering == "spin":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spin, then by spatial
                return (is_beta, spatial)
        elif ordering == "spatial":
            def reduce_index(n_orbsa, idx):
                is_beta = [idx[i] >= n_orbsa[i] for i in range(len(idx))]
                spatial = [idx[i] - n_orbsa[i] if is_beta[i] else idx[i]
                           for i in range(len(idx))]
                # Sort first by spatial, then by spin
                return (spatial, is_beta)

        if "ph" in axis_blocks:
            ret_s = []
            sp_s = self.axis_spaces["ph"]
            n_orbs_s = [self.mospaces.n_orbs(sp) for sp in sp_s]
            n_orbsa_s = [self.mospaces.n_orbs_alpha(sp) for sp in sp_s]
            for i in range(n_orbs_s[0]):
                for a in range(n_orbs_s[1]):
                    ret_s.append([((i, a), 1)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_s, idx) for idx, factor in x)
            ret_s.sort(key=sortfctn)
            ret_s.sort(key=sortfctn)
            ret.extend(ret_s)

        if "pphh" in axis_blocks:
            ret_d = []
            sp_d = self.axis_spaces["pphh"]
            n_orbsa_d = [self.mospaces.n_orbs_alpha(sp) for sp in sp_d]

            if sp_d[0] == sp_d[1] and sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / 2),
                               ((j, i, a, b), -1 / 2),
                               ((i, j, b, a), -1 / 2),
                               ((j, i, b, a), +1 / 2)]
                              for i in range(nso) for j in range(i)
                              for a in range(nsv) for b in range(a)])
            elif sp_d[2] == sp_d[3]:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                ret_d.extend([[((i, j, a, b), +1 / np.sqrt(2)),
                               ((i, j, b, a), -1 / np.sqrt(2))]
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(a)])
            else:
                nso = self.mospaces.n_orbs(sp_d[0])
                nsc = self.mospaces.n_orbs(sp_d[1])
                nsv = self.mospaces.n_orbs(sp_d[2])
                nsw = self.mospaces.n_orbs(sp_d[3])
                ret_d.append([((i, j, b, a), 1)
                              for i in range(nso) for j in range(nsc)
                              for a in range(nsv) for b in range(nsw)])

            def sortfctn(x):
                return min(reduce_index(n_orbsa_d, idx) for idx, factor in x)
            ret_d.sort(key=sortfctn)
            ret_d.sort(key=sortfctn)
            ret.extend(ret_d)

        if any(b not in ("ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than ph and pphh "
                                      "not implemented")
        return ret

    def to_ndarray(self, out=None):
        """
        Return the ADC matrix object as a dense numpy array. Converts the sparse
        internal representation of the ADC matrix to a dense matrix and return
        as a numpy array.

        Notes
        -----

        This method is only intended to be used for debugging and
        visualisation purposes as it involves computing a large amount of
        matrix-vector products and the returned array consumes a considerable
        amount of memory.

        The resulting matrix has no spin symmetry imposed, which means that
        its eigenspectrum may contain non-physical excitations (e.g. with linear
        combinations of α->β and α->α components in the excitation vector).

        This function has not been sufficiently tested to be considered stable.
        """
        # TODO Update to ph / pphh
        # TODO Still uses deprecated functions
        import tqdm

        from adcc import guess_zero

        # Get zero amplitude of the appropriate symmetry
        # (TODO: Only true for C1, where there is only a single irrep)
        ampl_zero = guess_zero(self)
        assert self.mospaces.point_group == "C1"

        # Build the shape of the returned array
        # Since the basis of the doubles block is not the unit vectors
        # this *not* equal to the shape of the AdcMatrix object
        basis = {b: self.dense_basis(b) for b in self.axis_blocks}
        mat_len = sum(len(basis[b]) for b in basis)

        if out is None:
            out = np.zeros((mat_len, mat_len))
        else:
            if out.shape != (mat_len, mat_len):
                raise ValueError("Output array has shape ({0:}, {1:}), but "
                                 "shape ({2:}, {2:}) is required."
                                 "".format(*out.shape, mat_len))
            out[:] = 0  # Zero all data in out.

        # Check for the cases actually implemented
        if any(b not in ("ph", "pphh") for b in self.axis_blocks):
            raise NotImplementedError("Blocks other than ph and pphh "
                                      "not implemented")
        if "ph" not in self.axis_blocks:
            raise NotImplementedError("Block 'ph' needs to be present")

        # Extract singles-singles block (contiguous)
        assert "ph" in self.axis_blocks
        n_orbs_ph = [self.mospaces.n_orbs(sp) for sp in self.axis_spaces["ph"]]
        n_ph = np.prod(n_orbs_ph)
        assert len(basis["ph"]) == n_ph
        view_ss = out[:n_ph, :n_ph].reshape(*n_orbs_ph, *n_orbs_ph)
        for i in range(n_orbs_ph[0]):
            for a in range(n_orbs_ph[1]):
                ampl = ampl_zero.copy()
                ampl.ph[i, a] = 1
                view_ss[:, :, i, a] = (self @ ampl).ph.to_ndarray()

        # Extract singles-doubles and doubles-doubles block
        if "pphh" in self.axis_blocks:
            assert self.axis_blocks == ["ph", "pphh"]
            view_sd = out[:n_ph, n_ph:].reshape(*n_orbs_ph, len(basis["pphh"]))
            view_dd = out[n_ph:, n_ph:]
            for j, bas1 in tqdm.tqdm(enumerate(basis["pphh"]),
                                     total=len(basis["pphh"])):
                ampl = ampl_zero.copy()
                for idx, val in bas1:
                    ampl.pphh[idx] = val
                ret_ampl = self @ ampl
                view_sd[:, :, j] = ret_ampl.ph.to_ndarray()

                for i, bas2 in enumerate(basis["pphh"]):
                    view_dd[i, j] = sum(val * ret_ampl.pphh[idx]
                                        for idx, val in bas2)

            out[n_ph:, :n_ph] = np.transpose(out[:n_ph, n_ph:])
        return out


class AdcBlockView(AdcMatrix):
    def __init__(self, fullmatrix, block):
        warnings.warn("The AdcBlockView class got deprecated and will be "
                      "removed in 0.16.0. Use the matrix.block_view "
                      "function instead.")
        assert isinstance(fullmatrix, AdcMatrix)

        self.__fullmatrix = fullmatrix
        self.__block = block
        if block == "s":
            block_orders = dict(ph_ph=fullmatrix.block_orders["ph_ph"],
                                ph_pphh=None, pphh_ph=None, pphh_pphh=None)
        else:
            raise NotImplementedError(f"Block {block} not implemented")
        super().__init__(fullmatrix.method, fullmatrix.ground_state,
                         block_orders=block_orders,
                         intermediates=fullmatrix.intermediates)


class AdcMatrixShifted(AdcMatrix):
    def __init__(self, matrix, shift=0.0):
        """
        Initialise a shifted ADC matrix. Applying this class to a vector ``v``
        represents an efficient version of ``matrix @ v + shift * v``.

        Parameters
        ----------
        matrix : AdcMatrix
            Matrix which is shifted
        shift : float
            Value by which to shift the matrix
        """
        super().__init__(matrix.method, matrix.ground_state,
                         block_orders=matrix.block_orders,
                         intermediates=matrix.intermediates)
        self.shift = shift

    def matvec(self, in_ampl):
        out = super().matvec(in_ampl)
        out = out + self.shift * in_ampl
        return out

    def to_ndarray(self, out=None):
        super().to_ndarray(self, out)
        out = out + self.shift * np.eye(*out.shape)
        return out

    def block_apply(self, block, in_vec):
        ret = super().block_apply(block, in_vec)
        inblock, outblock = block.split("_")
        if inblock == outblock:
            ret += self.shift * in_vec
        return ret

    def diagonal(self, block=None):
        out = super().diagonal(block)
        out = out + self.shift  # Shift the diagonal
        return out

    def block_view(self, block):
        raise NotImplementedError("Block-view not yet implemented for "
                                  "shifted ADC matrices.")
        # TODO The way to implement this is to ask the inner matrix to
        #      a block_view and then wrap that in an AdcMatrixShifted.
