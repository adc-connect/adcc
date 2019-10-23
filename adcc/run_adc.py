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
import sys
import warnings

from . import solver
from .guess import (guesses_any, guesses_singlet, guesses_spin_flip,
                    guesses_triplet)
from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .ExcitedStates import ExcitedStates
from .ReferenceState import ReferenceState as adcc_ReferenceState

from .solver.davidson import jacobi_davidson
from .solver.explicit_symmetrisation import (IndexSpinSymmetrisation,
                                             IndexSymmetrisation)

from libadcc import LazyMp, ReferenceState

__all__ = ["run_adc"]


def run_adc(data_or_matrix, n_states=None, kind="any", conv_tol=None,
            solver_method=None, guesses=None, n_guesses=None,
            n_guesses_doubles=None, output=sys.stdout, core_orbitals=None,
            frozen_core=None, frozen_virtual=None, method=None,
            n_singlets=None, n_triplets=None, n_spin_flip=None,
            n_core_orbitals=None,
            **solverargs):
    """Run an ADC calculation.

    Main entry point to run an ADC calculation. The reference to build the ADC
    calculation upon is supplied using the `data_or_matrix` argument.
    `adcc` is pretty flexible here. Possible options include:

        a. Hartree-Fock data from a host program, e.g. a molsturm SCF
           state, a pyscf SCF object or any class implementing the
           :py:class:`adcc.HartreeFockProvider` interface. From this data all
           objects mentioned in (b) to (d) will be implicitly created and will
           become available in the returned state.
        b. A :py:class:`adcc.ReferenceState` object
        c. A :py:class:`adcc.LazyMp` object
        d. A :py:class:`adcc.AdcMatrix` object

    Parameters
    ----------
    data_or_matrix
        Data containing the SCF reference
    n_states : int, optional
    kind : str, optional
    n_singlets : int, optional
    n_triplets : int, optional
    n_spin_flip : int, optional
        Specify the number and kind of states to be computed. Possible values
        for kind are "singlet", "triplet", "spin_flip" and "any", which is
        the default. For unrestricted references clamping spin-pure
        singlets/triplets is currently not possible and kind has to remain as
        "any". For restricted references `kind="singlets"` or `kind="triplets"`
        may be employed to enforce a particular excited states manifold.
        Specifying `n_singlets` is equivalent to setting `kind="singlet"` and
        `n_states=5`. Similarly for `n_triplets` and `n_spin_flip`.
        `n_spin_flip` is only valid for unrestricted references.

    conv_tol : float, optional
        Convergence tolerance to employ in the iterative solver for obtaining
        the ADC vectors (default: `1e-6` or SCF tolerance / 100,
        whatever is larger)

    solver_method : str, optional
        The eigensolver algorithm to use.

    n_guesses : int, optional
        Total number of guesses to compute. By default only guesses derived from
        the singles block of the ADC matrix are employed. See
        `n_guesses_doubles` for alternatives. If no number is given here
        `n_guesses = min(4, 2 * number of excited states to compute)`
        or a smaller number if the number of excitation is estimated to be less
        than the outcome of above formula.

    n_guesses_doubles : int, optional
        Number of guesses to derive from the doubles block. By default none
        unless n_guesses as explicitly given or automatically determined is
        larger than the number of singles guesses, which can be possibly found.

    guesses : list, optional
        Provide the guess vectors to be employed for the ADC run. Takes
        preference over `n_guesses` and `n_guesses_doubles`, such that these
        parameters are ignored.

    output : stream, optional
        Python stream to which output will be written. If `None` all output
        is disabled.

    core_orbitals : int or list or tuple, optional
        The orbitals to be put into the core-occupied space. For ways to
        define the core orbitals see the description in
        :py:class:`adcc.ReferenceState`.
        Required if core-valence separation is applied and the input data is
        given as data from the host program (i.e. option (a) discussed above)

    frozen_core : int or list or tuple, optional
        The orbitals to select as frozen core orbitals (i.e. inactive occupied
        orbitals for both the MP and ADC methods performed). For ways to define
        these see the description in :py:class:`adcc.ReferenceState`.

    frozen_virtual : int or list or tuple, optional
        The orbitals to select as frozen virtual orbitals (i.e. inactive
        virtuals for both the MP and ADC methods performed). For ways to define
        these see the description in :py:class:`adcc.ReferenceState`.

    Other parameters
    ----------------
    max_subspace : int, optional
        Maximal subspace size
    max_iter : int, optional
        Maximal number of iterations

    Returns
    -------
    ExcitedStates
        An :class:`adcc.ExcitedStates` object containing the
        :class:`adcc.AdcMatrix`, the :class:`adcc.LazyMp` ground state and the
        :class:`adcc.ReferenceState` as well as computed eigenpairs.

    Examples
    --------

    Run an ADC(2) calculation on top of a `pyscf` RHF reference of
    hydrogen flouride.

    >>> from pyscf import gto, scf
    ... mol = gto.mole.M(atom="H 0 0 0; F 0 0 1.1", basis="sto-3g")
    ... mf = scf.RHF(mol)
    ... mf.conv_tol_grad = 1e-8
    ... mf.kernel()
    ...
    ... state = adcc.run_adc(mf, method="adc2", n_singlets=3)

    The same thing can also be achieved using the `adcc.adcN` family of
    short-hands (see e.g. :py:func:`adcc.adc2`, :py:func:`adcc.cvs_adc2x`):

    >>> state = adcc.adc2(mf, n_singlets=3)

    Run a CVS-ADC(3) calculation of O2 with one core-occupied orbital

    >>> from pyscf import gto, scf
    ... mol = gto.mole.M(atom="O 0 0 0; O 0 0 1.2", basis="sto-3g")
    ... mf = scf.RHF(mol)
    ... mf.conv_tol_grad = 1e-8
    ... mf.kernel()
    ...
    ... state = adcc.cvs_adc3(mf, core_orbitals=1, n_singlets=3)
    """
    #
    # Input argument sanitisation
    #
    if solver_method is None:
        solver_method = "jacobi_davidson"

    # Step 1: Construct at least ReferenceState
    if not isinstance(data_or_matrix, AdcMatrix) and method is None:
        raise ValueError("method needs to be explicitly provided unless "
                         "data_or_matrix is an AdcMatrix.")
    if method is not None and not isinstance(method, AdcMethod):
        method = AdcMethod(method)

    if n_core_orbitals is not None:
        warnings.warn(DeprecationWarning("n_core_orbitals is a deprecated "
                                         "option. Use core_orbitals instead."))
        core_orbitals = n_core_orbitals

    if not isinstance(data_or_matrix, (ReferenceState, AdcMatrix, LazyMp)):
        if method.is_core_valence_separated and core_orbitals is None:
            raise ValueError("If core-valence separation approximation is "
                             "applied then the number of core orbitals needs "
                             "to be specified via the parameter "
                             "core_orbitals.")
        refstate = adcc_ReferenceState(data_or_matrix,
                                       core_orbitals=core_orbitals,
                                       frozen_core=frozen_core,
                                       frozen_virtual=frozen_virtual)
        data_or_matrix = refstate
    elif core_orbitals is not None:
        mospaces = data_or_matrix.mospaces
        warnings.warn("Ignored core_orbitals parameter because data_or_matrix"
                      " is a ReferenceState, a LazyMp or an AdcMatrix object "
                      " (which has a value of core_orbitals={})."
                      "".format(mospaces.n_orbs_alpha("o2")))
    elif frozen_core is not None:
        mospaces = data_or_matrix.mospaces
        warnings.warn("Ignored frozen_core parameter because data_or_matrix"
                      " is a ReferenceState, a LazyMp or an AdcMatrix object "
                      " (which has a value of frozen_core={})."
                      "".format(mospaces.n_orbs_alpha("o3")))
    elif frozen_virtual is not None:
        mospaces = data_or_matrix.mospaces
        warnings.warn("Ignored frozen_virtual parameter because data_or_matrix"
                      " is a ReferenceState, a LazyMp or an AdcMatrix object "
                      " (which has a value of frozen_virtual={})."
                      "".format(mospaces.n_orbs_alpha("v2")))

    # Step2: Make AdcMatrix
    if isinstance(data_or_matrix, (ReferenceState, LazyMp)):
        matrix = AdcMatrix(method, data_or_matrix)
    elif method is not None and method != data_or_matrix.method:
        print(method, data_or_matrix.method)
        warnings.warn("Ignored method parameter because data_or_matrix is an"
                      " AdcMatrix, which implicitly already defines the method")
    if isinstance(data_or_matrix, AdcMatrix):
        matrix = data_or_matrix
    refstate = matrix.reference_state

    if solver_method != "jacobi_davidson":
        raise NotImplementedError("Only the jacobi_davidson solver_method "
                                  "is implemented.")

    # Determine default ADC convergence tolerance
    if conv_tol is None:
        conv_tol = max(refstate.conv_tol / 100, 1e-6)
    if refstate.conv_tol > conv_tol:
        raise ValueError("Convergence tolerance of SCF results (== {}) needs to"
                         " be lower than ADC convergence tolerance parameter "
                         "conv_tol (== {}).".format(refstate.conv_tol,
                                                    conv_tol))

    # Normalise guess parameters
    if sum(nst is not None for nst in [n_states, n_singlets,
                                       n_triplets, n_spin_flip]) > 1:
        raise ValueError("One May only specify one out of n_states, "
                         "n_singlets, n_triplets and n_spin_flip")

    if n_singlets is not None:
        if not refstate.restricted:
            raise ValueError("The n_singlets parameter may only be employed "
                             "for restricted references")
        kind = "singlet"
        n_states = n_singlets
    if n_triplets is not None:
        if not refstate.restricted:
            raise ValueError("The n_triplets parameter may only be employed "
                             "for restricted references")
        kind = "triplet"
        n_states = n_triplets
    if n_spin_flip is not None:
        if refstate.restricted:
            raise ValueError("The n_spin_flip parameter may only be employed "
                             "for unrestricted references")
        kind = "spin_flip"
        n_states = n_spin_flip

    # Check if there are states to be computed
    if n_states is None or n_states == 0:
        raise ValueError("No excited states to be computed. Specify at least "
                         "one of n_states, n_singlets or n_triplets")
    if n_states < 0:
        raise ValueError("n_states needs to be positive")

    if kind not in ["any", "spin_flip", "singlet", "triplet"]:
        raise ValueError("The kind parameter may only take the values 'any', "
                         "'singlet' or 'triplet'")
    if kind in ["singlet", "triplet"] and not refstate.restricted:
        raise ValueError("kind==singlet and kind==triplet are only valid for "
                         "ADC calculations in combination with a restricted "
                         "ground state.")
    if kind in ["spin_flip"] and refstate.restricted:
        raise ValueError("kind==spin_flip is only valid for "
                         "ADC calculations in combination with an unrestricted "
                         "ground state.")

    explicit_symmetrisation = IndexSymmetrisation
    if kind in ["singlet", "triplet"]:
        explicit_symmetrisation = IndexSpinSymmetrisation(
            matrix, enforce_spin_kind=kind
        )

    # Guess parameters
    if n_guesses_doubles is not None and n_guesses_doubles > 0 \
       and "d" not in matrix.blocks:
        raise ValueError("n_guesses_doubles > 0 is only sensible if the ADC "
                         "method has a doubles block (i.e. it is *not* ADC(0), "
                         "ADC(1) or a variant thereof.")

    #
    # Obtain guesses
    #
    spin_change = None
    if guesses is not None:
        if len(guesses) < n_states:
            raise ValueError("Less guesses provided via guesses (== {}) "
                             "than states to be computed (== {})"
                             "".format(len(guesses), n_states))
        if n_guesses is not None:
            warnings.warn("Ignoring n_guesses parameter, since guesses are "
                          "explicitly provided")
    else:
        if n_guesses is None:
            n_guesses = estimate_n_guesses(matrix, n_states)
        if kind == "spin_flip":
            spin_change = -1

        guess_function = {"any": guesses_any, "singlet": guesses_singlet,
                          "triplet": guesses_triplet,
                          "spin_flip": guesses_spin_flip}
        guesses = find_guesses(guess_function[kind], matrix, n_guesses,
                               n_guesses_doubles=n_guesses_doubles)

    #
    # Run solver
    #
    jd_callback = None
    if output:
        def jd_callback(state, identifier):
            solver.davidson.default_print(state, identifier, output)
        kstr = " " if kind == "any" else " " + kind + " "
        print("Starting " + matrix.method.name + " " + kstr
              + "Jacobi-Davidson ...", file=output)

    jdres = jacobi_davidson(matrix, guesses, n_ep=n_states,
                            conv_tol=conv_tol, callback=jd_callback,
                            explicit_symmetrisation=explicit_symmetrisation,
                            **solverargs)
    exstates = ExcitedStates(jdres)
    exstates.kind = kind
    exstates.spin_change = spin_change
    return exstates


def estimate_n_guesses(matrix, n_states, singles_only=True):
    """
    Implementation of a basic heuristic to find a good number of guess
    vectors to be searched for using the find_guesses function.
    Internal function called from run_adc.

    matrix             ADC matrix
    n_states           Number of states to be computed
    singles_only       Try to stay withing the singles excitation space
                       with the number of guess vectors.
    """
    # Try to use at least 4 or twice the number of states
    # to be computed as guesses
    n_guesses = max(4, 2 * n_states)

    if singles_only:
        # Compute the maximal number of sensible singles block guesses.
        # This is roughly the number of occupied alpha orbitals
        # times the number of virtual alpha orbitals
        #
        # If the system is core valence separated, then only the
        # core electrons count as "occupied".
        mospaces = matrix.mospaces
        sp_occ = "o2" if matrix.is_core_valence_separated else "o1"
        n_virt_a = mospaces.n_orbs_alpha("v1")
        n_occ_a = mospaces.n_orbs_alpha(sp_occ)
        n_guesses = min(n_guesses, n_occ_a * n_virt_a)

    # Adjust if we overshoot the maximal number of sensible singles block
    # guesses, but make sure we get at least n_states guesses
    return max(n_states, n_guesses)


def find_guesses(guessfctn, matrix, n_guesses, n_guesses_doubles=None):
    """
    Use the provided guess function to find a particular number of guesses
    in the passed ADC matrix. If n_guesses_doubles is not None, this is
    number is always adhered to. Otherwise the number of doubles guesses
    is adjusted to fill up whatever the singles guesses cannot provide
    to reach n_guesses.
    Internal function called from run_adc.
    """
    if n_guesses_doubles is not None and n_guesses_doubles > 0 \
       and "d" not in matrix.blocks:
        raise ValueError("n_guesses_doubles > 0 is only sensible if the ADC "
                         "method has a doubles block (i.e. it is *not* ADC(0), "
                         "ADC(1) or a variant thereof.")

    # Determine number of singles guesses to request
    n_guess_singles = n_guesses
    if n_guesses_doubles is not None:
        n_guess_singles = n_guesses - n_guesses_doubles
    singles_guesses = guessfctn(matrix, n_guess_singles, block="s")

    if "d" in matrix.blocks:
        # Determine number of doubles guesses to request if not
        # explicitly specified
        if n_guesses_doubles is None:
            n_guesses_doubles = n_guesses - len(singles_guesses)
        doubles_guesses = guessfctn(matrix, n_guesses_doubles, block="d")
    else:
        doubles_guesses = []

    total_guesses = singles_guesses + doubles_guesses
    if len(total_guesses) < n_guesses:
        raise RuntimeError("Less guesses found than requested: {} found, "
                           "{} requested".format(len(total_guesses), n_guesses))
    return total_guesses
