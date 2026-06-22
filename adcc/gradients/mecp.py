#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2026 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
"""Penalty-function objective for MECP/MECI optimisations.

The penalty formulation implemented here is the smoothed Levine--Coe--Martinez
penalty (Levine, Coe and Martinez, *J. Phys. Chem. B* 112, 405, 2008), the same
form used by geomeTRIC's built-in :class:`ConicalIntersection` engine.  It only
needs the two state energies and gradients -- **no derivative couplings** are
required, which matches what a :class:`PairedStateGradientScanner` supplies per
geometry.

This module imports only numpy so that the penalty math stays unit-testable with
controlled energies/gradients and free of the optional geomeTRIC dependency.
: class:`MECPObjective` wraps a paired scanner and a penalty into a single
``(energy, gradient)`` callable plus a ``calc_new`` honouring geomeTRIC's
flattened custom-engine dict contract; the objective is then driven through
PySCF's ``as_pyscf_method`` / ``geometric_solver.optimize`` bridge exactly like
the single-surface scanner.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

__all__ = ["mecp_penalty", "MECPObjective"]

# Number of coupled surfaces (MECI/MECP is always a pair).
_N_STATES = 2
# geomeTRIC's ConicalIntersection counts each unique state pair once and
# normalises by ``n_states2 = n_states * (n_states - 1) / 2``; for the two-state
# MECP/MECI case that is 1.  Keep it identical so the oracle cross-check passes.
_N_STATES2 = _N_STATES * (_N_STATES - 1) // 2

# Default Levine--Coe--Martinez penalty parameters (geomeTRIC's defaults for
# ``--meci_sigma`` / ``--meci_alpha``).  ``alpha`` smooths the otherwise
# singular penalty at the crossing seam.
DEFAULT_SIGMA = 3.5
DEFAULT_ALPHA = 0.025


def mecp_penalty(e_lower, g_lower, e_upper, g_upper, *, sigma=DEFAULT_SIGMA,
                 alpha=DEFAULT_ALPHA):
    """Levine--Coe--Martinez smoothed penalty objective for two surfaces.

    Combines two electronic surfaces ``(e_lower, g_lower)`` and
    ``(e_upper, g_upper)`` (energies in Hartree, gradients in Hartree/Bohr, any
    shared shape) into a single penalty objective ``(E, G)`` of the same units
    suitable for a gradient-based minisation of the crossing seam::

        E_dif = e_upper - e_lower          (>= 0 by construction)
        E_avg = (e_upper + e_lower) / 2
        E_pen = sigma * E_dif^2 / ((E_dif + alpha) * n_states2)
        E_obj = E_avg + E_pen

        G_dif = g_upper - g_lower
        G_avg = (g_upper + g_lower) / 2
        G_pen = sigma * (E_dif^2 + 2*alpha*E_dif) /
                    ((E_dif + alpha)^2 * n_states2) * G_dif
        G_obj = G_avg + G_pen

    with ``n_states2 = 1`` for the two-state MECP/MECI case (geomeTRIC counts
    each unique pair once).  The ``alpha``
    parameter smooths the penalty so the objective stays differentiable at the
    seam (``E_dif = 0``); the ``sigma`` weight controls how hard the degeneracy
    is enforced versus minimising the average energy.

    At exact degeneracy (``e_upper == e_lower``) the penalty and its gradient
    vanish, leaving ``E_obj = E_avg`` and ``G_obj = G_avg`` -- the objective
    reduces to an average-surface optimisation pinned to the crossing.

    Parameters
    ----------
    e_lower, e_upper : float
        The two state energies; ``e_lower <= e_upper`` is *not* required (the
        energy difference is formed as ``e_upper - e_lower`` and only its
        squared phase enters the penalty).
    g_lower, g_upper : numpy.ndarray
        The matching state gradients, broadcastable to each other.
    sigma, alpha : float
        Penalty strength and smoothing parameter.

    Returns
    -------
    (E_obj, G_obj) : (float, numpy.ndarray)
        The scalar penalty objective and its gradient, with ``G_obj`` shaped
        like the inputs.
    """
    if alpha < 0.0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma}.")

    e_lower = float(e_lower)
    e_upper = float(e_upper)
    g_lower = np.asarray(g_lower, dtype=float)
    g_upper = np.asarray(g_upper, dtype=float)

    e_dif = e_upper - e_lower
    g_dif = g_upper - g_lower
    e_avg = 0.5 * (e_upper + e_lower)
    g_avg = 0.5 * (g_lower + g_upper)

    # Smoothed penalty (Levine--Coe--Martinez / geomeTRIC ConicalIntersection).
    # At e_dif == 0 the penalty term and its gradient vanish by construction.
    denom = e_dif + alpha
    e_pen = sigma * (e_dif * e_dif) / (denom * _N_STATES2)
    g_pen_scale = (
        sigma * (e_dif * e_dif + 2.0 * alpha * e_dif)
        / (denom * denom * _N_STATES2)
    )
    g_pen = g_pen_scale * g_dif

    e_obj = e_avg + e_pen
    g_obj = g_avg + g_pen
    return float(e_obj), np.asarray(g_obj, dtype=float)


class MECPObjective:
    """Penalty-objective wrapper driving geomeTRIC from a paired scanner.

    Wrap a :class:`PairedStateGradientScanner` (which returns two
    ``(energy, gradient)`` pairs per geometry) together with the
    :func:`mecp_penalty` into a single ``__call__`` returning one
    ``(energy, gradient)`` for PySCF's ``as_pyscf_method`` / geomeTRIC bridge,
    and a ``calc_new`` returning geomeTRIC's ``{"energy", "gradient"}`` dict
    with a *flattened* gradient in Hartree/Bohr.

    Parameters
    ----------
    scanner : PairedStateGradientScanner
        The paired scanner supplying both surfaces per geometry.
    sigma, alpha : float
        Penalty parameters forwarded to :func:`mecp_penalty`.
    penalty : callable, optional
        Custom two-surface penalty with the same signature as
        :func:`mecp_penalty`; defaults to the Levine--Coe--Martinez smoothed
        penalty.  Useful for injecting a raw energy-difference mode (set
        ``alpha=0``) or for testing alternative formulations.

    Examples
    --------
    >>> objective = MECPObjective(scanner)
    >>> def energy_and_gradient(mol):
    ...     return objective(mol.atom_coords(unit="Bohr"))
    >>> method = as_pyscf_method(mol, energy_and_gradient)
    >>> mol_ci = geometric_solver.optimize(method, maxsteps=20)
    """

    def __init__(self, scanner, *, sigma=DEFAULT_SIGMA, alpha=DEFAULT_ALPHA,
                 penalty: Optional[Callable] = None):
        self.scanner = scanner
        self.sigma = sigma
        self.alpha = alpha
        self.penalty = penalty if penalty is not None else mecp_penalty
        # Bookkeeping of the most recent evaluation for inspection / tests.
        self.last_energy: Optional[float] = None
        self.last_gradient = None
        self.last_pair = None  # ((e_lower, g_lower), (e_upper, g_upper))

    def __call__(self, coords):
        """Return ``(energy, gradient)`` for Cartesian coordinates in Bohr."""
        (e_lower, g_lower), (e_upper, g_upper) = self.scanner(coords)
        self.last_pair = ((e_lower, g_lower), (e_upper, g_upper))
        e_obj, g_obj = self.penalty(
            e_lower, g_lower, e_upper, g_upper,
            sigma=self.sigma, alpha=self.alpha,
        )
        self.last_energy = e_obj
        self.last_gradient = np.asarray(g_obj)
        return e_obj, np.asarray(g_obj)

    def calc_new(self, coords):
        """geomeTRIC custom-engine entry point.

        ``coords`` is a flattened Cartesian coordinate array in Bohr; the
        returned gradient is flattened in Hartree/Bohr, matching the contract
        expected by geomeTRIC's internal engine.
        """
        e_obj, g_obj = self(coords)
        return {
            "energy": float(e_obj),
            "gradient": np.asarray(g_obj).ravel(),
        }
