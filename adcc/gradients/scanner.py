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
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
"""Geometry-optimisation scanner for adcc nuclear gradients.

The scanner owns the per-geometry PySCF -> adcc -> gradient loop.  Users pass a
configured PySCF SCF object, whose settings define how SCF is performed at every
geometry.  adcc-specific keyword arguments are forwarded unchanged to
:func:`adcc.run_adc` for excited-state targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
import warnings

import numpy as np

from adcc.Excitation import Excitation
from adcc.LazyMp import LazyMp


@dataclass(frozen=True)
class GroundStateTarget:
    """Ground-state MP target for a nuclear-gradient scanner."""

    level: int = 2


@dataclass(frozen=True)
class ExcitedStateTarget:
    """Excited-state ADC target for a nuclear-gradient scanner."""

    method: str
    state_index: int = 0
    run_adc_kwargs: dict[str, Any] = field(default_factory=dict)

    def kwargs(self) -> dict[str, Any]:
        """Return keyword arguments forwarded to :func:`adcc.run_adc`."""
        ret = dict(self.run_adc_kwargs)
        ret["method"] = self.method
        ret.setdefault("output", None)
        return ret


@dataclass
class TrackingResult:
    """Diagnostic information for the most recent root-tracking decision."""

    index: int
    scores: np.ndarray


@dataclass
class _TrackingDescriptor:
    mol: Any
    transition_dm: Optional[np.ndarray]
    state_diffdm: Optional[np.ndarray]


class NuclearGradientScanner:
    """Callable scanner returning adcc energies and nuclear gradients.

    Parameters are usually created through :func:`nuclear_gradient_scanner`.
    ``scfres`` is a configured PySCF SCF object.  Its molecule and SCF settings
    are used as the template for every scanner call; only the nuclear geometry
    changes.  Coordinates are Cartesian in Bohr, energies are Hartree and
    gradients are Hartree/Bohr.
    """

    def __init__(self, scfres, *,
                 target: GroundStateTarget | ExcitedStateTarget | str | None = None,
                 method: Optional[str] = None, state_index: int = 0,
                 mp_level: int = 2, follow: str = "overlap",
                 tracking_min_score: float = 0.0, tracking_min_gap: float = 0.0,
                 gradient_kwargs: Optional[dict[str, Any]] = None,
                 **run_adc_kwargs):
        self._pyscf = _import_pyscf()
        if not _is_pyscf_scf(self._pyscf, scfres):
            raise TypeError(
                "nuclear_gradient_scanner expects a PySCF SCF object, e.g. "
                "scf.RHF(mol), as its first argument."
            )
        if not hasattr(scfres, "as_scanner"):
            raise TypeError("The provided PySCF SCF object has no as_scanner method.")

        self.scf_template = scfres
        self.scf_scanner = scfres.as_scanner()
        self.base_mol = scfres.mol.copy()
        self.atom_symbols = [self.base_mol.atom_symbol(i)
                             for i in range(self.base_mol.natm)]
        self.initial_coords = np.asarray(
            self.base_mol.atom_coords(unit="Bohr"), dtype=float
        )

        self.follow = follow
        self.tracking_min_score = tracking_min_score
        self.tracking_min_gap = tracking_min_gap
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.target = self._normalise_target(
            target, method, state_index, mp_level, run_adc_kwargs
        )

        self.previous_scf = None
        self.previous_states = None
        self.previous_excitation: Optional[Excitation] = None
        self.previous_descriptor: Optional[_TrackingDescriptor] = None
        self.last_scf = None
        self.last_states = None
        self.last_excitation = None
        self.last_gradient = None
        self.last_tracking: Optional[TrackingResult] = None

    @property
    def natoms(self) -> int:
        return len(self.atom_symbols)

    def __call__(self, coords):
        """Return ``(energy, gradient)`` for Cartesian coordinates in Bohr."""
        coords = self._coords_array(coords)
        scfres = self._run_scf(coords)
        target = self._build_target(scfres)

        from adcc.gradients import nuclear_gradient
        grad = nuclear_gradient(target, **self.gradient_kwargs)
        if isinstance(target, LazyMp):
            energy = target.energy(self.target.level)
        else:
            energy = target.total_energy

        self.last_scf = scfres
        self.last_gradient = grad
        self.previous_scf = scfres
        if isinstance(target, Excitation):
            self.previous_states = self.last_states
            self.previous_excitation = target
            self.previous_descriptor = self._descriptor(target, scfres.mol)
        return float(energy), np.asarray(grad.total)

    def calc_new(self, coords):
        """geomeTRIC custom-engine entry point.

        ``coords`` is a flattened Cartesian coordinate array in Bohr.  The
        returned gradient is flattened in Hartree/Bohr.
        """
        energy, gradient = self(coords)
        return {"energy": energy, "gradient": np.asarray(gradient).ravel()}

    def _normalise_target(self, target, method, state_index, mp_level,
                          run_adc_kwargs):
        if isinstance(target, (GroundStateTarget, ExcitedStateTarget)):
            return target
        if isinstance(target, str):
            if target.lower().startswith("mp"):
                return GroundStateTarget(level=_mp_level(target, mp_level))
            method = target
        if method is None or method.lower().startswith("mp"):
            if method is not None:
                mp_level = _mp_level(method, mp_level)
            if run_adc_kwargs:
                warnings.warn(
                    "Ignoring run_adc keyword arguments for ground-state MP "
                    "scanner target.", RuntimeWarning,
                )
            return GroundStateTarget(level=mp_level)
        return ExcitedStateTarget(
            method=method, state_index=state_index,
            run_adc_kwargs=dict(run_adc_kwargs),
        )

    def _coords_array(self, coords):
        coords = np.asarray(coords, dtype=float)
        if coords.shape == (3 * self.natoms,):
            coords = coords.reshape(self.natoms, 3)
        if coords.shape != (self.natoms, 3):
            raise ValueError(
                f"Expected coordinates with shape {(self.natoms, 3)} or "
                f"{(3 * self.natoms,)}, got {coords.shape}."
            )
        return coords

    def _mol_at(self, coords):
        # PySCF's set_geom_ preserves the original Mole settings (basis, charge,
        # spin, symmetry setting, unit conventions, etc.) and changes only atom
        # coordinates.  This keeps the scanner interface anchored on the user's
        # configured PySCF object rather than mirroring PySCF keyword arguments.
        return self.base_mol.set_geom_(coords, unit="Bohr", inplace=False)

    def _run_scf(self, coords):
        mol = self._mol_at(coords)
        self.scf_scanner(mol)
        if not self.scf_scanner.converged:
            raise RuntimeError("PySCF SCF did not converge at scanner geometry.")
        # Snapshot the current scanner state for adcc.  The scanner itself is
        # reused on the next geometry to keep PySCF's orbital/density continuity.
        return self.scf_scanner.undo_scanner()

    def _build_target(self, scfres):
        if isinstance(self.target, GroundStateTarget):
            from adcc import LazyMp, ReferenceState
            return LazyMp(ReferenceState(scfres))
        from adcc import run_adc
        states = run_adc(scfres, **self.target.kwargs())
        self.last_states = states
        excitation = self._select_excitation(states, scfres.mol)
        self.last_excitation = excitation
        return excitation

    def _select_excitation(self, states, mol):
        excitations = states.excitations
        if not excitations:
            raise RuntimeError("ADC calculation did not return any excited states.")
        if self.follow == "index" or self.previous_descriptor is None:
            index = self.target.state_index
            if index >= len(excitations) or index < -len(excitations):
                raise ValueError(
                    f"state_index {index} is out of range for "
                    f"{len(excitations)} computed states."
                )
            self.last_tracking = None
            return excitations[index]
        if self.follow != "overlap":
            raise ValueError("follow needs to be 'overlap' or 'index'.")
        scores = np.array([
            self._tracking_score(self.previous_descriptor,
                                 self._descriptor(excitation, mol), mol)
            for excitation in excitations
        ])
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        self.last_tracking = TrackingResult(best, scores)
        if scores[best] < self.tracking_min_score:
            raise RuntimeError(
                "Root tracking failed: best state-character overlap "
                f"{scores[best]:.6g} is below threshold "
                f"{self.tracking_min_score:.6g}."
            )
        if len(order) > 1 and scores[best] - scores[int(order[1])] < self.tracking_min_gap:
            raise RuntimeError(
                "Root tracking is ambiguous: best and second-best overlaps "
                f"differ by {scores[best] - scores[int(order[1])]:.6g}."
            )
        return excitations[best]

    def _descriptor(self, excitation, mol):
        return _TrackingDescriptor(
            mol=mol,
            transition_dm=_safe_ao_ndarray(excitation, "transition_dm_ao"),
            state_diffdm=_safe_ao_ndarray(excitation, "state_diffdm_ao"),
        )

    def _tracking_score(self, previous, current, current_mol):
        s_cross = self._pyscf.gto.intor_cross(
            "int1e_ovlp", previous.mol, current_mol
        )
        score = 0.0
        weight = 0
        for old, new, use_abs in [
            (previous.transition_dm, current.transition_dm, True),
            (previous.state_diffdm, current.state_diffdm, False),
        ]:
            if old is None or new is None:
                continue
            channel = density_overlap_score(
                old, new, s_cross,
                s_old=previous.mol.intor_symmetric("int1e_ovlp"),
                s_new=current_mol.intor_symmetric("int1e_ovlp"),
                use_abs=use_abs,
            )
            if np.isfinite(channel):
                score += channel
                weight += 1
        if weight == 0:
            raise RuntimeError(
                "Root tracking requested, but neither transition_dm_ao nor "
                "state_diffdm_ao could be computed."
            )
        return score / weight


def nuclear_gradient_scanner(*args, **kwargs) -> NuclearGradientScanner:
    """Create a PySCF/adcc nuclear-gradient scanner.

    The first argument is a configured PySCF SCF object.  Keyword arguments not
    consumed by the scanner are forwarded unchanged to :func:`adcc.run_adc` for
    excited-state targets.
    """
    return NuclearGradientScanner(*args, **kwargs)


def density_overlap_score(old_dm, new_dm, s_cross, *, s_old=None, s_new=None,
                          use_abs=False) -> float:
    """Cosine-like overlap of AO-basis density matrices at two geometries.

    ``s_cross`` is the rectangular AO overlap between the old and new PySCF
    molecules, i.e. ``pyscf.gto.intor_cross('int1e_ovlp', old_mol, new_mol)``.
    ``s_old`` and ``s_new`` are the AO overlap matrices within each geometry.
    They default to identity matrices for simple unit tests.
    """
    old_dm = np.asarray(old_dm)
    new_dm = np.asarray(new_dm)
    s_cross = np.asarray(s_cross)
    if s_old is None:
        s_old = np.eye(old_dm.shape[0])
    if s_new is None:
        s_new = np.eye(new_dm.shape[0])
    s_old = np.asarray(s_old)
    s_new = np.asarray(s_new)
    numerator = np.einsum("pq,pr,rs,qs->", old_dm, s_cross, new_dm, s_cross)
    old_norm = np.einsum("pq,pr,rs,qs->", old_dm, s_old, old_dm, s_old)
    new_norm = np.einsum("pq,pr,rs,qs->", new_dm, s_new, new_dm, s_new)
    denom = np.sqrt(abs(old_norm) * abs(new_norm))
    if denom == 0.0:
        return np.nan
    score = numerator / denom
    if use_abs:
        score = abs(score)
    return float(score)


def _safe_ao_ndarray(excitation, attr):
    try:
        value = getattr(excitation, attr)
        return value.to_ndarray()
    except Exception as exc:  # pragma: no cover - exercised by optional paths
        warnings.warn(
            f"Could not compute {attr} for root tracking: {exc}",
            RuntimeWarning,
        )
        return None


def _mp_level(method, default):
    try:
        return int(method.lower().removeprefix("mp"))
    except ValueError:
        return default


def _import_pyscf():
    try:
        from pyscf import gto, scf
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ModuleNotFoundError(
            "nuclear_gradient_scanner currently requires PySCF. Install PySCF "
            "and pass a configured PySCF SCF object."
        ) from exc
    return _PyscfModules(gto=gto, scf=scf)


@dataclass(frozen=True)
class _PyscfModules:
    gto: Any
    scf: Any


def _is_pyscf_scf(pyscf, obj) -> bool:
    return isinstance(obj, pyscf.scf.hf.SCF)
