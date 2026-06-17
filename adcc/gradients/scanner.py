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

The scanner owns the per-geometry PySCF -> adcc -> gradient loop.  It is
intentionally PySCF-only for now, because the production direct nuclear-gradient
path is PySCF-only as well.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
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
    n_states: Optional[int] = None
    kind: str = "any"
    n_singlets: Optional[int] = None
    n_triplets: Optional[int] = None
    n_spin_flip: Optional[int] = None
    core_orbitals: Any = None
    frozen_core: Any = None
    frozen_virtual: Any = None
    conv_tol: Optional[float] = None
    environment: Any = None
    solverargs: dict[str, Any] = field(default_factory=dict)

    def run_adc_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments forwarded to :func:`adcc.run_adc`."""
        ret = {
            "method": self.method,
            "n_states": self.n_states,
            "kind": self.kind,
            "n_singlets": self.n_singlets,
            "n_triplets": self.n_triplets,
            "n_spin_flip": self.n_spin_flip,
            "core_orbitals": self.core_orbitals,
            "frozen_core": self.frozen_core,
            "frozen_virtual": self.frozen_virtual,
            "conv_tol": self.conv_tol,
            "environment": self.environment,
            "output": None,
        }
        ret.update(self.solverargs)
        return {key: value for key, value in ret.items() if value is not None}


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


@dataclass
class _ScfSettings:
    conv_tol: float
    conv_tol_grad: Optional[float]
    max_cycle: int
    verbose: int
    attributes: dict[str, Any]


class NuclearGradientScanner:
    """Callable scanner returning adcc energies and nuclear gradients.

    Parameters are usually created through :func:`nuclear_gradient_scanner`.
    The scanner accepts Cartesian coordinates in Bohr and returns an energy in
    Hartree and a gradient in Hartree/Bohr.
    """

    def __init__(self, mol_template=None, *, scf_template=None,
                 scf_factory: Optional[Callable[[Any], Any]] = None,
                 target: GroundStateTarget | ExcitedStateTarget | str | None = None,
                 method: Optional[str] = None, state_index: int = 0,
                 mp_level: int = 2, n_states: Optional[int] = None,
                 kind: str = "any", n_singlets: Optional[int] = None,
                 n_triplets: Optional[int] = None,
                 n_spin_flip: Optional[int] = None, core_orbitals: Any = None,
                 frozen_core: Any = None, frozen_virtual: Any = None,
                 adc_conv_tol: Optional[float] = None, environment: Any = None,
                 scf_type: Optional[str] = None, scf_conv_tol: Optional[float] = None,
                 scf_conv_tol_grad: Optional[float] = None,
                 scf_max_cycle: Optional[int] = None, symmetry: Any = None,
                 verbose: Optional[int] = None, follow: str = "overlap",
                 tracking_min_score: float = 0.0, tracking_min_gap: float = 0.0,
                 gradient_kwargs: Optional[dict[str, Any]] = None,
                 adc_solverargs: Optional[dict[str, Any]] = None):
        self._pyscf = _import_pyscf()
        self.scf_factory = scf_factory
        self.scf_type = scf_type
        self._requested_symmetry = symmetry
        self.follow = follow
        self.tracking_min_score = tracking_min_score
        self.tracking_min_gap = tracking_min_gap
        self.gradient_kwargs = dict(gradient_kwargs or {})
        self.last_tracking: Optional[TrackingResult] = None

        if scf_template is None and _is_pyscf_scf(self._pyscf, mol_template):
            scf_template = mol_template
            mol_template = scf_template.mol
        if scf_template is not None and mol_template is None:
            mol_template = scf_template.mol
        if mol_template is None:
            raise ValueError("mol_template or scf_template needs to be provided.")

        self._init_mol_template(mol_template)
        self.scf_class = type(scf_template) if scf_template is not None else None
        self.scf_settings = self._capture_scf_settings(
            scf_template, scf_conv_tol, scf_conv_tol_grad, scf_max_cycle, verbose
        )
        self.target = self._normalise_target(
            target, method, state_index, mp_level, n_states, kind, n_singlets,
            n_triplets, n_spin_flip, core_orbitals, frozen_core, frozen_virtual,
            adc_conv_tol, environment, adc_solverargs or {}
        )

        self.previous_scf = None
        self.previous_dm = None
        self.previous_states = None
        self.previous_excitation: Optional[Excitation] = None
        self.previous_descriptor: Optional[_TrackingDescriptor] = None
        self.last_scf = None
        self.last_states = None
        self.last_excitation = None
        self.last_gradient = None

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
        self.previous_dm = scfres.make_rdm1()
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

    def _init_mol_template(self, mol):
        self.atom_symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
        self.basis = mol.basis
        self.ecp = getattr(mol, "ecp", None)
        self.charge = mol.charge
        self.spin = mol.spin
        self.cart = bool(getattr(mol, "cart", False))
        self.symmetry = (getattr(mol, "symmetry", False)
                         if self._requested_symmetry is None
                         else self._requested_symmetry)
        self.output = getattr(mol, "output", None)
        self.max_memory = getattr(mol, "max_memory", None)
        self.initial_coords = np.asarray(mol.atom_coords(unit="Bohr"), dtype=float)

    def _capture_scf_settings(self, scf_template, conv_tol, conv_tol_grad,
                              max_cycle, verbose):
        attributes = {}
        if scf_template is not None:
            for attr in [
                "diis_space", "diis_start_cycle", "damp", "level_shift",
                "direct_scf", "init_guess", "max_memory",
            ]:
                if hasattr(scf_template, attr):
                    attributes[attr] = getattr(scf_template, attr)
            default_conv_tol = scf_template.conv_tol
            default_conv_tol_grad = getattr(scf_template, "conv_tol_grad", None)
            default_max_cycle = scf_template.max_cycle
            default_verbose = scf_template.verbose
        else:
            default_conv_tol = 1e-11
            default_conv_tol_grad = 1e-9
            default_max_cycle = 150
            default_verbose = 0
        return _ScfSettings(
            conv_tol=default_conv_tol if conv_tol is None else conv_tol,
            conv_tol_grad=(default_conv_tol_grad if conv_tol_grad is None
                           else conv_tol_grad),
            max_cycle=default_max_cycle if max_cycle is None else max_cycle,
            verbose=default_verbose if verbose is None else verbose,
            attributes=attributes,
        )

    def _normalise_target(self, target, method, state_index, mp_level, n_states,
                          kind, n_singlets, n_triplets, n_spin_flip,
                          core_orbitals, frozen_core, frozen_virtual,
                          adc_conv_tol, environment, solverargs):
        if isinstance(target, (GroundStateTarget, ExcitedStateTarget)):
            return target
        if isinstance(target, str):
            if target.lower().startswith("mp"):
                try:
                    level = int(target.lower().removeprefix("mp"))
                except ValueError:
                    level = mp_level
                return GroundStateTarget(level=level)
            method = target
        if method is None or method.lower().startswith("mp"):
            if method and method.lower().startswith("mp"):
                try:
                    mp_level = int(method.lower().removeprefix("mp"))
                except ValueError:
                    pass
            return GroundStateTarget(level=mp_level)
        return ExcitedStateTarget(
            method=method, state_index=state_index, n_states=n_states, kind=kind,
            n_singlets=n_singlets, n_triplets=n_triplets,
            n_spin_flip=n_spin_flip, core_orbitals=core_orbitals,
            frozen_core=frozen_core, frozen_virtual=frozen_virtual,
            conv_tol=adc_conv_tol, environment=environment,
            solverargs=dict(solverargs),
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

    def _build_mol(self, coords):
        gto = self._pyscf.gto
        atom = [(sym, tuple(map(float, xyz)))
                for sym, xyz in zip(self.atom_symbols, coords)]
        kwargs = {
            "atom": atom,
            "basis": self.basis,
            "unit": "Bohr",
            "spin": self.spin,
            "charge": self.charge,
            "symmetry": self.symmetry,
            "parse_arg": False,
            "dump_input": False,
            "verbose": self.scf_settings.verbose,
            "cart": self.cart,
        }
        if self.ecp:
            kwargs["ecp"] = self.ecp
        mol = gto.M(**kwargs)
        if self.max_memory is not None:
            mol.max_memory = self.max_memory
        return mol

    def _build_scf(self, mol):
        scf = self._pyscf.scf
        if self.scf_factory is not None:
            mf = self.scf_factory(mol)
        elif self.scf_class is not None:
            mf = self.scf_class(mol)
        elif self.scf_type is not None:
            scf_type = self.scf_type.upper()
            if not hasattr(scf, scf_type):
                raise ValueError(f"Unknown PySCF SCF type '{self.scf_type}'.")
            mf = getattr(scf, scf_type)(mol)
        else:
            mf = scf.HF(mol)
        mf.conv_tol = self.scf_settings.conv_tol
        if self.scf_settings.conv_tol_grad is not None:
            mf.conv_tol_grad = self.scf_settings.conv_tol_grad
        mf.max_cycle = self.scf_settings.max_cycle
        mf.verbose = self.scf_settings.verbose
        for attr, value in self.scf_settings.attributes.items():
            if hasattr(mf, attr):
                setattr(mf, attr, value)
        return mf

    def _run_scf(self, coords):
        mol = self._build_mol(coords)
        mf = self._build_scf(mol)
        if self.previous_dm is None:
            mf.kernel()
        else:
            mf.kernel(dm0=self.previous_dm)
        if not mf.converged:
            raise RuntimeError("PySCF SCF did not converge at scanner geometry.")
        return mf

    def _build_target(self, scfres):
        if isinstance(self.target, GroundStateTarget):
            from adcc import LazyMp, ReferenceState
            return LazyMp(ReferenceState(scfres))
        from adcc import run_adc
        states = run_adc(scfres, **self.target.run_adc_kwargs())
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

    See :class:`NuclearGradientScanner` for the coordinate and unit contract.
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


def _import_pyscf():
    try:
        from pyscf import gto, scf
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ModuleNotFoundError(
            "nuclear_gradient_scanner currently requires PySCF. Install PySCF "
            "or pass a PySCF template object in an environment where PySCF is "
            "available."
        ) from exc
    return _PyscfModules(gto=gto, scf=scf)


@dataclass(frozen=True)
class _PyscfModules:
    gto: Any
    scf: Any


def _is_pyscf_scf(pyscf, obj) -> bool:
    return isinstance(obj, pyscf.scf.hf.SCF)
