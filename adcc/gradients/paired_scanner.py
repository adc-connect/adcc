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
"""Paired-surface nuclear-gradient scanner for MECP/MECI optimisations.

: class:`PairedStateGradientScanner` evaluates *two* electronic surfaces at a
single geometry from one SCF + one ADC (or, for MECP, one ``LazyMp`` plus one
ADC) and returns both ``(energy, gradient)`` pairs.  It reuses the
single-surface : class:`NuclearGradientScanner` plumbing (SCF lifecycle, AO
density-overlap root tracking, coordinate handling) and adds a **joint distinct
root selection** with a distinctness guard so the two tracked roots never
collapse onto the same candidate state.

A pure penalty objective that combines the two surfaces into one
``(energy, gradient)`` for a geomeTRIC-driven optimisation lives in a separate
module; this file owns only the paired scanner and its target dataclasses.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from adcc.LazyMp import LazyMp
from adcc.Excitation import Excitation

from adcc.gradients.scanner import (
    GroundStateTarget,
    NuclearGradientScanner,
    TrackingResult,
    _TrackingDescriptor,
    _check_ground_state_level,
    _mp_level,
    density_overlap_score,
)


__all__ = [
    "PairedExcitedStateTarget",
    "PairedGroundExcitedStateTarget",
    "PairedStateGradientScanner",
    "density_overlap_score",
]


# ---------------------------------------------------------------------------
# Paired target dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PairedExcitedStateTarget:
    """Two excited-state ADC roots evaluated from a single ``run_adc`` call.

    Used for MECI scans where both surfaces come from the same excited-state
    manifold.  ``state_indices`` are the *positional* seeds for the two tracked
    roots; overlap tracking (when enabled) follows their state character across
    geometries while keeping the two roots distinct.
    """

    method: str
    state_indices: tuple[int, int]
    run_adc_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if len(self.state_indices) != 2:
            raise ValueError(
                "PairedExcitedStateTarget.state_indices must contain exactly two "
                f"indices, got {self.state_indices!r}."
            )
        # Immutable tuple of plain ints.
        object.__setattr__(
            self, "state_indices",
            (int(self.state_indices[0]), int(self.state_indices[1])),
        )

    def kwargs(self) -> dict[str, Any]:
        """Return keyword arguments forwarded to : func:`adcc.run_adc`."""
        ret = dict(self.run_adc_kwargs)
        ret["method"] = self.method
        return ret


@dataclass(frozen=True)
class PairedGroundExcitedStateTarget:
    """One ground-state MP surface plus one excited-state ADC surface (MECP).

    The ground surface is always MP2 (the only level for which adcc provides a
    nuclear gradient) and is evaluated via : class:`adcc.LazyMp` /
    ``LazyMp.energy(2)``.  The excited surface is the single ADC root at
    ``state_index`` of the same SCF reference.  Only the excited root is
    overlap-tracked; the ground side is uniquely defined by the SCF reference.
    """

    method: str
    state_index: int = 0
    run_adc_kwargs: dict[str, Any] = field(default_factory=dict)
    level: int = 2

    def __post_init__(self):
        _check_ground_state_level(self.level)
        object.__setattr__(self, "state_index", int(self.state_index))

    def kwargs(self) -> dict[str, Any]:
        """Return keyword arguments forwarded to : func:`adcc.run_adc`."""
        ret = dict(self.run_adc_kwargs)
        ret["method"] = self.method
        return ret


# ---------------------------------------------------------------------------
# Paired scanner
# ---------------------------------------------------------------------------

class PairedStateGradientScanner(NuclearGradientScanner):
    """Callable scanner returning two ``(energy, gradient)`` surface pairs.

    ``scfres`` is a configured PySCF SCF object; its molecule and SCF settings
    are the template for every scanner call and only the geometry changes.
    Coordinates are Cartesian in Bohr, energies are Hartree and gradients are
    Hartree/Bohr.

    Two normalised target forms are supported:

    * **MECI** -- two excited roots from one ADC solve, via
      ``PairedExcitedStateTarget`` or the ``method=..., states=(i, j)``
      convenience.
    * **MECP** -- one MP2 ground surface plus one excited ADC root, via a
      ``PairedGroundExcitedStateTarget`` or the ``lower="mp2", upper=k``
      convenience (``method`` supplies the excited-state ADC method).

    ``__call__`` returns ``((e_lower, g_lower), (e_upper, g_upper))`` with the
    two surfaces sorted by energy so that ``e_lower <= e_upper``.  Both surfaces
    are also kept in *slot* order on :attr:`last_energies` / :attr:`last_gradients`
    (slot 0 = first tracked root / ground, slot 1 = second tracked root /
    excited) for diagnostics.
    """

    def __init__(self, scfres, *,
                 target: Optional[Any] = None,
                 method: Optional[str] = None,
                 states: Optional[Sequence[int]] = None,
                 state_indices: Optional[Sequence[int]] = None,
                 lower: Optional[Any] = None,
                 upper: Optional[int] = None,
                 mp_level: int = 2,
                 follow: str = "overlap",
                 tracking_min_score: float = 0.0,
                 tracking_min_gap: float = 0.0,
                 gradient_kwargs: Optional[dict[str, Any]] = None,
                 **run_adc_kwargs):
        # Reuse the single-surface plumbing: PySCF import/validation, the SCF
        # scanner lifecycle, coordinate helpers and the gradient kwargs store.
        # The parent normalises a placeholder single-surface target (an MP2
        # ground-state target by default); the paired target is built below and
        # takes precedence over the placeholder on every code path.
        super().__init__(
            scfres,
            follow=follow,
            tracking_min_score=tracking_min_score,
            tracking_min_gap=tracking_min_gap,
            gradient_kwargs=gradient_kwargs,
            mp_level=2,
        )

        self.paired_target = self._normalise_paired_target(
            target, method, states, state_indices, lower, upper, mp_level,
            run_adc_kwargs,
        )

        # Density-overlap tracking is designed for the single-surface scanner
        # (follow one fixed state character across geometries).  For an
        # excited/excited MECI pair the seam is defined by a degeneracy of the
        # energy-ordered two lowest roots, not by fixed state character, so
        # tracking by overlap fights the adiabatic reordering near the seam --
        # once the two roots' densities blur together it can lock a slot onto a
        # higher root and the optimisation diverges.  Use follow="index" (the
        # scanner energy-sorts the positional roots and the penalty drives the
        # gap), which is the contract geomeTRIC's ConicalIntersection engine
        # expects from its sub-engines.
        if self.follow == "overlap" and isinstance(
                self.paired_target, PairedExcitedStateTarget):
            warnings.warn(
                "follow='overlap' is not recommended for a MECI pair "
                "(excited/excited): near the crossing seam the two tracked "
                "roots' densities become near-degenerate and overlap tracking "
                "can flip a slot onto a higher root, diverging the penalty "
                "optimisation. Use follow='index' (the two lowest adiabatic "
                "roots, energy-sorted) for MECI optimisations; this is also "
                "what geomeTRIC's ConicalIntersection engine expects from its "
                "sub-engines.",
                UserWarning,
                stacklevel=2,
            )

        # Plural tracking state.  For MECP slot 0 (ground) is never tracked:
        # its descriptor / index stay ``None`` forever.
        self.previous_descriptors: tuple[Optional[_TrackingDescriptor],
                                         Optional[_TrackingDescriptor]] = (
            None, None)
        self.previous_indices: tuple[Optional[int], Optional[int]] = (None, None)

        # Last-call diagnostics (slot order).
        self.last_scf = None
        self.previous_scf = None
        self.last_states = None
        self.last_targets: tuple[Optional[Any], Optional[Any]] = (None, None)
        self.last_excitations: tuple[Optional[Excitation],
                                     Optional[Excitation]] = (None, None)
        self.last_gradients: tuple[np.ndarray, np.ndarray] = (None, None)
        self.last_gradient_results: tuple[Any, Any] = (None, None)
        self.last_energies: tuple[float, float] = (None, None)
        self.last_trackings: tuple[Optional[TrackingResult],
                                   Optional[TrackingResult]] = (None, None)
        # Energy-sorted insertion order of the two surfaces in the last call:
        # ``last_pair_order = (lo_slot, hi_slot)``.
        self.last_pair_order: tuple[int, int] = (0, 1)

    # -- public API --------------------------------------------------------

    def __call__(self, coords):
        """Return ``((e_lower, g_lower), (e_upper, g_upper))`` for ``coords``.

        ``coords`` are Cartesian coordinates in Bohr with shape
        ``(natoms, 3)`` or ``(3 * natoms,)``.  The two surfaces are sorted by
        energy so the returned lower energy is never larger than the upper one.
        """
        coords = self._coords_array(coords)
        scfres = self._run_scf(coords)
        targets = self._build_target(scfres)

        from adcc.gradients import nuclear_gradient
        grad_results = [
            nuclear_gradient(t, **self.gradient_kwargs) for t in targets
        ]
        grads = [np.asarray(g.total) for g in grad_results]
        energies = []
        for t in targets:
            if isinstance(t, LazyMp):
                energies.append(float(t.energy(self.paired_target.level)))
            else:
                energies.append(float(t.total_energy))

        self.last_scf = scfres
        self.previous_scf = scfres
        self.last_targets = tuple(targets)
        self.last_gradient_results = tuple(grad_results)
        self.last_gradients = tuple(grads)
        self.last_energies = tuple(energies)

        # Guard against non-finite surface results *before* the energy sort:
        # np.argsort silently reorders slots on NaN, which would otherwise let a
        # borderline ADC solve hand misleading (lower, upper) pairs -- and a
        # NaN objective -- to geomeTRIC.
        for slot, e in enumerate(energies):
            if not np.isfinite(e) or not np.all(np.isfinite(grads[slot])):
                raise RuntimeError(
                    f"Non-finite surface result on paired slot {slot} "
                    f"(energy={e!r}); cannot drive a penalty optimisation."
                )

        order = np.argsort(energies)
        lo, hi = int(order[0]), int(order[1])
        self.last_pair_order = (lo, hi)
        return ((energies[lo], grads[lo]), (energies[hi], grads[hi]))

    def calc_new(self, coords):
        """geomeTRIC-style entry point returning *both* surface pairs.

        Unlike the single-surface scanner, this returns a dictionary carrying
        the two energies and (flattened) gradients separately rather than a
        single ``energy``/``gradient`` pair.  A pure penalty objective combining
        the two surfaces into one geomeTRIC-bound surface lives in a separate
        module and drives the optimizer; this method is intended for inspection
        and testing.
        """
        (e_lo, g_lo), (e_hi, g_hi) = self(coords)
        g_lo = np.asarray(g_lo)
        g_hi = np.asarray(g_hi)
        return {
            "energies": np.array([e_lo, e_hi]),
            "gradients": np.stack([g_lo, g_hi]),
            "energy_lower": float(e_lo),
            "energy_upper": float(e_hi),
            "gradient_lower": g_lo.ravel(),
            "gradient_upper": g_hi.ravel(),
        }

    # -- target normalisation ---------------------------------------------

    def _normalise_paired_target(self, target, method, states, state_indices,
                                 lower, upper, mp_level, run_adc_kwargs):
        if isinstance(target, PairedExcitedStateTarget):
            return target
        if isinstance(target, PairedGroundExcitedStateTarget):
            # ``__post_init__`` already validated the MP level.
            return target
        if isinstance(target, str):
            method = target
        run_adc_kwargs = dict(run_adc_kwargs)

        states = states if states is not None else state_indices

        # MECP form: lower=<ground MP spec>, upper=<excited index>.
        if lower is not None and upper is not None:
            if isinstance(lower, GroundStateTarget):
                level = lower.level
            elif isinstance(lower, str):
                level = _mp_level(lower, mp_level)
            elif isinstance(lower, int):
                level = lower
            else:
                raise TypeError(
                    "lower must be 'mp2', an int MP level, or a "
                    "GroundStateTarget, got "
                    f"{type(lower).__name__}."
                )
            _check_ground_state_level(level)
            if not isinstance(upper, int):
                raise TypeError(
                    "upper must be an int excited-state index, got "
                    f"{type(upper).__name__}."
                )
            if method is None:
                raise ValueError(
                    "An ADC method is required for the excited side of an "
                    "MECP scanner (pass method=...)."
                )
            return PairedGroundExcitedStateTarget(
                method=method, state_index=upper,
                run_adc_kwargs=dict(run_adc_kwargs), level=level,
            )

        # MECI form: two excited indices from one ADC solve.
        if states is not None:
            if not isinstance(states, (tuple, list)):
                raise TypeError(
                    "states must be a 2-tuple of excited-state indices, got "
                    f"{type(states).__name__}."
                )
            states = tuple(states)
            if len(states) != 2:
                raise ValueError(
                    "states must contain exactly two excited-state indices, "
                    f"got {len(states)}."
                )
            if method is None:
                raise ValueError(
                    "An ADC method is required for a paired excited-state "
                    "scanner (pass method=...)."
                )
            return PairedExcitedStateTarget(
                method=method, state_indices=tuple(states),
                run_adc_kwargs=dict(run_adc_kwargs),
            )

        raise ValueError(
            "PairedStateGradientScanner needs a paired target: pass a "
            "PairedExcitedStateTarget / PairedGroundExcitedStateTarget, or the "
            "convenience forms method=..., states=(i, j) (MECI) / "
            "lower='mp2', upper=k (MECP)."
        )

    # -- per-geometry build -----------------------------------------------

    def _build_target(self, scfres):
        """Run SCF-anchored adcc once and return the two surface targets."""
        from adcc import ReferenceState, run_adc

        if isinstance(self.paired_target, PairedExcitedStateTarget):
            states = run_adc(scfres, **self.paired_target.kwargs())
            self.last_states = states
            chosen, descriptors, trackings = self._select_excitations(
                states, scfres.mol
            )
            self.last_trackings = tuple(trackings)
            self.last_excitations = (
                states.excitations[chosen[0]], states.excitations[chosen[1]],
            )
            self.last_targets = self.last_excitations
            self.previous_descriptors = (
                descriptors[chosen[0]], descriptors[chosen[1]],
            )
            self.previous_indices = (chosen[0], chosen[1])
            return self.last_targets

        # MECP: MP2 ground + single tracked excited root from one SCF + one ADC.
        mp = LazyMp(ReferenceState(scfres))
        states = run_adc(scfres, **self.paired_target.kwargs())
        self.last_states = states
        excitation, tracking, descriptor, idx = self._select_excited_root(
            states, scfres.mol,
            self.previous_descriptors[1], self.previous_indices[1],
            self.paired_target.state_index,
        )
        self.last_trackings = (None, tracking)
        self.last_excitations = (None, excitation)
        self.last_targets = (mp, excitation)
        self.previous_descriptors = (None, descriptor)
        self.previous_indices = (None, idx)
        return self.last_targets

    # -- root selection ----------------------------------------------------

    def _select_excitations(self, states, mol):
        """Jointly select two distinct excited roots for the MECI pair.

        Reuses the single-surface AO density-overlap tracking machinery via
        :meth:`NuclearGradientScanner._tracking_score` and adds a joint
        distinctness guard so the two tracked roots never collapse onto the same
        candidate state.  Returns ``(chosen, descriptors, trackings)`` where
        ``chosen`` is the two selected positional indices (slot order).
        """
        excitations = states.excitations
        if not excitations:
            raise RuntimeError(
                "ADC calculation did not return any excited states."
            )
        if len(excitations) < 2:
            raise RuntimeError(
                "Paired excited-state scanner needs at least two excited "
                f"states, got {len(excitations)}."
            )
        n = len(excitations)
        descriptors = [
            self._descriptor(excitation, mol) for excitation in excitations
        ]

        unavailable: set[str] = set()
        score_vectors: list[Optional[np.ndarray]] = [None, None]
        for slot in range(2):
            prev = self.previous_descriptors[slot]
            if self.follow == "overlap" and prev is not None:
                score_vectors[slot] = np.array([
                    self._tracking_score(prev, descriptors[k], mol, unavailable)
                    for k in range(n)
                ])

        # Slots without a score vector fall back to their positional seed index.
        target_indices = tuple(self.paired_target.state_indices)
        fixed = [None, None]
        for slot in range(2):
            if score_vectors[slot] is None:
                idx = int(target_indices[slot])
                if idx >= n or idx < -n:
                    raise ValueError(
                        f"state_index {idx} is out of range for {n} computed "
                        "states."
                    )
                fixed[slot] = idx % n

        free_slots = [s for s in range(2) if score_vectors[s] is not None]

        if len(free_slots) == 0:
            chosen = [fixed[0], fixed[1]]
        elif len(free_slots) == 1:
            s = free_slots[0]
            other = 1 - s
            other_idx = fixed[other]
            order = np.argsort(score_vectors[s])[::-1]
            pick = None
            for o in order:
                if int(o) != other_idx:
                    pick = int(o)
                    break
            if pick is None:
                raise RuntimeError(
                    "Root tracking could not find two distinct excited roots "
                    "for the paired scanner."
                )
            chosen = [None, None]
            chosen[s] = pick
            chosen[other] = other_idx
        else:
            # Both slots track: jointly maximise the summed overlap score over
            # all distinct ordered pairs.  Near-degenerate ties (combined scores
            # within ``eps``) are broken towards the pair with the smaller
            # excitation-energy gap, i.e. the seam the optimisation is hunting.
            used = {f for f in fixed if f is not None}
            available = [k for k in range(n) if k not in used]
            if len(available) < 2:
                raise RuntimeError(
                    "Not enough distinct excited states to keep the two "
                    "tracked roots apart."
                )
            eps = 1e-9
            best_pair = None
            best_score = -np.inf
            best_tie_gap = np.inf
            for i in available:
                for j in available:
                    if i == j:
                        continue
                    total = float(score_vectors[0][i] + score_vectors[1][j])
                    tie_gap = abs(
                        float(excitations[i].excitation_energy)
                        - float(excitations[j].excitation_energy)
                    )
                    better = total > best_score + eps
                    tie = abs(total - best_score) <= eps and tie_gap < best_tie_gap
                    if better or tie:
                        best_pair = (i, j)
                        best_score = total
                        best_tie_gap = tie_gap
            chosen = [best_pair[0], best_pair[1]]

        if chosen[0] == chosen[1]:
            raise RuntimeError(
                "Distinctness guard: the two tracked roots collapsed onto "
                f"state {chosen[0]}. Cannot keep the paired surfaces apart."
            )

        trackings: list[Optional[TrackingResult]] = [None, None]
        for slot in range(2):
            sv = score_vectors[slot]
            idx = chosen[slot]
            if sv is None:
                # Positional seeding: mirror the single-surface scanner, which
                # records no tracking diagnostic on the seeding call.
                continue
            other = chosen[1 - slot]
            # Gap to the best *alternative* root available to this slot, i.e.
            # excluding the partner's selected root.  When exactly two excited
            # states were computed (n == 2) each slot is forced onto the
            # partner's leftover root, so no per-channel "best vs second-best"
            # gap is measurable here: ``remaining`` is empty.
            remaining = [k for k in range(n) if k != idx and k != other]
            if remaining:
                second = max(remaining, key=lambda k: sv[k])
                gap = float(sv[idx] - sv[second])
            elif self.tracking_min_gap > 0:
                raise RuntimeError(
                    f"tracking_min_gap > 0 needs at least three computed "
                    f"excited states to measure a per-channel gap for paired "
                    f"slot {slot} (only {n} available)."
                )
            else:
                gap = float("inf")
            prev_idx = self.previous_indices[slot]
            switched = prev_idx is not None and idx != prev_idx
            trackings[slot] = TrackingResult(
                index=idx, scores=sv, best_score=float(sv[idx]), gap=gap,
                previous_index=prev_idx, switched=bool(switched),
                unavailable_channels=tuple(sorted(unavailable)),
            )
            if sv[idx] < self.tracking_min_score:
                raise RuntimeError(
                    f"Root tracking failed for paired slot {slot}: best "
                    f"state-character overlap {sv[idx]:.6g} is below "
                    f"threshold {self.tracking_min_score:.6g}."
                )
            if gap < self.tracking_min_gap:
                raise RuntimeError(
                    f"Root tracking is ambiguous for paired slot {slot}: "
                    f"best and second-best overlaps differ by {gap:.6g}."
                )
        return chosen, descriptors, trackings

    def _select_excited_root(self, states, mol, prev_descriptor, prev_index,
                             seed_index):
        """Select a single tracked excited root (MECP excited side).

        A faithful paired-side copy of
        :meth:`NuclearGradientScanner._select_excitation` that keeps the
        previous-descriptor / previous-index state local to this slot.  Returns
        ``(excitation, tracking, descriptor, index)``.
        """
        excitations = states.excitations
        if not excitations:
            raise RuntimeError(
                "ADC calculation did not return any excited states."
            )
        n = len(excitations)

        if self.follow == "index" or prev_descriptor is None:
            idx = int(seed_index)
            if idx >= n or idx < -n:
                raise ValueError(
                    f"state_index {idx} is out of range for {n} computed states."
                )
            idx %= n
            descriptor = self._descriptor(excitations[idx], mol)
            return excitations[idx], None, descriptor, idx

        unavailable: set[str] = set()
        descriptors = [
            self._descriptor(excitation, mol) for excitation in excitations
        ]
        scores = np.array([
            self._tracking_score(prev_descriptor, descriptors[k], mol, unavailable)
            for k in range(n)
        ])
        order = np.argsort(scores)[::-1]
        best = int(order[0])
        if len(order) > 1:
            gap = float(scores[best] - scores[int(order[1])])
        elif self.tracking_min_gap > 0:
            raise RuntimeError(
                "tracking_min_gap > 0 needs at least two computed excited "
                f"states to measure a gap (only {n} available)."
            )
        else:
            gap = float("inf")
        switched = prev_index is not None and best != prev_index
        tracking = TrackingResult(
            index=best, scores=scores, best_score=float(scores[best]), gap=gap,
            previous_index=prev_index, switched=bool(switched),
            unavailable_channels=tuple(sorted(unavailable)),
        )
        if scores[best] < self.tracking_min_score:
            raise RuntimeError(
                "Root tracking failed: best state-character overlap "
                f"{scores[best]:.6g} is below threshold "
                f"{self.tracking_min_score:.6g}."
            )
        if len(order) > 1 and gap < self.tracking_min_gap:
            raise RuntimeError(
                "Root tracking is ambiguous: best and second-best overlaps "
                f"differ by {gap:.6g}."
            )
        return excitations[best], tracking, descriptors[best], best
