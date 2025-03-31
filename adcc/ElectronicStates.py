import numpy as np
import warnings

from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .LazyMp import LazyMp
from .OneParticleOperator import OneParticleOperator, product_trace
from .OperatorIntegrals import OperatorIntegrals
from .misc import cached_member_function, requires_module
from .ReferenceState import ReferenceState
from .solver.SolverStateBase import EigenSolverStateBase
from .StateView import StateView
from .timings import Timer


_timer_name = "_property_timer"


class ElectronicStates:
    # The module where the equations for the ADC scheme are implemented, e.g.,
    # adc_pp for PP-ADC. Overwrite the class variable on child classes to
    # forward the call to the corresponding module. This assumes that the
    # structure within each module is consistent for all adc_types, which should
    # be fine I think.
    _module = None
    # The type used to obtain a view on a single state, e.g., Excitation for the
    # ExcitedStates class. Has to be set on the corresponding child class.
    _state_view_type = None

    def __init__(self, data, method: str = None,
                 property_method: str = None) -> None:
        """
        Construct an ElectronicStates class from some data obtained from an
        iterative solver or another :class:`ElectronicStates` instance.

        Parameters
        ----------
        data
            Any kind of iterative solver state. Typically derived off
            a :class:`solver.EigenSolverStateBase`.
        method : str, optional
            Provide an explicit method parameter if data contains none.
        property_method : str, optional
            Provide an explicit method for property calculations to
            override the automatic selection.
    """
        self.matrix: AdcMatrix = data.matrix
        self.ground_state: LazyMp = self.matrix.ground_state
        self.reference_state: ReferenceState = (
            self.matrix.ground_state.reference_state
        )
        self.operators: OperatorIntegrals = self.reference_state.operators

        # List of all the objects which have timers (do not yet collect
        # timers, since new times might be added implicitly at a later point)
        self._property_timer: Timer = Timer()
        self._timed_objects = [("", self.reference_state),
                               ("adcmatrix", self.matrix),
                               ("mp", self.ground_state),
                               ("intermediates", self.matrix.intermediates)]
        if hasattr(data, "timer"):
            datakey = getattr(data, "algorithm", data.__class__.__name__)
            self._timed_objects.append((datakey, data))

        # Copy some optional attributes
        for optattr in ["converged", "spin_change", "kind", "n_iter"]:
            if hasattr(data, optattr):
                setattr(self, optattr, getattr(data, optattr))

        self.method = getattr(data, "method", method)
        if self.method is None:
            self.method = self.matrix.method
        if not isinstance(self.method, AdcMethod):
            self.method: AdcMethod = AdcMethod(self.method)

        if property_method is None:
            if self.method.level < 3:
                property_method = self.method
            else:
                # Auto-select ADC(2) properties for ADC(3) calc
                property_method = self.method.at_level(2)
        elif not isinstance(property_method, AdcMethod):
            property_method = AdcMethod(property_method)
        self._property_method: AdcMethod = property_method

        # Special stuff for special solvers
        if isinstance(data, EigenSolverStateBase):
            self._excitation_vector = data.eigenvectors
            self._excitation_energy_uncorrected = data.eigenvalues
            self.residual_norm = data.residual_norms
        else:  # other Solver or e.g. from another ExcitedStates instance
            if hasattr(data, "eigenvalues"):
                self._excitation_energy_uncorrected = data.eigenvalues
            if hasattr(data, "eigenvectors"):
                self._excitation_vector = data.eigenvectors
            # if both excitation_energy and excitation_energy_uncorrected
            # are present, the latter one has priority: otherwise corrections
            # would be added again below!
            if hasattr(data, "excitation_energy"):
                self._excitation_energy_uncorrected = \
                    data.excitation_energy.copy()
            if hasattr(data, "excitation_energy_uncorrected"):
                self._excitation_energy_uncorrected =\
                    data.excitation_energy_uncorrected.copy()
            if hasattr(data, "excitation_vector"):
                self._excitation_vector = data.excitation_vector

        self._excitation_energy = self._excitation_energy_uncorrected.copy()

        # Collect all excitation energy corrections
        self._excitation_energy_corrections: list[EnergyCorrection] = []
        # copy energy corrections if possible
        # and avoids re-computation of the corrections
        if hasattr(data, "_excitation_energy_corrections"):
            for eec in data._excitation_energy_corrections:
                correction_energy = getattr(data, eec.name)
                if hasattr(self, eec.name):
                    raise ValueError(f"{self.__name__} already has an attribute "
                                     f"with the name '{eec.name}'")
                setattr(self, eec.name, correction_energy)
                self._excitation_energy += correction_energy
                self._excitation_energy_corrections.append(eec)

    def __len__(self) -> int:
        return self.size

    @property
    def size(self) -> int:
        return self._excitation_energy.size

    @property
    def property_method(self) -> AdcMethod:
        """The method used to evaluate ADC properties"""
        return self._property_method

    @property
    def timer(self) -> Timer:
        """Return a cumulative timer collecting timings from the calculation"""
        ret = Timer()
        for key, obj in self._timed_objects:
            ret.attach(obj.timer, subtree=key)
        ret.attach(self._property_timer, subtree="properties")
        ret.time_construction = self.reference_state.timer.time_construction
        return ret

    @property
    def excitation_energy(self):
        """Excitation energies including all corrections in atomic units"""
        return self._excitation_energy

    @property
    def excitation_energy_uncorrected(self):
        """Excitation energies without any corrections in atomic units"""
        return self._excitation_energy_uncorrected

    @property
    def excitation_vector(self):
        """List of excitation vectors"""
        return self._excitation_vector

    @property
    def state_diffdm(self) -> list[OneParticleOperator]:
        """List of difference density matrices of all computed states"""
        return [self._state_diffdm(state_n=i) for i in range(self.size)]

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _state_diffdm(self, state_n: int) -> OneParticleOperator:
        """Computes the difference density matrix for a single state"""
        evec = self.excitation_vector[state_n]
        return self._module.state_diffdm(
            self.property_method, self.ground_state, evec, self.matrix.intermediates
        )

    @property
    def state_dm(self) -> list[OneParticleOperator]:
        """List of state density matrices of all computed states"""
        return [self._state_dm(i) for i in range(self.size)]

    def _state_dm(self, state_n: int) -> OneParticleOperator:
        """List of state density matrices of all computed states"""
        mp_density = self.ground_state.density(self.property_method.level)
        diffdm = self._state_diffdm(state_n)
        return mp_density + diffdm

    @property
    def state_dipole_moment(self) -> np.ndarray:
        """Array of state dipole moments"""
        return np.array([self._state_dipole_moment(i) for i in range(self.size)])

    @cached_member_function(timer=_timer_name, separate_timings_by_args=False)
    def _state_dipole_moment(self, state_n: int) -> np.ndarray:
        """Computes the state dipole moment for a single state"""
        pmethod = self.property_method
        if pmethod.level == 0:
            gs_dip_moment = self.reference_state.dipole_moment
        else:
            gs_dip_moment = self.ground_state.dipole_moment(pmethod.level)

        dipole_integrals = self.operators.electric_dipole
        ddm = self._state_diffdm(state_n)
        return (gs_dip_moment
                + np.array([product_trace(comp, ddm) for comp in dipole_integrals]))

    def _state_view(self, state_n: int):
        """
        Provides a view onto a single state and his properties. This method has
        to be implemented on the child classes, since the view depends on the
        adc_variant.
        """
        return NotImplemented

    def _add_energy_correction(self, correction: "EnergyCorrection") -> None:
        assert isinstance(correction, EnergyCorrection)
        if hasattr(self, correction.name):
            raise ValueError(f"{self.__class__.__name__} already has an attribute "
                             f" with the name '{correction.name}'")
        correction_energy = np.array([
            correction(self._state_view(i)) for i in range(self.size)
        ])
        setattr(self, correction.name, correction_energy)
        self._excitation_energy += correction_energy
        self._excitation_energy_corrections.append(correction)

    def __iadd__(self, other):
        if isinstance(other, EnergyCorrection):
            self._add_energy_correction(other)
        elif isinstance(other, list):
            for k in other:
                self += k
        else:
            return NotImplemented
        return self

    def __add__(self, other):
        if not isinstance(other, (EnergyCorrection, list)):
            return NotImplemented
        ret = self.__class__(self, self.method, self.property_method)
        ret += other
        return ret

    @property
    def excitation_property_keys(self) -> list[str]:
        """
        Extracts the names of available excited state and transition properties.
        """
        # NOTE: this currently assumes that all available properties are available
        # on the corresponding state_view class
        assert self._state_view_type is not None
        blacklist = ("parent_state")
        ret = []
        for key in dir(self._state_view_type):
            # private fields or ao transformed densities or other fields
            if key.startswith("_") or key.endswith("_ao") or key in blacklist:
                continue
            ret.append(key)
        return ret

    @requires_module("pandas")
    def to_dataframe(self, gauge_origin="origin"):
        """
        Exports the class object as :class:`pandas.DataFrame`.
        Atomic units are used for all values.
        """
        import pandas as pd
        propkeys = self.excitation_property_keys
        propkeys.extend([k.name for k in self._excitation_energy_corrections])
        data = {
            "excitation": np.arange(0, self.size, dtype=int),
            "kind": np.tile(self.kind, self.size)
        }
        for key in propkeys:
            try:
                d = getattr(self, key)
            except NotImplementedError:
                # some properties are not available for every backend
                continue
            # This assumes that all callables take a single argument:
            # the gauge origin
            if callable(d):
                try:
                    d = d(gauge_origin)
                except NotImplementedError:
                    # some properties are not available for every backend
                    continue

            if not isinstance(d, np.ndarray):
                continue
            if not np.issubdtype(d.dtype, np.number):
                continue

            if d.ndim == 1:
                data[key] = d
            elif d.ndim == 2 and d.shape[1] == 3:
                for i, p in enumerate(["x", "y", "z"]):
                    data[f"{key}_{p}"] = d[:, i]
            elif d.ndim == 3 and d.shape[1:] == (3, 3):
                for i, p in enumerate(["x", "y", "z"]):
                    for j, q in enumerate(["x", "y", "z"]):
                        data[f"{key}_{p}{q}"] = d[:, i, j]
            elif d.ndim > 3:
                warnings.warn(f"Exporting NumPy array for property {key}"
                              f" with shape {d.shape} not supported.")
                continue
        df = pd.DataFrame(data=data)
        df.set_index("excitation")
        return df


class EnergyCorrection:
    def __init__(self, name: str, function: callable) -> None:
        """A helper class to represent excitation energy
        corrections.

        Parameters
        ----------
        name : str
            descriptive name of the energy correction
        function : callable
            function that takes a :py:class:`Excitation`
            as single argument and returns the energy
            correction as a float
        """
        if not isinstance(name, str):
            raise TypeError("name needs to be a string.")
        if not callable(function):
            raise TypeError("function needs to be callable.")
        self.name: str = name
        self.function: callable = function

    def __call__(self, excitation: StateView):
        assert isinstance(excitation, StateView)
        return self.function(excitation)
