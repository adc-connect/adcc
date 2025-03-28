import numpy as np

from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .Excitation import Excitation
from .LazyMp import LazyMp
from .OperatorIntegrals import OperatorIntegrals
from .ReferenceState import ReferenceState
from .solver.SolverStateBase import EigenSolverStateBase
from .timings import Timer


class ElectronicStates:
    """
    Construct an ElectronicStates class from some data obtained from an iterative
    solver or another :class:`ElectronicStates` instance.

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
    def __init__(self, data, method: str = None,
                 property_method: str = None) -> None:
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
        self._excitation_energy_corrections = []
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
    # @mark_excitation_property()
    def excitation_energy(self):
        """Excitation energies including all corrections in atomic units"""
        return self._excitation_energy

    @property
    # @mark_excitation_property()
    def excitation_energy_uncorrected(self):
        """Excitation energies without any corrections in atomic units"""
        return self._excitation_energy_uncorrected

    @property
    # @mark_excitation_property()
    def excitation_vector(self):
        """List of excitation vectors"""
        return self._excitation_vector

    def _add_energy_correction(self, correction: "EnergyCorrection") -> None:
        assert isinstance(correction, EnergyCorrection)
        if hasattr(self, correction.name):
            raise ValueError(f"{self.__name__} already has an attribute "
                             f" with the name '{correction.name}'")
        # TODO: need to figure out how to dispatch to the Excitation interface
        # for IP/EA
        return NotImplemented
        correction_energy = np.array([correction(exci)
                                      for exci in self.excitations])
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
        self.name = name
        self.function = function

    def __call__(self, excitation: Excitation):
        assert isinstance(excitation, Excitation)
        return self.function(excitation)
