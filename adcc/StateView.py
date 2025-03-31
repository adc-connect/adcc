import numpy as np

from .AmplitudeVector import AmplitudeVector
from .OneParticleOperator import OneParticleOperator


class StateView:
    def __init__(self, parent_state, index: int):
        """
        Construct an StateView instance from an :class:`adcc.ElectronicStates`
        parent object.

        The class provides access to the properties of a single state.

        Parameters
        ----------
        parent_state
            :class:`adcc.ElectronicState` object from which the StateView
            is derived
        index : int
            Index of the state the constructed :class:`adcc.StateView`
            should refer to (0-based)
        """
        from .ElectronicStates import ElectronicStates
        # valid range for index: -n_states <= index < +n_states
        if index >= parent_state.size or \
                (index < 0 and abs(index) > parent_state.size):
            raise ValueError(f"index {index} is out of range for a parent state "
                             f"with {parent_state.size} states.")
        self._parent_state: ElectronicStates = parent_state
        self.index: int = index

    @property
    def parent_state(self):
        return self._parent_state

    @property
    def excitation_energy(self):
        """Excitation energy including all corrections in atomic units"""
        return self._parent_state.excitation_energy[self.index]

    @property
    def excitation_energy_uncorrected(self):
        """Excitation energy without any corrections in atomic units"""
        return self._parent_state._excitation_energy_uncorrected[self.index]

    @property
    def excitation_vector(self) -> AmplitudeVector:
        """The excitation vector"""
        return self._parent_state._excitation_vector[self.index]

    @property
    def state_diffdm(self) -> OneParticleOperator:
        """The difference density matrix"""
        return self._parent_state._state_diffdm(self.index)

    @property
    def state_diffdm_ao(self) -> OneParticleOperator:
        """The difference density matrix in the AO basis"""
        return sum(self.state_diffdm.to_ao_basis())

    @property
    def state_dm(self) -> OneParticleOperator:
        """The state density matrix"""
        return self._parent_state._state_dm(self.index)

    @property
    def state_dm_ao(self) -> OneParticleOperator:
        """The state density matrix in the AO basis"""
        return sum(self.state_dm.to_ao_basis())

    @property
    def state_dipole_moment(self) -> np.ndarray:
        """Array of state dipole moments"""
        return self._parent_state._state_dipole_moment(self.index)
