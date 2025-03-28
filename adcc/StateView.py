from .AmplitudeVector import AmplitudeVector


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
        # valid range for index: -n_states <= index < +n_states
        if index >= parent_state.size or \
                (index < 0 and abs(index) > parent_state.size):
            raise ValueError(f"index {index} is out of range for a parent state "
                             f"with {parent_state.size} states.")
        self._parent_state = parent_state
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
