import numpy as np
from scipy import constants
import warnings

from .AdcMatrix import AdcMatrix
from .AdcMethod import AdcMethod
from .AmplitudeVector import AmplitudeVector
from .FormatDominantElements import FormatDominantElements
from .FormatIndex import (
    FormatIndexBase, FormatIndexAdcc, FormatIndexHfProvider, FormatIndexHomoLumo
)
from .LazyMp import LazyMp
from .OneParticleOperator import OneParticleOperator, product_trace
from .OperatorIntegrals import OperatorIntegrals
from .misc import cached_member_function, requires_module
from .ReferenceState import ReferenceState
from .solver.SolverStateBase import EigenSolverStateBase
from .StateView import StateView
from .timings import Timer
from .visualisation.Spectrum import Spectrum


_timer_name = "_property_timer"


class ElectronicStates:
    # The module where the equations for the ADC scheme are implemented, e.g.,
    # adc_pp for PP-ADC. Overwrite the class variable on child classes to
    # forward the call to the corresponding module. This assumes that the
    # structure within each module is consistent for all adc_types, which should
    # be fine I think.
    _module = None
    # The class used to obtain a view on a single state, e.g., Excitation for the
    # ExcitedStates class. Has to be defined on the corresponding child class.
    _state_view_cls: StateView = None

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

    def _state_view(self, state_n: int) -> StateView:
        """
        Provides a view onto a single state and his properties.
        """
        return self._state_view_cls(self, state_n)

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
        assert self._state_view_cls is not None
        blacklist = ("parent_state")
        ret = []
        for key in dir(self._state_view_cls):
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

    @requires_module("matplotlib")
    def _plot_spectrum(self, ylabel: str, yvalues: np.ndarray, xaxis: str = "eV",
                       broadening="lorentzian", width: float = 0.01,
                       width_unit: str = "au", **kwargs):
        """
        One-shot plotting function for the spectrum generated by all states
        known to this class.

        Makes use of the :class:`adcc.visualisation.Spectrum` class in order to
        generate and format the spectrum to be plotted,
        using many sensible defaults.

        Parameters
        ----------
        ylabel: str
            The label for the y-Axis.
        yvalues: np.ndarray
            The y-values to plot.
        xaxis : str
            Energy unit to be used on the x-Axis. Options:
            ["eV", "au", "nm", "cm-1"].
        broadening : str or None or callable, optional
            The broadening type to used for the computed excitations.
            A value of None disables broadening any other value is passed
            straight to
            :func:`adcc.visualisation.Spectrum.broaden_lines`.
        width : float, optional
            Gaussian broadening standard deviation or Lorentzian broadening
            gamma parameter. The value should be given in atomic units
            and will be converted to the unit of the energy axis.
        width_unit: str, optional
            The unit the width is given in. All xaxis options except "nm" are
            possible.
        """

        def convert_x_units(spectrum: Spectrum):
            # modifies the spectrum in place by converting the x values to
            # the desired unit
            if xaxis in ["au", "hartree", "a.u."]:
                spectrum.xlabel = "Energy (au)"
            elif xaxis == "ev":
                eV = constants.value("Hartree energy in eV")
                spectrum.x = spectrum.x * eV
                spectrum.xlabel = "Energy (eV)"
            elif xaxis == "nm":
                hc = constants.h * constants.c
                Eh = constants.value("Hartree energy")
                spectrum.x = hc / (spectrum.x * Eh) * 1e9
                spectrum.xlabel = "Wavelength (nm)"
            elif xaxis in ["cm-1", "cm^-1", "cm^{-1}"]:
                towvn = constants.value("hartree-inverse meter relationship") / 100
                spectrum.x = spectrum.x * towvn
                spectrum.xlabel = "Wavenumbers (cm^{-1})"
            else:
                raise ValueError(f"Unknown xaxis specifier: {xaxis}")

        # convert the width to au
        width_unit = width_unit.lower()
        if width_unit in ["au", "hartree", "a.u."]:
            pass
        elif width_unit == "ev":
            width = width / constants.value("Hartree energy in eV")
        elif width_unit in ["cm-1", "cm^-1", "cm^{-1}"]:
            towvn = constants.value("hartree-inverse meter relationship") / 100
            width = width / towvn
        else:
            raise ValueError(f"Invalid width_unit '{width_unit}'")

        # first build the spectra in atomic units on the x-axis
        spectrum = Spectrum(x=self.excitation_energy, y=yvalues, ylabel=ylabel)
        if broadening:  # generate a broadened spectrum
            broadened_spectrum: Spectrum = spectrum.broaden_lines(
                width=width, shape=broadening
            )
        xaxis = xaxis.lower()
        # convert the x-unit of the dicrete spectrum
        convert_x_units(spectrum)
        if broadening:
            # convert the x-unit of the broadened spectrum
            convert_x_units(broadened_spectrum)
            # plot both spectra:
            # - avoid labeling the discrete spectrum
            kwdisc = {k: v for k, v in kwargs.items() if k != "label"}
            plots = spectrum.plot(style="discrete", **kwdisc)
            # - use the same color as for the discrete spectrum
            kwbroad = {k: v for k, v in kwargs.items() if k != "color"}
            disc_color = plots[0].get_color()
            plots.extend(broadened_spectrum.plot(
                color=disc_color, style="continuous", **kwbroad
            ))
        else:
            # plot the discrete spectrum
            plots = spectrum.plot(style="discrete", **kwargs)
        return plots

    def _describe(self, columns: list["TableColumn"]):
        """
        Return a string providing a human-readable description of the class

        Parameters
        ----------
        columns: list[TableColumn]
            The columns of the table that is generated by the method.
        """
        lines = []
        # - Determine the overall width of the table
        table_width = sum(c.width for c in columns)
        # the width of the header will be taken account below
        # but we also need to consider the excitation energy corrections
        # that are printed below after
        if self._excitation_energy_corrections:
            correction_head = "Excitation energy includes these corrections:"
            corr_width = max(len(correction_head),
                             *(len(c.name) for c in
                               self._excitation_energy_corrections))
            if corr_width > table_width:
                # add padding to the last column
                new_width = columns[-1].width + corr_width - table_width
                columns[-1] = columns[-1].with_width(new_width)
                table_width = corr_width

        # - Format the header
        # Format the method
        method = self.method.name
        if self.property_method != self.method:
            method += f" ({self.property_method.name})"
        # Format the state information: kind, spin_change and convergence
        state_info = []
        if hasattr(self, "kind") and self.kind:
            state_info.append(self.kind)
        if hasattr(self, "spin_change") and self.spin_change is not None and \
                self.spin_change != 0:
            state_info.append(f"(ΔMS={self.spin_change:+2d})")
        if hasattr(self, "converged"):
            conv = "converged" if self.converged else "NOT CONVERGED"
            if state_info:  # add separator to previous entry
                state_info[-1] += ","
            state_info.append(conv)
        state_info = " ".join(state_info)
        # actually format the header
        if table_width > len(method) + 4:
            header = (  # -4 for the spaces
                f"{method:s}    {state_info:>{str(table_width - len(method) - 4)}s}"
            )
        else:  # only few small columns -> header will determine the table width
            header = f"{method:s}    {state_info:s}"

        # if the header is wider than the current table width:
        # add padding to the last column
        if len(header) > table_width:
            new_width = columns[-1].width + len(header) - table_width
            columns[-1] = columns[-1].with_width(new_width)
            table_width = len(header)
        assert len(header) == table_width

        # The hline separator for the table:
        # +2: add a space to the left and the right
        table_hline = "+" + "-" * (table_width + 2) + "+"

        # Actually format the table
        lines.append(table_hline)
        lines.append(f"| {header} |")
        lines.append(table_hline)
        lines.append(f"| {''.join(c.header for c in columns)} |")
        lines.append(f"| {''.join(c.unit for c in columns)} |")
        assert all(len(c.values) == self.size for c in columns)
        for state_values in zip(*[c.values for c in columns]):
            lines.append(f"| {''.join(state_values)} |")
        lines.append(table_hline)
        # Potentially format energy corrections
        if self._excitation_energy_corrections:
            assert len(correction_head) <= table_width
            lines.append(f"| {correction_head:<{str(table_width)}s} |")
            maxlen = table_width - 5  # -5 for the spaces and - below
            for eec in self._excitation_energy_corrections:
                lines.append(f"|    - {eec.name:<{maxlen}s} |")
        return "\n".join(lines)

    def describe_amplitudes(self, tolerance: float = 0.01,
                            index_format=None) -> str:
        """
        Return a string describing the dominant amplitudes of each
        excitation vector in human-readable form. The ``kwargs``
        are for :py:class:`FormatExcitationVector`.

        Parameters
        ----------
        tolerance : float, optional
            Minimal absolute value of the excitation amplitudes considered.

        index_format : NoneType or str or FormatIndexBase, optional
            Formatter to use for displaying tensor indices.
            Valid are ``"adcc"`` to keep the adcc-internal indexing,
            ``"hf"`` to select the HFProvider indexing, ``"homolumo"``
            to index relative on the HOMO / LUMO / HOCO orbitals.
            If ``None`` an automatic selection will be made.
        """
        eV = constants.value("Hartree energy in eV")
        formatter = FormatAmplitudeVector(
            matrix=self.matrix, tolerance=tolerance, index_format=index_format
        )
        # optimise the formatting by pre-inspecting the amplitude coefficients
        # to determine the (str) length of indices that will be printed.
        formatter.optimise_formatting(tuple(self.excitation_vector))
        # first format all the headers to determine their length
        headers = []
        for i, energy in enumerate(self.excitation_energy):
            headers.append(
                f"State {i:3d}, {energy:13.7g} au, {energy * eV:13.7g} eV"
            )
        # determine the required linewidth
        width = max(formatter.linewidth, *(len(h) for h in headers))
        hline = "+" + (width + 2) * "-" + "+"  # +2: space to the left and right

        lines = []
        for header, amplitude in zip(headers, self.excitation_vector):
            lines.append(hline)
            lines.append(f"| {header:<{width}s} |")
            lines.append(hline)
            # format the vector and add the table borders + spaces
            for coefficient in formatter.format(amplitude):
                lines.append(
                    f"| {coefficient:^{width}s} |"
                )
            lines.append(hline)
        return "\n".join(lines)


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


class TableColumn:
    # represents a column in the table generated by the describe method
    def __init__(self, header: str, values: tuple[str, ...], unit: str = "",
                 width: int = None, align: str = "^"):
        # the minimum required width of the column
        min_width = max(len(header), len(unit), *(len(v) for v in values))
        if width is None or width < min_width:
            width = min_width + 1

        self.width = width
        self.header = f"{header:{align}{str(width)}s}"
        self.values = tuple(f"{v:{align}{str(width)}s}" for v in values)
        self.unit = f"{unit:{align}{str(width)}s}"
        self.align = align

    def with_width(self, width: int) -> "TableColumn":
        # remove whitespaces in case we want to shrink the column!
        values = tuple(v.strip() for v in self.values)
        return TableColumn(
            header=self.header.strip(), values=values, unit=self.unit.strip(),
            width=width, align=self.align
        )

    def with_align(self, align: str) -> "TableColumn":
        # remove whitespaces so we can change the alignment
        values = tuple(v.strip() for v in self.values)
        return TableColumn(
            header=self.header.strip(), values=values, unit=self.unit.strip(),
            width=self.width, align=align
        )


class FormatAmplitudeVector:
    def __init__(self, matrix: AdcMatrix, tolerance: float = 0.01,
                 index_format: str = None):
        """
        Set up a formatter class for formatting excitation vectors.

        Parameters
        ----------
        tolerance : float, optional
            Minimal absolute value of the excitation amplitudes considered

        index_format : NoneType or str or FormatIndexBase, optional
            Formatter to use for displaying tensor indices.
            Valid are ``"adcc"`` to keep the adcc-internal indexing,
            ``"hf"`` to select the HFProvider indexing, ``"homolumo"``
            to index relative on the HOMO / LUMO / HOCO orbitals.
            If ``None`` an automatic selection will be made.
        """
        self.matrix = matrix
        refstate = matrix.reference_state
        # default for index formatting
        if index_format is None:
            closed_shell = refstate.n_alpha == refstate.n_beta
            if closed_shell and refstate.is_aufbau_occupation:
                index_format = "homolumo"
            else:
                index_format = "hf"

        if index_format == "adcc":
            index_format = FormatIndexAdcc(refstate)
        elif index_format == "hf":
            index_format = FormatIndexHfProvider(refstate)
        elif index_format == "homolumo":
            index_format = FormatIndexHomoLumo(refstate)
        elif not isinstance(index_format, FormatIndexBase):
            raise ValueError("Unsupported value for index_format: "
                             + str(index_format))

        self.tensor_format = FormatDominantElements(
            matrix.mospaces, tolerance, index_format
        )
        self.index_format = self.tensor_format.index_format
        # Formatting used for the values (coefficients)
        self.value_format = "{:+8.3g}"

    def optimise_formatting(self, vectors: list[AmplitudeVector]) -> None:
        """
        Optimise the formatting of the amplitude vectors by inspecting the entries
        of the given amplitude vectors in advance.
        This enables to nicely format indices by e.g. determining the maximum
        length of indices that will be printed.
        """
        if not isinstance(vectors, (list, tuple)):
            return self.optimise_formatting([vectors])

        for vector in vectors:
            for block, spaces in self.matrix.axis_spaces.items():
                self.tensor_format.optimise_formatting((spaces, vector[block]))

    @property
    def linewidth(self) -> int:
        """
        The width of a line if an amplitude is formatted with this class
        """
        # This assumes a formatting: idx1 idx2 -> idx3 idx4   aa->aa   number
        nblk = len(self.matrix.axis_blocks[-1])  # axis blocks is sorted by len
        # n_idx * (len(index) + space) + 2 for the arrow
        width_indices = nblk * (self.index_format.max_n_characters + 1) + 2
        width_spins = nblk + 2  # n_idx * 1 (for a/b) + 2 for the arrow
        # the width of the numerical value of the coefficient
        width_value = len(self.value_format.format(0))
        # +6: 3 spaces between indices and spin + 3 spaces between spins and value
        return width_indices + width_spins + width_value + 6

    def format(self, amplitude: AmplitudeVector) -> list[str]:
        """
        Formats the given amplitude vector.
        """
        empty_idx = self.index_format.max_n_characters * " "
        idx_spin_gap = " " * 3
        spin_coeff_gap = " " * 3

        if self.matrix.axis_blocks == ["ph"]:
            formats = {"ov": (
                "{} -> {}" + idx_spin_gap + "{}->{}"
                + spin_coeff_gap + self.value_format
            )}
        elif self.matrix.axis_blocks == ["ph", "pphh"]:
            formats = {
                "ov": (
                    "{} " + empty_idx + " -> {} " + empty_idx + idx_spin_gap
                    + "{} ->{} " + spin_coeff_gap + self.value_format
                ),
                "oovv": (
                    "{} {} -> {} {}" + idx_spin_gap + "{}{}->{}{}"
                    + spin_coeff_gap + self.value_format
                )
            }
        else:
            raise NotImplementedError("Unknown ADC matrix structure")

        lines = []
        for block, spaces in self.matrix.axis_spaces.items():
            # remove the numbers from the spaces: o1 -> o
            space_str = "".join(c for c in "".join(spaces) if c.isalpha())

            assert block in amplitude
            formatted = self.tensor_format.format_as_list(spaces, amplitude[block])
            for indices, spins, coeff in formatted:
                lines.append(
                    formats[space_str].format(*indices, *spins, coeff)
                )
        return lines
