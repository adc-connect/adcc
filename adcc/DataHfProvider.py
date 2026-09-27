## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
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
import re
from collections.abc import Mapping
from string import Formatter
from typing import Any, ClassVar, Protocol, TypeVar, runtime_checkable

import h5py
import numpy as np

from libadcc import HartreeFockProvider

from .backends import OperatorIntegralProvider
from .typing import (
    Array1D,
    Array2D,
    Array4D,
    Coordinate,
    DipoleLikeArray,
    FloatArray,
    GaugeOrigin,
    QuadrupoleLikeArray,
    ShapeT,
    Slices2D,
    Slices4D,
    is_dipole_like_array,
    is_float_array,
    is_quadrupole_like_array,
)

DataT = TypeVar("DataT")


@runtime_checkable
class ArrayHandle(Protocol):
    """
    An array handle that can be inspected and sliced, e.g. a np.ndarray or h5py.Dataset.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype[Any]: ...
    def __getitem__(self, key: Any) -> np.ndarray: ...
    def __array__(self, dtype: Any = ...) -> np.ndarray: ...


class MissingDataError(KeyError, NotImplementedError):
    """Error raised when the key is not available in the data container."""


def _load_from_data(data: Mapping[str, Any], key: str, default: Any = None) -> object:
    """
    Walk the ``data`` container and extract the desired data.
    Returns ``default`` if ``key`` is absent. Note that ``key`` is split at `/`
    to access nested data containers.
    """
    # check whether the h5py.File handle is still valid
    if isinstance(data, h5py.Group) and not data:
        raise ValueError("Cannot load data. HDF5 file was closed.")
    value: object = data
    for sub_key in key.split("/"):
        if not isinstance(value, Mapping) or sub_key not in value:
            return default
        value = value[sub_key]
    return value


def _convert_scalar(value: object, scalar_cls: type) -> object:
    """
    Applies some explicit, lossless conversions of ``value`` towards ``scalar_cls``.
    """
    if scalar_cls is str and isinstance(value, bytes):
        return value.decode()
    if scalar_cls is float and isinstance(value, int):
        return float(value)
    if scalar_cls is bool and isinstance(value, int) and value in (0, 1):
        return bool(value)
    return value


def _load_scalar_value(data: Mapping[str, Any], key: str, scalar_cls: type[DataT]) -> DataT:
    """
    Loads a scalar value (a value which is not an array) from ``data`` and ensures
    that the scalar value is of the correct type.
    """
    value = _load_from_data(data, key=key, default=None)
    if value is None:
        raise MissingDataError(f"Could not load scalar value from key '{key}'.")
    # check if the value is an array, validate the shape and import the entry
    # as native python type
    if isinstance(value, ArrayHandle):
        if value.shape not in ((), (1,)):
            raise ValueError(
                f"Expected a scalar under key '{key}'. Got entry with shape {value.shape}"
            )
        value = np.asarray(value).item()
    # some explicit type conversions. They live in a helper to avoid inferred types, that can not
    # be resolved by the isinstance check below.
    value = _convert_scalar(value, scalar_cls)
    if not isinstance(value, scalar_cls):
        raise TypeError(
            f"Scalar value loaded from key '{key}' has not the correct type. "
            f"Expected '{scalar_cls.__name__}', got '{value.__class__.__name__}'."
        )
    return value


def _get_array(
    data: Mapping[str, Any], key: str, shape: tuple[int, ...] | None = None
) -> ArrayHandle:
    """
    Returns the raw handle (np.ndarray or h5py.Dataset) to the array stored under ``key``
    without loading it. Optionally the shape is validated.
    """
    value = _load_from_data(data, key=key, default=None)
    if value is None:
        raise MissingDataError(f"Could not load array value from key '{key}'.")
    if not isinstance(value, ArrayHandle):
        raise TypeError(
            f"Expected a 'np.ndarray' or a 'h5py.Dataset' like object under key '{key}', "
            f"got '{value.__class__.__name__}'."
        )
    if shape is not None and value.shape != shape:
        raise ValueError(
            f"Array value under key '{key}' has invalid shape {value.shape}. Expected {shape}."
        )
    # only accept data types that can safely be cast to np.float64
    if not np.can_cast(value.dtype, np.float64, "safe"):
        raise TypeError(
            f"Array value under key '{key}' has dtype {value.dtype}, which can not be safely "
            "cast to 'np.float64'."
        )
    return value


def _load_array_value(data: Mapping[str, Any], key: str, shape: ShapeT) -> FloatArray[ShapeT]:
    """
    Loads a array from ``data`` and ensures it has the correct shape and dtype.
    """
    handle = _get_array(data, key=key, shape=shape)
    value = np.asarray(handle, dtype=np.float64)
    assert is_float_array(value, shape=shape)
    return value


class DataOperatorIntegralProvider(OperatorIntegralProvider):
    # dict holding pairs of operator names and the paths to the integrals in the data.
    # for nested containers the keys are separated by "/" and format strings are used
    # to dynamically adjust the keys at runtime to e.g. allow the selection of multiple
    # gauge origins.
    _operator_keys: ClassVar[dict[str, str]] = {
        "overlap": "overlap",
        "electric_dipole": "multipoles/elec_1",
        "electric_quadrupole": "multipoles/elec_2_{gauge_origin}",
        "magnetic_dipole": "magnetic_moments/mag_1_{gauge_origin}",
        "electric_dipole_velocity": "derivatives/elec_vel_1",
    }

    def __init__(self, data: Mapping[str, Any], n_bas: int, backend: str = "data"):
        """
        Access and load operator integrals from the ``data`` container and verify their
        shape against the provided number of basis functions ``n_bas``.
        """
        self._data: Mapping[str, Any] = data
        self._n_bas: int = n_bas
        self._backend: str = backend

    @property
    def backend(self) -> str:
        return self._backend

    def _resolve_key(self, name: str, gauge_origin: GaugeOrigin = "origin") -> str:
        """
        Returns the path to the given operator in the data.
        """
        key = self._operator_keys.get(name, None)
        if key is None:
            raise ValueError(
                f"Cannot load unknown operator integral {name}. "
                f"Known integrals are {tuple(self._operator_keys)}."
            )
        if not isinstance(gauge_origin, str):
            raise NotImplementedError(
                f"The {self.backend} backend only supports named gauge origins such as 'origin'."
                f"Got '{gauge_origin}'"
            )
        # for non-format strings nothing happens
        return key.format(gauge_origin=gauge_origin)

    def _contains(self, name: str) -> bool:
        """
        Whether the data container contains any data for the given integral.
        For keys that are format strings (e.g. gauge origin dependent integrals)
        it is checked whether data is available for ANY value of the format fields.
        """
        key = self._operator_keys.get(name, None)
        if key is None:
            return False
        parent, _, final = key.rpartition("/")
        # verify that parent is no format string
        if any(field is not None for _, field, _, _ in Formatter().parse(parent)):
            raise ValueError(
                "Format strings are only supported in the final component of an operator key. "
                f"Got '{key}' for the operator {name}."
            )
        # partially load the data
        data = _load_from_data(self._data, key=parent, default={}) if parent else self._data
        if not isinstance(data, Mapping):
            return False
        # work through the format string and replace possible format fields by wildcards
        # mag_{n}_{gauge_origin}_foo -> mag_.+_.+_foo
        pattern = re.compile(
            "".join(
                re.escape(literal) + ("" if field is None else ".+")
                for literal, field, _, _ in Formatter().parse(final)
            )
        )
        return any(pattern.fullmatch(stored) for stored in data)

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(name for name in self._operator_keys if self._contains(name))

    def _load_operator(
        self, name: str, shape: ShapeT, gauge_origin: GaugeOrigin = "origin"
    ) -> FloatArray[ShapeT]:
        """
        Load a given operator from the data container and verify its ``shape`` and ``dtype``.
        """
        key = self._resolve_key(name, gauge_origin=gauge_origin)
        return _load_array_value(self._data, key=key, shape=shape)

    @property
    def overlap(self) -> Array2D:
        return self._load_operator("overlap", (self._n_bas, self._n_bas))

    @property
    def electric_dipole(self) -> DipoleLikeArray:
        res = tuple(self._load_operator("electric_dipole", (3, self._n_bas, self._n_bas)))
        assert is_dipole_like_array(res)
        return res

    @property
    def electric_dipole_velocity(self) -> DipoleLikeArray:
        res = tuple(self._load_operator("electric_dipole_velocity", (3, self._n_bas, self._n_bas)))
        assert is_dipole_like_array(res)
        return res

    def magnetic_dipole(self, gauge_origin: GaugeOrigin = "origin") -> DipoleLikeArray:
        res = tuple(
            self._load_operator(
                "magnetic_dipole", (3, self._n_bas, self._n_bas), gauge_origin=gauge_origin
            )
        )
        assert is_dipole_like_array(res)
        return res

    def electric_quadrupole(self, gauge_origin: GaugeOrigin = "origin") -> QuadrupoleLikeArray:
        res = tuple(
            self._load_operator(
                "electric_quadrupole", (9, self._n_bas, self._n_bas), gauge_origin=gauge_origin
            )
        )
        assert is_quadrupole_like_array(res)
        return res


class DataHfProvider(HartreeFockProvider):
    def __init__(self, data: Mapping[str, Any]):
        """
        Initialise the DataHfProvider class with the `data` being a supported
        data container (e.g. a python dictionary or HDF5 file).
        Let `nf` denote the number of Fock spin orbitals (i.e. the sum of both
        the alpha and the beta orbitals) and `nb` the number of basis functions.
        With `array` we indicate either a `np.array` or an HDF5 dataset.
        The following keys are required in the container:

        1. **restricted** (`bool`): `True` for a restricted SCF calculation,
           `False` otherwise
        2. **conv_tol** (`float`): Tolerance value used for SCF convergence,
           should be roughly equivalent to l2 norm of the Pulay error.
        3. **orbcoeff_fb** (`array` with dtype `float`, size `(nf, nb)`):
           SCF orbital coefficients, i.e. the uniform transform from the basis
           to the molecular orbitals.
        4. **occupation_f** (`array` with dtype `float`, size `(nf, )`):
           Occupation number for each SCF orbitals (i.e. diagonal of the HF
           density matrix in the SCF orbital basis).
        5. **orben_f** (`array` with dtype `float`, size `(nf, )`):
           SCF orbital energies
        6. **fock_ff** (`array` with dtype `float`, size `(nf, nf)`):
           Fock matrix in SCF orbital basis. Notice, the full matrix is expected
           also for restricted calculations.
        7. **eri_phys_asym_ffff** (`array` with dtype `float`,
           size `(nf, nf, nf, nf)`): Antisymmetrised electron-repulsion integral
           tensor in the SCF orbital basis, using the Physicists' indexing
           convention, i.e. that the index tuple `(i,j,k,l)` refers to
           the integral :math:`\\langle ij || kl \\rangle`, i.e.

           .. math::
              \\int_\\Omega \\int_\\Omega d r_1 d r_2 \\frac{
              \\phi_i(r_1) \\phi_j(r_2)
              \\phi_k(r_1) \\phi_l(r_2)}{|r_1 - r_2|}
              - \\int_\\Omega \\int_\\Omega d r_1 d r_2 \\frac{
              \\phi_i(r_1) \\phi_j(r_2)
              \\phi_l(r_1) \\phi_k(r_2)}{|r_1 - r_2|}

           The full tensor (including zero blocks) is expected.

        As an alternative to `eri_phys_asym_ffff`, the user may provide

        8. **eri_ffff** (`array` with dtype `float`, size `(nf, nf, nf, nf)`):
           Electron-repulsion integral tensor in chemists' notation.
           The index tuple `(i,j,k,l)` thus refers to the integral
           :math:`(ij|kl)`, which is

           .. math::
              \\int_\\Omega \\int_\\Omega d r_1 d r_2
              \\frac{\\phi_i(r_1) \\phi_j(r_1)
              \\phi_k(r_2) \\phi_l(r_2)}{|r_1 - r_2|}

           Notice, that no antisymmetrisation has been applied in this tensor.

        The above keys define the least set of quantities to start a calculation
        in `adcc`. In order to have access to properties such as dipole moments
        or to get the correct state energies, further keys are highly
        recommended to be provided as well.

        9. **energy_scf** (`float`): Final total SCF energy of both electronic
           and nuclear energy terms.
        10. **nuclear_repulsion_energy** (`float`): The nuclear repulsion energy.
        11. **multipoles**: Container with electric and nuclear
            multipole moments. Can be another dictionary or simply an HDF5
            group.

              - **elec_1** (`array`, size `(3, nb, nb)`):
                Electric dipole moment integrals in the atomic orbital basis
                (i.e. the discretisation basis with `nb` elements). First axis
                indicates cartesian component (x, y, z).
              - **elec_2_{gauge_origin}** (`array`, size `(9, nb, nb)`):
                Electric quadrupole moment integrals in the atomic orbital basis
                for the named gauge origin (e.g. `elec_2_origin`,
                `elec_2_mass_center` or `elec_2_charge_center`). First axis
                indicates cartesian component (xx, xy, xz, yx, yy, yz, zx, zy, zz).
              - **nuclear_0** (`float`): Total nuclear charge
              - **nuclear_1** (`array` size `(3, )`): Nuclear dipole moment
              - **nuclear_2_{gauge_origin}** (`array` size `(6, )`): Nuclear
                quadrupole moment for the named gauge origin. Components are
                (xx, xy, xz, yy, yz, zz).

        12. **spin_multiplicity** (`int`): The spin multiplicity of the HF
            ground state described by the data. A value of `0` (for unknown)
            should be supplied for unrestricted calculations.
            (default: 0 for unrestricted calculations and (nalpha - nbeta + 1) for restricted)
        13. **overlap** (`array`, size `(nb, nb)`): Overlap matrix in the atomic
            orbital basis.
        14. **magnetic_moments**: Container with magnetic moment integrals.

              - **mag_1_{gauge_origin}** (`array`, size `(3, nb, nb)`):
                Imaginary part of the magnetic dipole integrals in the atomic
                orbital basis for the named gauge origin (e.g. `mag_1_origin`).
                First axis indicates cartesian component (x, y, z).

        15. **derivatives**: Container with derivative integrals.

              - **elec_vel_1** (`array`, size `(3, nb, nb)`):
                Imaginary part of the electric dipole integrals in the velocity
                gauge in the atomic orbital basis. First axis indicates
                cartesian component (x, y, z).

        A descriptive string for the backend can be supplied optionally as well.
        In case of using a python `dict` as the data container, this should be
        done using the key `backend`. For an HDF5 file, this should be done
        using the attribute `backend`. Defaults based on the filename are
        generated.

        Parameters
        ----------
        data : Mapping[str, Any]
            Data container (e.g. a python dict or h5py.File) containing the
            HartreeFock data to use. For the required keys see details above.
        """

        # Do not forget the next line, otherwise weird errors result
        super().__init__()
        if not isinstance(data, Mapping):
            raise TypeError("The data container has to a Mapping like 'dict' or 'h5py.File'.")
        self._data: Mapping[str, Any] = data

        # Setup integral data. The provider locates and validates the integrals
        # in the data container itself, see DataOperatorIntegralProvider.
        self.operator_integral_provider = DataOperatorIntegralProvider(
            data, n_bas=self.get_n_bas(), backend=self.get_backend()
        )

    #
    # Required keys
    #
    def get_restricted(self) -> bool:
        return _load_scalar_value(self._data, key="restricted", scalar_cls=bool)

    def get_conv_tol(self) -> float:
        return _load_scalar_value(self._data, key="conv_tol", scalar_cls=float)

    def _occupation_f(self) -> ArrayHandle:
        nf = 2 * self.get_n_orbs_alpha()
        return _get_array(self._data, key="occupation_f", shape=(nf,))

    def fill_occupation_f(self, out: Array1D) -> None:
        out[:] = self._occupation_f()

    def _orbcoeff_fb(self) -> ArrayHandle:
        # there is no point in validating the shape, since orbcoeff_fb is used to determine
        # the number of alpha orbitals and basis functions anyway.
        # But ensure that the array is 2 dimensional!
        res = _get_array(self._data, key="orbcoeff_fb")
        if len(res.shape) != 2:
            raise ValueError(
                "'orbcoeff_fb' has to be a 2 dimensional array. Found array with "
                f"shape {res.shape} under key 'orbcoeff_fb'."
            )
        return res

    def fill_orbcoeff_fb(self, out: Array2D) -> None:
        out[:] = self._orbcoeff_fb()

    def fill_orben_f(self, out: Array1D) -> None:
        nf = 2 * self.get_n_orbs_alpha()
        out[:] = _load_array_value(self._data, key="orben_f", shape=(nf,))

    def fill_fock_ff(self, slices: Slices2D, out: Array2D) -> None:
        # avoid loading the full array and instead slice the handle
        nf = 2 * self.get_n_orbs_alpha()
        out[:] = _get_array(self._data, key="fock_ff", shape=(nf, nf))[slices]

    def fill_eri_ffff(self, slices: Slices4D, out: Array4D) -> None:
        # avoid loading the full array and instead slice the handle
        nf = 2 * self.get_n_orbs_alpha()
        out[:] = _get_array(self._data, key="eri_ffff", shape=(nf, nf, nf, nf))[slices]

    def fill_eri_phys_asym_ffff(self, slices: Slices4D, out: Array4D) -> None:
        # Only required if eri_ffff not provided
        # avoid loading the full array and instead slice the handle
        nf = 2 * self.get_n_orbs_alpha()
        out[:] = _get_array(self._data, key="eri_phys_asym_ffff", shape=(nf, nf, nf, nf))[slices]

    #
    # Recommended keys
    #
    def get_backend(self) -> str:
        # try to load from attrs for h5py.File
        if isinstance(self._data, h5py.File):
            try:
                return _load_scalar_value(self._data.attrs, key="backend", scalar_cls=str)
            except MissingDataError:
                pass
        # lookup in the mapping or return some fallback default
        try:
            return _load_scalar_value(self._data, key="backend", scalar_cls=str)
        except MissingDataError:
            if isinstance(self._data, h5py.File):
                return f'<HDF5 file "{self._data.filename}">'
            else:
                return self._data.__class__.__name__

    def get_energy_scf(self) -> float:
        return _load_scalar_value(self._data, key="energy_scf", scalar_cls=float)

    def get_nuclear_repulsion_energy(self) -> float:
        return _load_scalar_value(self._data, key="nuclear_repulsion_energy", scalar_cls=float)

    def get_nuclear_multipole(
        self, order: int, gauge_origin: Coordinate = (0.0, 0.0, 0.0)
    ) -> Array1D:
        key = f"multipoles/nuclear_{order}"
        if order == 0:
            charge = _load_scalar_value(self._data, key=key, scalar_cls=float)
            return np.array([charge])
        elif order == 1:
            return _load_array_value(self._data, key=key, shape=(3,))
        elif order == 2:
            # TODO: how to format a coordinate origin in the key?
            # I think this currently assumes a str origin, which contradicts the interface
            key += f"_{gauge_origin}"
            return _load_array_value(self._data, key=key, shape=(6,))
        raise NotImplementedError(f"Nuclear multipole with order {order} is not available.")

    def get_spin_multiplicity(self) -> int:
        try:
            return _load_scalar_value(self._data, key="spin_multiplicity", scalar_cls=int)
        except MissingDataError:
            # this should be catched already on the C++ side
            if not self.get_restricted():
                raise ValueError(
                    "'spin_multiplicity' not provided in the data. Can only be determined "
                    "automatically for a restricted reference."
                )
            noa = self.get_n_orbs_alpha()
            occupations = self._occupation_f()
            # use round instead of plain int cast to obtain 4.999 -> 5
            return round(np.sum(occupations[:noa])) - round(np.sum(occupations[noa:])) + 1

    #
    # Deduced keys
    #
    def get_n_orbs_alpha(self) -> int:
        nf = self._orbcoeff_fb().shape[0]
        if nf % 2:
            raise ValueError(
                f"First axis of 'orbcoeff_fb' should have even length, got length {nf}."
            )
        return nf // 2

    def get_n_bas(self) -> int:
        return self._orbcoeff_fb().shape[1]

    def has_eri_phys_asym_ffff(self) -> bool:
        return _load_from_data(self._data, key="eri_phys_asym_ffff", default=None) is not None
