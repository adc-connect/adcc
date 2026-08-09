//
// Copyright (C) 2026 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#pragma once
#include "../exceptions.hh"
#include <cstddef>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <string>

namespace libadcc {

namespace py = pybind11;

// Small wrapper around py::array_t to enable the correct
// type hints for the dimensionality of numpy arrays in the stub file.
// Important difference: In contrast to py::array_t<T> other types are not
// silently casted (and copied) to T if necessary (e.g. int -> double).
// Instead this class throws if it is constructed from e.g. std::array_t<int>
template <typename T, size_t Ndim>
class NDArray : public py::array_t<T> {
  static_assert(Ndim > 0, "NDArray requires at least one dimension");

 public:
  // Needed by the type caster: it builds the value via reinterpret_borrow /
  // reinterpret_steal, which require the (handle, borrowed_t/stolen_t) ctors.
  using py::array_t<T>::array_t;

  // Converting constructor.
  NDArray(py::array array)
        : py::array_t<T>(py::reinterpret_borrow<py::array_t<T>>(validate(array))) {}

  // Hides py::array_t<T>::check_, which only validates the dtype. isinstance<NDArray>
  // dispatches to this, i.e., the type caster (which builds the value through
  // reinterpret_borrow and therefore bypasses the converting constructor above)
  // rejects arrays of the wrong dimensionality.
  static bool check_(py::handle h) {
    return py::array_t<T>::check_(h) &&
           py::reinterpret_borrow<py::array>(h).ndim() == static_cast<py::ssize_t>(Ndim);
  }

 private:
  static const py::array& validate(const py::array& array) {
    if (!py::isinstance<py::array_t<T>>(array)) throw runtime_error("Invalid array type");
    if (array.ndim() != static_cast<py::ssize_t>(Ndim)) {
      throw dimension_mismatch("Expected a " + std::to_string(Ndim) +
                               "-dimensional array, but got " +
                               std::to_string(array.ndim()) + " dimensions.");
    }
    return array;
  }
};
}  // namespace libadcc

namespace pybind11 {
namespace detail {

// Create a string of Ndim "int, int, ..." as shape for the np.ndarray type hint
template <size_t Ndim>
struct NdimName {
  // NdimName<0> would underflow to NdimName<SIZE_MAX> (Ndim == 1 is specialized below)
  static_assert(Ndim > 1, "NDArray requires at least one dimension");
  static constexpr auto value = NdimName<Ndim - 1>::value + const_name(", int");
};
template <>
struct NdimName<1> {
  static constexpr auto value = const_name("int");
};

template <typename T, std::size_t Ndim>
struct handle_type_name<libadcc::NDArray<T, Ndim>> {
  static constexpr auto name = const_name("numpy.ndarray[tuple[") +
                               NdimName<Ndim>::value + const_name("], numpy.dtype[") +
                               npy_format_descriptor<T>::name + const_name("]]");
};

}  // namespace detail
}  // namespace pybind11
