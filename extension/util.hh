//
// Copyright (C) 2018 by the adcc authors
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
#include <adcc/Tensor.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

/** Make a py::tuple from a vector representing the shape */
py::tuple shape_tuple(const std::vector<size_t>& shape);

/** Check for the passed array shape and extract the array data pointer.
 *
 * \throws invalid_argument if the array size/dimension is not matching
 */
template <typename T>
T* extract_array_data(const py::array_t<T>& array, std::vector<size_t> shape);

/** Build an array using the provided data pointer
 *  and shape */
template <typename T>
py::array_t<T> make_array(T* data, std::vector<size_t> shape,
                          const py::handle& owner = py::handle());

/** Convert a list of tensors to a vector of shared pointers to Tensor */
template <typename Listlike>
std::vector<std::shared_ptr<Tensor>> extract_tensors(const Listlike& in);

/** Convert a vector of shared pointers of Tensor to a py::list */
py::list pack_tensors(const std::vector<std::shared_ptr<Tensor>>& list);

}  // namespace py_iface
}  // namespace adcc
