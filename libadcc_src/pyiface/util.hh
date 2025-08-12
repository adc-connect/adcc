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
#include "..//Tensor.hh"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace libadcc {

namespace py = pybind11;

/** Make a py::tuple from a vector representing the shape */
py::tuple shape_tuple(const std::vector<size_t>& shape);

/** Convert a list of tensors to a vector of shared pointers to Tensor */
template <typename Listlike>
std::vector<std::shared_ptr<Tensor>> extract_tensors(const Listlike& in);

}  // namespace libadcc
