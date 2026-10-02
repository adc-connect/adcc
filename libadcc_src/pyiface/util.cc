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

#include "util.hh"
#include "../exceptions.hh"

namespace libadcc {

namespace py = pybind11;

py::typing::Tuple<size_t, py::ellipsis> shape_tuple(const std::vector<size_t>& shape) {
  switch (shape.size()) {
    case 0:
      throw runtime_error("Encountered unexpected dimensionality 0.");
    case 1:
      return py::make_tuple(shape[0]);
    case 2:
      return py::make_tuple(shape[0], shape[1]);
    case 3:
      return py::make_tuple(shape[0], shape[1], shape[2]);
    case 4:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3]);
    case 5:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4]);
    case 6:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    case 7:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                            shape[6]);
    case 8:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                            shape[6], shape[7]);
    default:
      throw not_implemented_error(
            "shape_tuple only implemented up to dimensionality 8 so far.");
      // TensorImpl is only implemented up to 4 indices so far
      // libtensor only supports up to 8 indices at the moment
  }
}

std::vector<std::shared_ptr<Tensor>> extract_tensors(const py::sequence& in) {
  std::vector<std::shared_ptr<Tensor>> ret;
  ret.reserve(in.size());
  for (py::object elem : in) {
    if (!py::isinstance<Tensor>(elem)) {
      throw py::type_error(
            "Expected a sequence of tensors but one element is of type '" +
            py::str(py::type::of(elem).attr("__name__")).cast<std::string>() + "'.");
    }
    ret.push_back(elem.cast<std::shared_ptr<Tensor>>());
  }
  return ret;
}

std::vector<size_t> extract_axes(py::handle axes) {
  if (!py::isinstance<Sequence<py::ssize_t>>(axes)) {
    throw py::type_error(
          "Expected a sequence of axes, but got " +
          py::str(py::type::of(axes).attr("__name__")).cast<std::string>() + ".");
  }
  auto axes_as_seq = py::reinterpret_borrow<py::sequence>(axes);
  std::vector<size_t> ret;
  ret.reserve(axes_as_seq.size());
  for (py::object elem : axes_as_seq) {
    try {
      ret.push_back(elem.cast<size_t>());
    } catch (const py::cast_error&) {
      throw py::type_error(
            "Expected a sequence of non-negative integers as axes, but got " +
            py::repr(elem).cast<std::string>() + ".");
    }
  }
  return ret;
}

}  // namespace libadcc
