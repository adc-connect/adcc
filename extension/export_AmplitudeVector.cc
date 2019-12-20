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
#include <adcc/AmplitudeVector.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {
namespace py = pybind11;

static std::shared_ptr<AmplitudeVector> makeAmplitudeVector(py::tuple in) {
  return std::make_shared<AmplitudeVector>(extract_tensors(in));
}

static void AmplitudeVector_set_from_tuple(AmplitudeVector& self, py::tuple in) {
  self.resize(py::len(in));
  for (size_t i = 0; i < py::len(in); ++i) {
    self[i] = in[i].cast<std::shared_ptr<Tensor>>();
  }
}

static py::tuple AmplitudeVector_to_tuple(const AmplitudeVector& self) {
  switch (self.size()) {
    case 0:
      return py::make_tuple();
    case 1:
      return py::make_tuple(self[0]);
    case 2:
      return py::make_tuple(self[0], self[1]);
    case 3:
      return py::make_tuple(self[0], self[1], self[2]);
    default:
      throw not_implemented_error(
            "AmplitudeVector_to_tuple only implemented up to block size 3 so far.");
  }
}

static size_t AmplitudeVector___len__(const AmplitudeVector& self) { return self.size(); }

/** Exports adcc/AmplitudeVector.hh to python */
void export_AmplitudeVector(py::module& m) {

  py::class_<AmplitudeVector, std::shared_ptr<AmplitudeVector>>(
        m, "AmplitudeVector",
        "Class representing an AmplitudeVector. Python binding to "
        ":cpp:class:`adcc::AmplitudeVector`.")
        .def(py::init(&makeAmplitudeVector))
        .def("set_from_tuple", &AmplitudeVector_set_from_tuple)
        .def("to_tuple", &AmplitudeVector_to_tuple)
        .def("__len__", &AmplitudeVector___len__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
