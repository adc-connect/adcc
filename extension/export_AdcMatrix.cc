//
// Copyright (C) 2018 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include "util.hh"
#include <adcc/AdcMatrix.hh>
#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static py::tuple AdcMatrix_shape(const AdcMatrix& self) {
  return shape_tuple(self.shape());
}

static size_t AdcMatrix__len__(const AdcMatrix& self) { return self.shape()[0]; }

static py::list AdcMatrix_blocks(const AdcMatrix& self) {
  py::list ret;
  for (auto& str : self.blocks()) {
    ret.append(str);
  }
  return ret;
}

/** Exports adcc/AdcMatrix.hh to python */
void export_AdcMatrix(py::module& m) {
  py::class_<AdcMatrix, std::shared_ptr<AdcMatrix>>(
        m, "AdcMatrix", "Class representing the AdcMatrix in various variants.")
        .def(py::init<std::string, std::shared_ptr<LazyMp>>())
        .def_property("intermediates", &AdcMatrix::get_intermediates_ptr,
                      &AdcMatrix::set_intermediates_ptr)
        .def_property_readonly("reference_state", &AdcMatrix::reference_state_ptr)
        .def_property_readonly("ground_state", &AdcMatrix::ground_state_ptr)
        .def("compute_apply", &AdcMatrix::compute_apply)
        .def("diagonal", &AdcMatrix::diagonal)
        .def("has_block", &AdcMatrix::has_block)
        .def_property_readonly("shape", &AdcMatrix_shape)
        .def("block_spaces", &AdcMatrix::block_spaces)
        .def("__len__", &AdcMatrix__len__)
        .def_property_readonly("blocks", &AdcMatrix_blocks)
        .def("compute_matvec", &AdcMatrix::compute_matvec)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
