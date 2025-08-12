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

#include "../backend.hh"
#include "../exceptions.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace libadcc {

void export_AdcMemory(py::module& m);
void export_adc_pp(py::module& m);
void export_HartreeFockProvider(py::module& m);
void export_MoIndexTranslation(py::module& m);
void export_MoSpaces(py::module& m);
void export_ReferenceState(py::module& m);
void export_Symmetry(py::module& m);
void export_Tensor(py::module& m);
void export_threading(py::module& m);

}  // namespace libadcc

PYBIND11_MODULE(libadcc, m) {
  libadcc::export_AdcMemory(m);
  libadcc::export_adc_pp(m);
  libadcc::export_HartreeFockProvider(m);
  libadcc::export_MoIndexTranslation(m);
  libadcc::export_MoSpaces(m);
  libadcc::export_ReferenceState(m);
  libadcc::export_Symmetry(m);
  libadcc::export_Tensor(m);
  libadcc::export_threading(m);

  // Set metadata about libtensor
  py::dict tensor_backend;
  const libadcc::TensorBackend& back = libadcc::tensor_backend();
  tensor_backend["name"]             = back.name;
  tensor_backend["version"]          = back.version;
  tensor_backend["authors"]          = back.authors;
  tensor_backend["features"]         = back.features;
  tensor_backend["blas"]             = back.blas;
  m.attr("__backend__")              = tensor_backend;

  // Exception translation
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const libadcc::not_implemented_error& ex) {
      PyErr_SetString(PyExc_NotImplementedError, ex.what());
    } catch (const libadcc::invalid_argument& ex) {
      PyErr_SetString(PyExc_ValueError, ex.what());
    }
  });
}
