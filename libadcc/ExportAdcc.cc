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

#include <adcc/backend.hh>
#include <adcc/exceptions.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adcc {
namespace py_iface {
void export_AdcMemory(py::module& m);
void export_amplitude_vector_enforce_spin_kind(py::module& m);
void export_guesses(py::module& m);
void export_HartreeFockProvider(py::module& m);
void export_MoIndexTranslation(py::module& m);
void export_MoSpaces(py::module& m);
void export_OneParticleOperator(py::module& m);
void export_ReferenceState(py::module& m);
void export_Symmetry(py::module& m);
void export_Tensor(py::module& m);
void export_threading(py::module& m);
}  // namespace py_iface
}  // namespace adcc

PYBIND11_MODULE(libadcc, m) {
  namespace pyif = adcc::py_iface;

  pyif::export_AdcMemory(m);
  pyif::export_threading(m);
  pyif::export_HartreeFockProvider(m);
  pyif::export_MoSpaces(m);
  pyif::export_Symmetry(m);
  pyif::export_MoIndexTranslation(m);
  pyif::export_Tensor(m);
  pyif::export_ReferenceState(m);

  pyif::export_OneParticleOperator(m);
  pyif::export_guesses(m);
  pyif::export_amplitude_vector_enforce_spin_kind(m);

  // Set metadata about libtensor
  py::dict tensor_backend;
  const adcc::TensorBackend& back = adcc::tensor_backend();
  tensor_backend["name"]          = back.name;
  tensor_backend["version"]       = back.version;
  tensor_backend["authors"]       = back.authors;
  tensor_backend["features"]      = back.features;
  tensor_backend["blas"]          = back.blas;
  m.attr("__backend__")           = tensor_backend;

  // Exception translation
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const adcc::not_implemented_error& ex) {
      PyErr_SetString(PyExc_NotImplementedError, ex.what());
    } catch (const adcc::invalid_argument& ex) {
      PyErr_SetString(PyExc_ValueError, ex.what());
    }
  });
}
