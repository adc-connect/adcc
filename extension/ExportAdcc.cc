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

#include <adcc/exceptions.hh>
#include <adcc/version.hh>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace adcc {
namespace py_iface {
void export_AdcIntermediates(py::module& m);
void export_AdcMatrix(py::module& m);
void export_AdcMemory(py::module& m);
void export_AmplitudeVector(py::module& m);
void export_CtxMap(py::module& m);
void export_HartreeFockProvider(py::module& m);
void export_HfData(py::module& m);
void export_LazyMp(py::module& m);
void export_OneParticleDensityMatrix(py::module& m);
void export_ReferenceState(py::module& m);
void export_Tensor(py::module& m);
void export_ThreadPool(py::module& m);
void export_amplitude_vector_enforce_spin_kind(py::module& m);
void export_compute_one_particle_densities(py::module& m);
void export_solve_adcman_davidson(py::module& m);
void export_tmp_run_prelim(py::module& m);
}  // namespace py_iface
}  // namespace adcc

PYBIND11_MODULE(libadcc, m) {
  namespace bpif = adcc::py_iface;

  bpif::export_CtxMap(m);
  bpif::export_AdcMemory(m);
  bpif::export_ThreadPool(m);
  bpif::export_HartreeFockProvider(m);
  bpif::export_HfData(m);

  bpif::export_Tensor(m);
  bpif::export_ReferenceState(m);
  bpif::export_OneParticleDensityMatrix(m);
  bpif::export_LazyMp(m);
  bpif::export_AdcIntermediates(m);
  bpif::export_AmplitudeVector(m);
  bpif::export_AdcMatrix(m);
  bpif::export_amplitude_vector_enforce_spin_kind(m);
  bpif::export_compute_one_particle_densities(m);

  bpif::export_tmp_run_prelim(m);
  bpif::export_solve_adcman_davidson(m);

  // Set metadata about libadcc
  m.attr("__version__")    = adcc::version::version_string();
  m.attr("__build_type__") = adcc::version::is_debug() ? "Debug" : "Release";

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const adcc::not_implemented_error& ex) {
      PyErr_SetString(PyExc_NotImplementedError, ex.what());
    }
  });
}
