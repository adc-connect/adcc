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

#include <adcc/exceptions.hh>
#include <adcc/metadata.hh>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace adcc {
namespace py_iface {
void export_AdcIntermediates(py::module& m);
void export_AdcMatrix(py::module& m);
void export_AdcMemory(py::module& m);
void export_amplitude_vector_enforce_spin_kind(py::module& m);
void export_AmplitudeVector(py::module& m);
void export_CachingPolicy(py::module& m);
void export_compute_modified_transition_moments(py::module& m);
void export_compute_one_particle_densities(py::module& m);
void export_guesses(py::module& m);
void export_HartreeFockProvider(py::module& m);
void export_LazyMp(py::module& m);
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
  pyif::export_CachingPolicy(m);

  pyif::export_HartreeFockProvider(m);
  pyif::export_MoSpaces(m);
  pyif::export_Symmetry(m);
  pyif::export_MoIndexTranslation(m);
  pyif::export_Tensor(m);
  pyif::export_ReferenceState(m);

  pyif::export_OneParticleOperator(m);
  pyif::export_LazyMp(m);
  pyif::export_AdcIntermediates(m);
  pyif::export_AmplitudeVector(m);
  pyif::export_AdcMatrix(m);
  pyif::export_guesses(m);
  pyif::export_amplitude_vector_enforce_spin_kind(m);
  pyif::export_compute_modified_transition_moments(m);
  pyif::export_compute_one_particle_densities(m);

  // Set metadata about libadcc
  m.attr("__version__")    = adcc::version::version_string();
  m.attr("__build_type__") = adcc::version::is_debug() ? "Debug" : "Release";
  m.attr("__authors__")    = adcc::__authors__();
  m.attr("__email__")      = adcc::__email__();

  // Set libadcc feature list
  py::list features;
  for (auto& feature : adcc::__features__()) features.append(feature);
  m.attr("__features__") = features;

  // Set libadcc components list
  py::list components;
  for (const adcc::Component& comp : adcc::__components__()) {
    py::dict d;
    d["name"]        = comp.name;
    d["version"]     = comp.version;
    d["description"] = comp.description;
    d["authors"]     = comp.authors;
    d["doi"]         = comp.doi;
    d["website"]     = comp.website;
    d["licence"]     = comp.licence;
    components.append(d);
  }
  m.attr("__components__") = components;

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
