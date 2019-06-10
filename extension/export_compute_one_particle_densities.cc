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
#include <adcc/compute_gs2state_optdm.hh>
#include <adcc/compute_state2state_optdm.hh>
#include <adcc/compute_state_diffdm.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::shared_ptr<OneParticleDensityMatrix> compute_state_diffdm(
      std::string method, std::shared_ptr<const LazyMp> ground_state_ptr,
      const AmplitudeVector& excitation_amplitude,
      std::shared_ptr<AdcIntermediates> intermediates_ptr) {
  return adcc::compute_state_diffdm(method, ground_state_ptr, excitation_amplitude,
                                    intermediates_ptr);
}

static std::shared_ptr<OneParticleDensityMatrix> compute_gs2state_optdm(
      std::string method, std::shared_ptr<const LazyMp> ground_state_ptr,
      const AmplitudeVector& excitation_amplitude,
      std::shared_ptr<AdcIntermediates> intermediates_ptr) {
  return adcc::compute_gs2state_optdm(method, ground_state_ptr, excitation_amplitude,
                                      intermediates_ptr);
}

static std::shared_ptr<OneParticleDensityMatrix> compute_state2state_optdm(
      std::string method, std::shared_ptr<const LazyMp> ground_state_ptr,
      const AmplitudeVector& excitation_amplitude_from,
      const AmplitudeVector& excitation_amplitude_to,
      std::shared_ptr<AdcIntermediates> intermediates_ptr) {
  return adcc::compute_state2state_optdm(method, ground_state_ptr,
                                         excitation_amplitude_from,
                                         excitation_amplitude_to, intermediates_ptr);
}

/** Exports adcc/compute_one_particle_densities.hh to python */
void export_compute_one_particle_densities(py::module& m) {
  m.def("compute_state_diffdm", &compute_state_diffdm);
  m.def("compute_gs2state_optdm", &compute_gs2state_optdm);
  m.def("compute_state2state_optdm", &compute_state2state_optdm);
}

}  // namespace py_iface
}  // namespace adcc
