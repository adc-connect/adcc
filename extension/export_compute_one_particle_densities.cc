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
#include <adcc/compute_gs2state_optdm.hh>
#include <adcc/compute_state2state_optdm.hh>
#include <adcc/compute_state_diffdm.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

/** Exports adcc/compute_one_particle_densities.hh to python */
void export_compute_one_particle_densities(py::module& m) {
  m.def("compute_state_diffdm", &adcc::compute_state_diffdm);
  m.def("compute_gs2state_optdm", &adcc::compute_gs2state_optdm);
  m.def("compute_state2state_optdm", &adcc::compute_state2state_optdm);
}

}  // namespace py_iface
}  // namespace adcc
