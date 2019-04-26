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
#include <adcc/solve_adcman_davidson.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

#ifdef ADCC_WITH_ADCMAN
py::array AdcmanDavidsonState__eigenvalues(const py::object& obj) {
  const AdcmanDavidsonState& self = obj.cast<const AdcmanDavidsonState&>();
  return make_array(self.eigenvalues.data(), {self.eigenvalues.size()}, obj);
}

py::array AdcmanDavidsonState__residual_norms(const py::object& obj) {
  const AdcmanDavidsonState& self = obj.cast<const AdcmanDavidsonState&>();
  return make_array(self.residual_norms.data(), {self.residual_norms.size()}, obj);
}

py::array AdcmanDavidsonState__residuals_converged(const py::object& obj) {
  const AdcmanDavidsonState& self = obj.cast<const AdcmanDavidsonState&>();
  py::array ret = py::array_t<bool>(py::make_tuple(self.residuals_converged.size()));
  std::copy(self.residuals_converged.begin(), self.residuals_converged.end(),
            reinterpret_cast<bool*>(ret.mutable_data()));
  return ret;
}

py::list AdcmanDavidsonState__singles_block(const AdcmanDavidsonState& self) {
  py::list ret;
  for (auto& v : self.singles_block) ret.append(v);
  return ret;
}

py::list AdcmanDavidsonState__doubles_block(const AdcmanDavidsonState& self) {
  py::list ret;
  for (auto& v : self.doubles_block) ret.append(v);
  return ret;
}

void export_solve_adcman_davidson(py::module& m) {
  py::class_<AdcmanDavidsonState, std::shared_ptr<AdcmanDavidsonState>>(
        m, "AdcmanDavidsonState", "Class to hold the results from an ADCman davidson",
        // The following is needed to dynamically add attributes to the
        // AdcmanDavidsonState later
        py::dynamic_attr())
        .def(py::init<>())
        .def_readonly("kind", &AdcmanDavidsonState::kind)
        .def_property_readonly("eigenvalues", &AdcmanDavidsonState__eigenvalues)
        .def_property_readonly("residual_norms", &AdcmanDavidsonState__residual_norms)
        .def_property_readonly("residuals_converged",
                               &AdcmanDavidsonState__residuals_converged)
        .def_property_readonly("singles_block", &AdcmanDavidsonState__singles_block)
        .def_property_readonly("doubles_block", &AdcmanDavidsonState__doubles_block)
        .def_readonly("ctx", &AdcmanDavidsonState::ctx_ptr)
        //
        ;

  m.def("solve_adcman_davidson", &solve_adcman_davidson,
        "Run the davidson solver in adcman and print the results.");
}
#endif  // ADCC_WITH_ADCMAN

}  // namespace py_iface
}  // namespace adcc
