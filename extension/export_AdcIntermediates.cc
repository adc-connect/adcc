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

#include "convert_timer.hh"
#include <adcc/AdcIntermediates.hh>
#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::string AdcIntermediates__repr__(const AdcIntermediates& self) {
  std::stringstream ss;
  ss << "AdcIntermediates(" << self << ")";
  return ss.str();
}

void export_AdcIntermediates(py::module& m) {

  py::class_<AdcIntermediates, std::shared_ptr<AdcIntermediates>>(
        m, "AdcIntermediates",
        "Class holding the computed ADC intermediates. Python binding to "
        ":cpp:class:`adcc::AdcIntermediates`.")
        .def(py::init<std::shared_ptr<const LazyMp>>())
        .def_property_readonly("adc2_i1", &AdcIntermediates::compute_adc2_i1)
        .def_property_readonly("adc2_i2", &AdcIntermediates::compute_adc2_i2)
        .def_property_readonly("adc3_m11", &AdcIntermediates::compute_adc3_m11)
        .def_property_readonly("adc3_pia", &AdcIntermediates::compute_adc3_pia)
        .def_property_readonly("adc3_pib", &AdcIntermediates::compute_adc3_pib)
        .def_property_readonly("cv_p_oo", &AdcIntermediates::compute_cv_p_oo)
        .def_property_readonly("cv_p_ov", &AdcIntermediates::compute_cv_p_ov)
        .def_property_readonly("cv_p_vv", &AdcIntermediates::compute_cv_p_vv)
        .def_property_readonly("cvs_adc3_m11", &AdcIntermediates::compute_cvs_adc3_m11)
        .def("__repr__", &AdcIntermediates__repr__)
        .def_property_readonly(
              "timer",
              [](const AdcIntermediates& self) { return convert_timer(self.timer()); },
              "Obtain the timer object of this class.")
        //
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
