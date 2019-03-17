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
        m, "AdcIntermediates", "Class holding the computed ADC intermediates.")
        .def(py::init<std::shared_ptr<LazyMp>>())
        .def_property_readonly("adc2_i1", &AdcIntermediates::cache_adc2_i1)
        .def_property_readonly("adc2_i2", &AdcIntermediates::cache_adc2_i2)
        .def_property_readonly("adc3_m11", &AdcIntermediates::cache_adc3_m11)
        .def_property_readonly("adc3_pia", &AdcIntermediates::cache_adc3_pia)
        .def_property_readonly("adc3_pib", &AdcIntermediates::cache_adc3_pib)
        .def_property_readonly("cv_p_oo", &AdcIntermediates::cache_cv_p_oo)
        .def_property_readonly("cv_p_ov", &AdcIntermediates::cache_cv_p_ov)
        .def_property_readonly("cv_p_vv", &AdcIntermediates::cache_cv_p_vv)
        .def("__repr__", &AdcIntermediates__repr__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
