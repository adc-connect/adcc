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

#include <adcc/AdcMemory.hh>
#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::string AdcMemory___repr__(const AdcMemory& self) {
  std::stringstream ss;

  ss << "AdcMemory(";
  if (self.use_std_allocator()) {
    ss << "use_std_allocator=True";
  } else {
    double mem_mb = static_cast<double>(self.max_memory()) / 1024. / 1024.;
    ss << "max_memory=" << mem_mb << "MiB, tbs_param=" << self.tbs_param()
       << ", pagefile_directory=" << self.pagefile_directory();
  }
  ss << ")";
  return ss.str();
}

/** Export adcc/AdcMemory.hh */
void export_AdcMemory(py::module& m) {
  // boost::noncopyable
  py::class_<AdcMemory, std::shared_ptr<AdcMemory>>(
        m, "AdcMemory",
        "Class controlling the memory allocations for adcc ADC calculations.")
        .def(py::init<>())
        .def_property_readonly("max_memory", &AdcMemory::max_memory)
        .def_property_readonly("pagefile_directory", &AdcMemory::pagefile_directory)
        .def_property_readonly("tbs_param", &AdcMemory::tbs_param)
        .def_property_readonly("use_std_allocator", &AdcMemory::use_std_allocator)
        .def("initialise", &AdcMemory::initialise,
             "Initialise the AdcMemory memory management. The parameters are "
             "pagefile_directory, "
             "max_memory, tbs_param")
        .def("__repr__", &AdcMemory___repr__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
