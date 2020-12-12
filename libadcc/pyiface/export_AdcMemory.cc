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

#include "../AdcMemory.hh"
#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>

namespace libadcc {

using namespace pybind11::literals;
namespace py = pybind11;

static std::string AdcMemory___repr__(const AdcMemory& self) {
  std::stringstream ss;

  ss << "AdcMemory(allocator=" << self.allocator() << ", ";
  if (self.allocator() != "standard") {
    ss << "pagefile_directory=" << self.pagefile_directory() << ", ";
  }
  ss << "contraction_batch_size=" << self.contraction_batch_size() << ", ";
  ss << "max_block_size=" << self.max_block_size() << ")";
  return ss.str();
}

void export_AdcMemory(py::module& m) {
  py::class_<AdcMemory, std::shared_ptr<AdcMemory>>(
        m, "AdcMemory",
        "Class controlling the memory allocations for adcc ADC calculations. Python "
        "binding to :cpp:class:`libadcc::AdcMemory`.")
        .def(py::init<>())
        .def_property_readonly("allocator", &AdcMemory::allocator,
                               "Return the allocator to which the class is initialised.")
        .def_property_readonly("pagefile_directory", &AdcMemory::pagefile_directory,
                               "Return the pagefile_directory value:\nNote: This value "
                               "is only meaningful if allocator != \"standard\"")
        .def_property_readonly(
              "max_block_size", &AdcMemory::max_block_size,
              "Return the maximal block size a tenor may have along each axis.")
        .def_property("contraction_batch_size", &AdcMemory::contraction_batch_size,
                      &AdcMemory::set_contraction_batch_size,
                      "Get or set the batch size for contraction, i.e. the number of "
                      "elements handled simultaneously in a tensor contraction.")
        .def("initialise", &AdcMemory::initialise, "pagefile_directory"_a,
             "max_block_size"_a, "allocator"_a)
        .def("__repr__", &AdcMemory___repr__)
        //
        ;
}

}  // namespace libadcc
