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

#include <adcc/AdcMemory.hh>
#include <memory>
#include <pybind11/pybind11.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::string AdcMemory___repr__(const AdcMemory& self) {
  std::stringstream ss;

  ss << "AdcMemory(allocator=" << self.allocator() << ", ";
  if (self.allocator() != "standard") {
    double mem_mb = static_cast<double>(self.max_memory()) / 1024. / 1024.;
    ss << "max_memory=" << mem_mb << "MiB, ";
    ss << "pagefile_directory=" << self.pagefile_directory() << ", ";
  }
  ss << "tensor_block_size=" << self.tbs_param();
  ss << ", contraction_batch_size=" << self.contraction_batch_size() << ")";
  return ss.str();
}

/** Export adcc/AdcMemory.hh */
void export_AdcMemory(py::module& m) {
  // boost::noncopyable
  py::class_<AdcMemory, std::shared_ptr<AdcMemory>>(
        m, "AdcMemory",
        "Class controlling the memory allocations for adcc ADC calculations. Python "
        "binding to "
        ":cpp:class:`adcc::AdcMemory`.")
        .def(py::init<>())
        .def_property_readonly("allocator", &AdcMemory::allocator,
                               "Return the allocator to which the class is initialised.")
        .def_property_readonly("max_memory", &AdcMemory::max_memory,
                               "Return the max_memory parameter value to which the class "
                               "was initialised.\nNote: This value is only a meaningful "
                               "upper bound if allocator != \"standard\"")
        .def_property_readonly("pagefile_directory", &AdcMemory::pagefile_directory,
                               "Return the pagefile_directory value:\nNote: This value "
                               "is only meaningful if allocator != \"standard\"")
        .def_property_readonly("tensor_block_size", &AdcMemory::tbs_param,
                               "Return the tensor_block_size parameter.")
        .def_property("contraction_batch_size", &AdcMemory::contraction_batch_size,
                      &AdcMemory::set_contraction_batch_size,
                      "Get or set the batch size for contraction, i.e. the number of "
                      "blocks handled simultaneously in a tensor contraction.")
        .def("initialise", &AdcMemory::initialise,
             "Initialise the adcc memory management.\n\n"
             "@param   max_memory   Estimate for the maximally employed memory\n"
             "@param tensor_block_size   This parameter roughly has the meaning\n"
             "                           of how many indices are handled together\n"
             "                           on operations. A good value is 16 for most\n"
             "                           nowaday CPU cachelines.\n"
             "@param pagefile_prefix     Directory prefix for storing temporary\n"
             "                           cache files.\n"
             "@param allocator   The allocator to be used. Valid values are \"libxm\",\n"
             "                   \"libvmm\", \"standard\" and \"default\", where "
             "\"default\"\n"
             "                   uses a default chosen from the first three.")
        .def("__repr__", &AdcMemory___repr__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
