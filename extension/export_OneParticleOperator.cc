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
#include <adcc/OneParticleOperator.hh>
#include <adcc/ReferenceState.hh>
#include <adcc/exceptions.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::string OneParticleOperator___repr__(const OneParticleOperator& self) {
  std::stringstream ss;
  ss << "OneParticleOperator(";
  ss << "is_symmetric=" << (self.is_symmetric() ? "true" : "false") << ", ";
  ss << "blocks=";
  bool first = true;
  for (auto& block : self.blocks()) {
    ss << (first ? "" : ",") << block;
    first = false;
  }
  ss << ")";
  return ss.str();
}

static size_t OneParticleOperator___len__(const OneParticleOperator& self) {
  return self.shape()[0];
}

static py::tuple OneParticleOperator_shape(const OneParticleOperator& self) {
  return shape_tuple(self.shape());
}

static py::array_t<scalar_type> OneParticleOperator__to_ndarray(
      const OneParticleOperator& self) {
  // Get an empty array of the required shape and export the data into it.
  py::array_t<scalar_type> res(self.shape());
  self.export_to(res.mutable_data(), self.size());
  return res;
}

py::tuple OneParticleOperator__to_ao_basis_ref(const OneParticleOperator& self,
                                               std::shared_ptr<ReferenceState> ref_ptr) {
  auto ret = self.to_ao_basis(ref_ptr);
  return py::make_tuple(ret.first, ret.second);
}

py::tuple OneParticleOperator__to_ao_basis_coeff(const OneParticleOperator& self,
                                                 py::dict coeff_ptrs) {
  std::map<std::string, std::shared_ptr<Tensor>> coefficient_ptrs;
  for (auto pair : coeff_ptrs) {
    const std::string key = pair.first.cast<std::string>();
    coefficient_ptrs[key] = pair.second.cast<std::shared_ptr<Tensor>>();
  }
  auto ret = self.to_ao_basis(coefficient_ptrs);
  return py::make_tuple(ret.first, ret.second);
}

/** Exports adcc/OneParticleOperator.hh to python */
void export_OneParticleOperator(py::module& m) {
  void (OneParticleOperator::*set_zero_block_1)(std::string) =
        &OneParticleOperator::set_zero_block;

  py::class_<OneParticleOperator, std::shared_ptr<OneParticleOperator>>(
        m, "OneParticleOperator",
        "Class representing a one-particle operator. Also used for one-particle"
        " (transition) density matrices. Python binding to "
        ":cpp:class:`adcc::OneParticleOperator`.")
        .def(py::init<std::shared_ptr<const MoSpaces>, bool, std::string>(),
             "Construct OneParticleOperator object. All blocks are "
             "initialised as zero blocks.\n"
             "\n"
             "mospaces       MoSpaces object\n"
             "is_symmetric    Is the operator symmetric\n"
             "cartesian transform    Symbol for cartesian transformation (see "
             "make_symmetry for details\n.")
        //
        .def_property_readonly("ndim", &OneParticleOperator::ndim)
        .def_property_readonly("shape", &OneParticleOperator_shape)
        .def_property_readonly("size", &OneParticleOperator::size)
        .def(
              "block",
              [](const OneParticleOperator& self, std::string block) {
                return self.block(block);
              },
              "Obtain a non-zero block from the matrix (e.g. o1o1, o1v1). If the block "
              "is "
              "a zero block, the function throws an ArgumentError.")
        .def("set_block", &OneParticleOperator::set_block,
             "Set a block of the matrix (e.g. o1o1, o1v1)")
        .def("set_zero_block", set_zero_block_1,
             "Set a block of the matrix (e.g. o1o1, o1v1) to be explicitly zero.")
        .def("is_zero_block", &OneParticleOperator::is_zero_block)
        .def("copy", &OneParticleOperator::copy,
             "Return a deep copy of this OneParticleOperator.\n\nThis can be used to "
             "conveniently add tensor corrections on top of existing OneParticleOperator "
             "objects.")
        .def_property_readonly("is_symmetric", &OneParticleOperator::is_symmetric)
        .def_property_readonly("cartesian_transform",
                               &OneParticleOperator::cartesian_transform)
        .def_property_readonly("blocks", &OneParticleOperator::blocks)
        .def_property_readonly("blocks_nonzero", &OneParticleOperator::blocks_nonzero)
        .def_property_readonly("mospaces", &OneParticleOperator::mospaces_ptr)
        .def("has_block", &OneParticleOperator::has_block)
        .def("to_ao_basis", &OneParticleOperator__to_ao_basis_ref)
        .def("to_ao_basis", &OneParticleOperator__to_ao_basis_coeff)
        .def("to_ndarray", &OneParticleOperator__to_ndarray,
             "Return the operator in MOs as a full, non-sparse numpy array.")
        .def("__getitem__",
             [](OneParticleOperator& self, std::string block) { return self[block]; })
        .def("__repr__", &OneParticleOperator___repr__)
        .def("__len__", &OneParticleOperator___len__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
