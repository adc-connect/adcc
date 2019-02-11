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
#include <adcc/OneParticleDensityMatrix.hh>
#include <adcc/ReferenceState.hh>
#include <adcc/exceptions.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::string OneParticleDensityMatrix___repr__(
      const OneParticleDensityMatrix& self) {
  std::stringstream ss;
  ss << "OneParticleDensityMatrix(";
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

static size_t OneParticleDensityMatrix___len__(const OneParticleDensityMatrix& self) {
  return self.shape()[0];
}

static py::tuple OneParticleDensityMatrix_shape(const OneParticleDensityMatrix& self) {
  return shape_tuple(self.shape());
}

static py::array OneParticleDensityMatrix__to_ndarray(
      const OneParticleDensityMatrix& self) {
  // Get an empty array of the required shape and export the data into it.
  py::array_t<scalar_type> res(self.shape());
  self.export_to(res.mutable_data(), self.size());
  return res;
}

std::shared_ptr<Tensor> OneParticleDensityMatrix___getitem__(
      const OneParticleDensityMatrix& self, std::string block) {
  return self.block(block);
}

std::shared_ptr<Tensor> OneParticleDensityMatrix__block(
      const OneParticleDensityMatrix& self, std::string block) {
  return self.block(block);
}

py::list OneParticleDensityMatrix__blocks(const OneParticleDensityMatrix& self) {
  py::list ret;
  for (auto& b : self.blocks()) {
    ret.append(b);
  }
  return ret;
}

py::list OneParticleDensityMatrix__orbital_subspaces(
      const OneParticleDensityMatrix& self) {
  py::list ret;
  for (auto& b : self.orbital_subspaces()) {
    ret.append(b);
  }
  return ret;
}

py::tuple OneParticleDensityMatrix__transform_to_ao_basis_ref(
      const OneParticleDensityMatrix& self, std::shared_ptr<ReferenceState> ref_ptr) {
  auto ret = self.transform_to_ao_basis(ref_ptr);
  return py::make_tuple(ret.first, ret.second);
}

py::tuple OneParticleDensityMatrix__transform_to_ao_basis_coeff(
      const OneParticleDensityMatrix& self, py::dict coeff_ptrs) {

  std::map<std::string, std::shared_ptr<Tensor>> coefficient_ptrs;
  for (auto pair : coeff_ptrs) {
    const std::string key = pair.first.cast<std::string>();
    coefficient_ptrs[key] = pair.second.cast<std::shared_ptr<Tensor>>();
  }
  auto ret = self.transform_to_ao_basis(coefficient_ptrs);
  return py::make_tuple(ret.first, ret.second);
}

/** Exports adcc/OneParticleDensityMatrix.hh to python */
void export_OneParticleDensityMatrix(py::module& m) {
  void (OneParticleDensityMatrix::*set_zero_block_1)(std::string) =
        &OneParticleDensityMatrix::set_zero_block;

  py::class_<OneParticleDensityMatrix, std::shared_ptr<OneParticleDensityMatrix>>(
        m, "OneParticleDensityMatrix",
        "Class representing a one-particle (transition) density matrix.")
        .def_property_readonly("ndim", &OneParticleDensityMatrix::ndim)
        .def_property_readonly("shape", &OneParticleDensityMatrix_shape)
        .def_property_readonly("size", &OneParticleDensityMatrix::size)
        .def("block", &OneParticleDensityMatrix__block,
             "Obtain a block from the matrix (e.g. o1o1, o1v1)")
        .def("set_block", &OneParticleDensityMatrix::set_block,
             "Set a block of the matrix (e.g. o1o1, o1v1)")
        .def("set_zero_block", set_zero_block_1,
             "Set a block of the matrix (e.g. o1o1, o1v1) to be explicitly zero.")
        .def("is_zero_block", &OneParticleDensityMatrix::is_zero_block)
        .def("empty_like", &OneParticleDensityMatrix::empty_like)
        .def_property_readonly("is_symmetric", &OneParticleDensityMatrix::is_symmetric)
        .def_property_readonly("blocks", &OneParticleDensityMatrix__blocks)
        .def_property_readonly("orbital_subspaces",
                               &OneParticleDensityMatrix__orbital_subspaces)
        .def("has_block", &OneParticleDensityMatrix::has_block)
        .def("transform_to_ao_basis",
             &OneParticleDensityMatrix__transform_to_ao_basis_ref)
        .def("transform_to_ao_basis",
             &OneParticleDensityMatrix__transform_to_ao_basis_coeff)
        .def("to_ndarray", &OneParticleDensityMatrix__to_ndarray,
             "Return the density matrix in MOs as a full, non-sparse numpy array.")
        .def("__getitem__", &OneParticleDensityMatrix___getitem__)
        .def("__repr__", &OneParticleDensityMatrix___repr__)
        .def("__len__", &OneParticleDensityMatrix___len__)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
