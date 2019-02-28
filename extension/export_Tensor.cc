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
#include <adcc/Tensor.hh>
#include <adcc/exceptions.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

static std::vector<std::vector<size_t>> parse_permutations(const py::list& permutations) {
  std::vector<std::vector<size_t>> vec_perms;
  for (auto tpl : permutations) {
    std::vector<size_t> perms;
    for (auto itm : tpl) {
      perms.push_back(itm.cast<size_t>());
    }
    vec_perms.push_back(perms);
  }
  return vec_perms;
}  // namespace py_iface

//
// Extra defs
//
static py::tuple Tensor_shape(const Tensor& self) { return shape_tuple(self.shape()); }

static py::array_t<scalar_type> Tensor_to_ndarray(const Tensor& self) {
  // Get an empty array of the required shape and export the data into it.
  py::array_t<scalar_type> res(self.shape());
  self.export_to(res.mutable_data(), self.size());
  return res;
}

static void Tensor_from_ndarray_tol(Tensor& self, py::array_t<scalar_type> in_array,
                                    double symmetry_tolerance) {
  py::ssize_t nd = in_array.ndim();
  if (nd < 1) throw invalid_argument("Cannot import from 0D array.");

  py::ssize_t pysize = 1;
  for (py::ssize_t i = 0; i < nd; ++i) pysize *= in_array.shape(i);

  const size_t size = static_cast<size_t>(pysize);
  self.import_from(in_array.data(), size, symmetry_tolerance);
}

static void Tensor_from_ndarray(Tensor& self, py::array in_array) {
  Tensor_from_ndarray_tol(self, in_array, 0.0);
}

static scalar_type Tensor_dot(const Tensor& self, std::shared_ptr<Tensor> other) {
  return self.dot(other);
}

static py::array Tensor_dot_list(const Tensor& self, py::list tensors) {
  std::vector<std::shared_ptr<Tensor>> parsed = extract_tensors(tensors);
  std::vector<scalar_type> dots               = self.dot(parsed);

  py::array_t<scalar_type> ret(dots.size());
  std::copy(dots.begin(), dots.end(), ret.mutable_data());

  return ret;
}

static std::shared_ptr<Tensor> Tensor_transpose_1(const Tensor& self) {
  return self.transpose();
}

static std::shared_ptr<Tensor> Tensor_transpose_2(const Tensor& self, py::tuple axes) {
  std::vector<size_t> vec_axes(py::len(axes));
  for (size_t i = 0; i < py::len(axes); ++i) {
    vec_axes[i] = axes[i].cast<size_t>();
  }
  return self.transpose(vec_axes);
}

static std::shared_ptr<Tensor> Tensor_add_linear_combination(
      std::shared_ptr<Tensor> self, py::array_t<scalar_type> coefficients,
      py::list tensors) {
  if (coefficients.ndim() != 1) {
    throw invalid_argument("coefficients array needs to have exactly one dimension.");
  }
  size_t in_size       = static_cast<size_t>(coefficients.shape(0));
  scalar_type* in_data = coefficients.mutable_data();
  std::vector<scalar_type> scalars(in_size);

  std::copy(in_data, in_data + in_size, scalars.data());
  std::vector<std::shared_ptr<Tensor>> parsed = extract_tensors(tensors);
  self->add(scalars, parsed);
  return self;
}

static void Tensor_symmetrise_to(const Tensor& self, std::shared_ptr<Tensor> other,
                                 py::list permutations) {
  self.symmetrise_to(other, parse_permutations(permutations));
}

static void Tensor_antisymmetrise_to(const Tensor& self, std::shared_ptr<Tensor> other,
                                     py::list permutations) {
  self.antisymmetrise_to(other, parse_permutations(permutations));
}

static std::shared_ptr<Tensor> multiply(std::shared_ptr<Tensor> a,
                                        std::shared_ptr<Tensor> b,
                                        std::shared_ptr<Tensor> out) {
  a->multiply_to(b, out);
  return out;
}

static std::shared_ptr<Tensor> divide(std::shared_ptr<Tensor> a,
                                      std::shared_ptr<Tensor> b,
                                      std::shared_ptr<Tensor> out) {
  a->divide_to(b, out);
  return out;
}

static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b,
                                   std::shared_ptr<Tensor> out) {
  a->copy_to(out);
  out->add(b);
  return out;
}

static std::shared_ptr<Tensor> subtract(std::shared_ptr<Tensor> a,
                                        std::shared_ptr<Tensor> b,
                                        std::shared_ptr<Tensor> out) {
  a->copy_to(out);
  out->add(-1.0, b);
  return out;
}

static std::shared_ptr<Tensor> contract_to(std::string contraction,
                                           std::shared_ptr<Tensor> a,
                                           std::shared_ptr<Tensor> b,
                                           std::shared_ptr<Tensor> out) {
  a->contract_to(contraction, b, out);
  return out;
}

//
// Implementation of python-side special functions
//    See https://docs.python.org/3/library/operator.html for details
//

static std::string Tensor___str__(const Tensor& self) {
  // TODO extremely rudimentary information for now
  //      goal would be a human-readable representation instead
  std::stringstream ss;
  ss << self;
  return ss.str();
}

static std::string Tensor___repr__(const Tensor& self) {
  // TODO extremely rudimentary information for now
  //      goal would be an unambiguous representation instead
  //
  // Potentially a good idea is to alter the TensorImpl.print function
  // directly instead
  std::stringstream ss;
  ss << self;
  return ss.str();
}

static size_t Tensor___len__(const Tensor& self) { return self.shape()[0]; }

//
// Operations with a scalar
//
static std::shared_ptr<Tensor> Tensor_scalar__imul__(std::shared_ptr<Tensor> self,
                                                     scalar_type number) {
  self->scale(number);
  return self;
}

static std::shared_ptr<Tensor> Tensor_scalar__mul__(const std::shared_ptr<Tensor>& self,
                                                    scalar_type number) {
  auto cpy_ptr = self->copy();
  return Tensor_scalar__imul__(cpy_ptr, number);
}

static std::shared_ptr<Tensor> Tensor_scalar__itruediv__(
      const std::shared_ptr<Tensor>& self, scalar_type number) {
  self->scale(1. / number);
  return self;
}

static std::shared_ptr<Tensor> Tensor_scalar__truediv__(
      const std::shared_ptr<Tensor>& self, scalar_type number) {
  auto cpy_ptr = self->copy();
  return Tensor_scalar__itruediv__(cpy_ptr, number);
}

// TODO missing:
//        - addition with a scalar
//        - subtraction with a scalar

//
// Operations with another tensor
//

static std::shared_ptr<Tensor> Tensor__iadd__(std::shared_ptr<Tensor> self,
                                              const std::shared_ptr<Tensor>& other) {
  self->add(other);
  return self;
}

static std::shared_ptr<Tensor> Tensor__add__(const std::shared_ptr<Tensor>& self,
                                             const std::shared_ptr<Tensor>& other) {
  auto cpy_ptr = self->copy();
  return Tensor__iadd__(cpy_ptr, other);
}

static std::shared_ptr<Tensor> Tensor__isub__(std::shared_ptr<Tensor> self,
                                              const std::shared_ptr<Tensor>& other) {
  self->add(-1.0, other);
  return self;
}

static std::shared_ptr<Tensor> Tensor__sub__(const std::shared_ptr<Tensor>& self,
                                             const std::shared_ptr<Tensor>& other) {
  auto cpy_ptr = self->copy();
  return Tensor__isub__(cpy_ptr, other);
}

static std::shared_ptr<Tensor> Tensor__mul__(const std::shared_ptr<Tensor>& self,
                                             const std::shared_ptr<Tensor>& other) {
  auto out_ptr = self->nosym_like();
  self->multiply_to(other, out_ptr);
  return out_ptr;
}

static std::shared_ptr<Tensor> Tensor__truediv__(const std::shared_ptr<Tensor>& self,
                                                 const std::shared_ptr<Tensor>& other) {
  auto out_ptr = self->nosym_like();
  self->divide_to(other, out_ptr);
  return out_ptr;
}

static std::shared_ptr<Tensor> Tensor__matmul__(const std::shared_ptr<Tensor>& self,
                                                const std::shared_ptr<Tensor>& other) {
  return contract("ij,jk->ik", self, other);
}

void export_Tensor(py::module& m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(
        m, "Tensor",
        "Class representing the Tensor objects used for computations in adcman")
        .def_property_readonly("ndim", &adcc::Tensor::ndim)
        .def_property_readonly("shape", &Tensor_shape)
        .def_property_readonly("size", &adcc::Tensor::size)
        .def_property_readonly("mutable", &adcc::Tensor::is_mutable)
        .def("set_immutable", &adcc::Tensor::set_immutable,
             "Set the tensor as immutable, allowing some optimisations to be performed.")
        .def("copy", &Tensor::copy, "Returns a deep copy of the tensor.")
        .def("copy_to", &Tensor::copy_to,
             "Writes a deep copy of the tensor to another tensor")
        .def("empty_like", &Tensor::zeros_like)  // TODO used to be empty_like
        .def("zeros_like", &Tensor::zeros_like)
        .def("ones_like", &Tensor::ones_like)
        .def("nosym_like", &Tensor::nosym_like)
        .def("set_mask", &adcc::Tensor::set_mask,
             "Set all elements corresponding to an index mask, which is given by a "
             "string eg. 'iijkli' sets elements T_{iijkli}")
        .def("dot", &Tensor_dot)
        .def("dot", &Tensor_dot_list)
        .def("transpose", &Tensor_transpose_1)
        .def("transpose", &Tensor_transpose_2)
        .def("symmetrise_to", &Tensor_symmetrise_to)
        .def("antisymmetrise_to", &Tensor_antisymmetrise_to)
        .def("add_linear_combination", &Tensor_add_linear_combination,
             "Add a linear combination of tensors to this tensor")
        .def("to_ndarray", &Tensor_to_ndarray,
             "Export the tensor data to a standard np::ndarray by making a copy.")
        .def("set_from_ndarray", &Tensor_from_ndarray,
             "Set all tensor elements from a standard np::ndarray by making a copy. "
             "Provide an optional tolerance argument to increase the tolerance for the "
             "chekc for symmetry consistency.")
        .def("set_from_ndarray", &Tensor_from_ndarray_tol,
             "Set all tensor elements from a standard np::ndarray by making a copy. "
             "Provide an optional tolerance argument to increase the tolerance for the "
             "chekc for symmetry consistency.")
        .def("describe_symmetry", &Tensor::describe_symmetry,
             "Return a string providing a hopefully discriptive rerpesentation of the "
             "symmetry information stored inside the tensor.")
        .def("__len__", &Tensor___len__)
        .def("__repr__", &Tensor___repr__)
        .def("__str__", &Tensor___str__)
        //
        .def("__imul__", &Tensor_scalar__imul__)          // tensor *= scalar
        .def("__mul__", &Tensor_scalar__mul__)            // tensor * scalar
        .def("__rmul__", &Tensor_scalar__mul__)           // scalar * tensor
        .def("__itruediv__", &Tensor_scalar__itruediv__)  // tensor /= scalar
        .def("__truediv__", &Tensor_scalar__truediv__)    // tensor / scalar
                                                          //
        .def("__mul__", &Tensor__mul__,
             "Multiply two tensors elementwise. Notice that this function discards any "
             "symmetry.")  // tensor * tensor
        .def("__truediv__", &Tensor__truediv__,
             "Divide two tensors elementwise. Notice that this function discards any "
             "symmetry.")                  // tensor / tensor
        .def("__iadd__", &Tensor__iadd__)  // tensor += tensor
        .def("__add__", &Tensor__add__)    // tensor + tensor
        .def("__isub__", &Tensor__isub__)  // tensor -= tensor
        .def("__sub__", &Tensor__sub__)    // tensor - tensor
        //
        .def("__matmul__", &Tensor__matmul__)  // tensor @ tensor
        //
        ;

  m.def("multiply", &multiply);
  m.def("divide", &divide);
  m.def("add", &add);
  m.def("subtract", &subtract);
  m.def("contract_to", &contract_to);
  m.def("contract", &adcc::contract);
}

}  // namespace py_iface
}  // namespace adcc
