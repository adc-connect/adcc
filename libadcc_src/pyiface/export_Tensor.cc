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

#include "../Tensor.hh"
#include "../exceptions.hh"
#include "util.hh"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <sstream>

namespace libadcc {

namespace py = pybind11;
using namespace pybind11::literals;
typedef std::shared_ptr<Tensor> ten_ptr;

static std::vector<std::vector<size_t>> parse_permutations(
      const py::iterable& permutations) {
  bool iterator_of_ints = true;
  for (auto tpl : permutations) {
    if (!py::isinstance<py::int_>(tpl)) {
      iterator_of_ints = false;
      break;
    }
  }

  if (iterator_of_ints) {
    std::vector<size_t> perms;
    for (auto itm : permutations) {
      perms.push_back(itm.cast<size_t>());
    }
    return std::vector<std::vector<size_t>>{perms};
  }

  std::vector<std::vector<size_t>> vec_perms;
  for (auto tpl : permutations) {
    std::vector<size_t> perms;
    for (auto itm : tpl) {
      perms.push_back(itm.cast<size_t>());
    }
    vec_perms.push_back(perms);
  }
  return vec_perms;
}

static std::vector<size_t> convert_index_tuple(const ten_ptr& self, py::tuple idcs) {
  if (idcs.size() != self->ndim()) {
    throw py::value_error(
          "Number of elements passed in index tuple (== " + std::to_string(idcs.size()) +
          ") and dimensionality of tensor (== " + std::to_string(self->ndim()) +
          ") do not agree. Note, that at the moment any kind of slicing operation "
          "(including partial slicing) are not yet implemented.");
  }
  const std::vector<size_t> shape = self->shape();
  std::vector<size_t> ret(idcs.size());
  for (size_t i = 0; i < idcs.size(); ++i) {
    ptrdiff_t idx;

    try {
      idx = idcs[i].cast<ptrdiff_t>();
    } catch (const py::cast_error& c) {
      throw py::cast_error(
            "Right now only integer indices are supported. Any kind of slicing operation "
            "(including partial slicing) are not yet implemented.");
    }
    if (idx < 0) {
      auto si = static_cast<size_t>(-idx);
      if (si > shape[i]) {
        throw py::index_error("index " + std::to_string(idx) +
                              " is out of bounds for axis " + std::to_string(i) +
                              " with size " + std::to_string(shape[i]));
      }
      ret[i] = shape[i] - si;
    } else {
      auto si = static_cast<size_t>(idx);
      if (si > shape[i]) {
        throw py::index_error("index " + std::to_string(idx) +
                              " is out of bounds for axis " + std::to_string(i) +
                              " with size " + std::to_string(shape[i]));
      }
      ret[i] = si;
    }
  }
  return ret;
}

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

static ten_ptr Tensor_from_ndarray_tol(
      ten_ptr self,
      py::array_t<scalar_type, py::array::c_style | py::array::forcecast> in_array,
      double symmetry_tolerance) {
  py::ssize_t nd = in_array.ndim();
  if (nd < 1) throw invalid_argument("Cannot import from 0D array.");

  py::ssize_t pysize = 1;
  for (py::ssize_t i = 0; i < nd; ++i) pysize *= in_array.shape(i);

  const size_t size = static_cast<size_t>(pysize);
  self->import_from(in_array.data(), size, symmetry_tolerance);
  return self;
}

static ten_ptr Tensor_from_ndarray(ten_ptr self, py::array in_array) {
  return Tensor_from_ndarray_tol(self, in_array, 0.0);
}

static ten_ptr Tensor_set_random(ten_ptr self) {
  self->set_random();
  return self;
}

static scalar_type Tensor_dot(const Tensor& self, ten_ptr other) {
  return self.dot({other})[0];
}

static py::array_t<scalar_type> Tensor_dot_list(const Tensor& self, py::list tensors) {
  std::vector<ten_ptr> parsed   = extract_tensors(tensors);
  std::vector<scalar_type> dots = self.dot(parsed);
  py::array_t<scalar_type> ret(dots.size());
  std::copy(dots.begin(), dots.end(), ret.mutable_data());

  return ret;
}

static ten_ptr Tensor_transpose_1(const Tensor& self) {
  std::vector<size_t> vec_axes(self.ndim());
  for (size_t i = 0; i < self.ndim(); ++i) {
    vec_axes[self.ndim() - i - 1] = i;
  }
  return self.transpose(vec_axes);
}

static ten_ptr Tensor_transpose_2(const Tensor& self, py::tuple axes) {
  std::vector<size_t> vec_axes(py::len(axes));
  for (size_t i = 0; i < py::len(axes); ++i) {
    vec_axes[i] = axes[i].cast<size_t>();
  }
  return self.transpose(vec_axes);
}

static ten_ptr Tensor_symmetrise_1(const Tensor& self, py::list permutations) {
  return self.symmetrise(parse_permutations(permutations));
}

static ten_ptr Tensor_symmetrise_2(const Tensor& self, py::args permutations) {
  if (py::len(permutations) == 0) {
    if (self.ndim() != 2) {
      throw invalid_argument(
            "symmetrise without arguments may only be used for matrices.");
    }
    return self.symmetrise({{0, 1}});
  }
  return self.symmetrise(parse_permutations(permutations));
}

static ten_ptr Tensor_antisymmetrise_1(const Tensor& self, py::list permutations) {
  return self.antisymmetrise(parse_permutations(permutations));
}

static ten_ptr Tensor_antisymmetrise_2(const Tensor& self, py::args permutations) {
  if (py::len(permutations) == 0) {
    if (self.ndim() != 2) {
      throw invalid_argument(
            "antisymmetrise without arguments may only be used for matrices.");
    }
    return self.antisymmetrise({{0, 1}});
  }
  return self.antisymmetrise(parse_permutations(permutations));
}

static py::object tensordot_1(ten_ptr a, ten_ptr b, py::iterable axes) {
  if (py::len(axes) != 2) {
    throw invalid_argument("axes needs to be an iterable of length 2");
  }
  std::vector<std::vector<size_t>> c_axes;
  for (py::handle ax : axes) {
    std::vector<size_t> res;
    for (py::handle elem : ax.cast<py::iterable>()) {
      res.push_back(elem.cast<size_t>());
    }
    c_axes.push_back(res);
  }

  TensorOrScalar res = a->tensordot(b, {c_axes[0], c_axes[1]});
  if (res.tensor_ptr == nullptr) {
    return py::cast(res.scalar);
  } else {
    return py::cast(res.tensor_ptr);
  }
}
static py::object tensordot_2(ten_ptr a, ten_ptr b, size_t axes) {
  std::vector<size_t> a_axes;
  std::vector<size_t> b_axes;
  for (size_t i = 0; i < axes; ++i) {
    a_axes.push_back(a->ndim() - axes + i);
    b_axes.push_back(i);
  }

  TensorOrScalar res = a->tensordot(b, {a_axes, b_axes});
  if (res.tensor_ptr == nullptr) {
    return py::cast(res.scalar);
  } else {
    return py::cast(res.tensor_ptr);
  }
}

static py::object tensordot_3(ten_ptr a, ten_ptr b) { return tensordot_2(a, b, 2); }

static ten_ptr Tensor_diagonal(ten_ptr ten, py::args permutations) {
  std::vector<size_t> axes;
  if (py::len(permutations) == 0) {
    axes.push_back(0);
    axes.push_back(1);
  } else {
    for (auto itm : permutations) axes.push_back(itm.cast<size_t>());
  }
  return ten->diagonal(axes);
}

static ten_ptr direct_sum(ten_ptr a, ten_ptr b) { return a->direct_sum(b); }

static double Tensor_trace_1(std::string subscripts, const Tensor& tensor) {
  return tensor.trace(subscripts);
}
static double Tensor_trace_2(const Tensor& tensor) {
  if (tensor.ndim() != 2) {
    throw invalid_argument(
          "trace function without arguments may only be used for matrices.");
  }
  return tensor.trace("ii");
}

static ten_ptr linear_combination_strict(
      py::array_t<scalar_type, py::array::c_style> coefficients, py::list tensors) {

  if (coefficients.ndim() != 1) {
    throw invalid_argument("coefficients array needs to have exactly one dimension.");
  }
  size_t in_size             = static_cast<size_t>(coefficients.shape(0));
  const scalar_type* in_data = coefficients.data();
  std::vector<scalar_type> scalars(in_size);
  std::copy(in_data, in_data + in_size, scalars.data());
  std::vector<ten_ptr> parsed = extract_tensors(tensors);

  auto ret = parsed[0]->zeros_like();
  ret->add_linear_combination(scalars, parsed);
  return ret;
}

//
// Element access
//

static py::list Tensor_select_n_min(const ten_ptr& self, size_t n) {
  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret = self->select_n_min(n);
  py::list li;
  for (auto p : ret) li.append(py::make_tuple(p.first, p.second));
  return li;
}

static py::list Tensor_select_n_max(const ten_ptr& self, size_t n) {
  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret = self->select_n_max(n);
  py::list li;
  for (auto p : ret) li.append(py::make_tuple(p.first, p.second));
  return li;
}

static py::list Tensor_select_n_absmin(const ten_ptr& self, size_t n) {
  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret = self->select_n_absmin(n);
  py::list li;
  for (auto p : ret) li.append(py::make_tuple(p.first, p.second));
  return li;
}

static py::list Tensor_select_n_absmax(const ten_ptr& self, size_t n) {
  std::vector<std::pair<std::vector<size_t>, scalar_type>> ret = self->select_n_absmax(n);
  py::list li;
  for (auto p : ret) li.append(py::make_tuple(p.first, p.second));
  return li;
}

static bool Tensor_is_allowed(const ten_ptr& self, py::tuple idcs) {
  return self->is_element_allowed(convert_index_tuple(self, idcs));
}

static scalar_type Tensor__getitem__(const ten_ptr& self, py::tuple idcs) {
  return self->get_element(convert_index_tuple(self, idcs));
}

static scalar_type Tensor__setitem__(const ten_ptr& self, py::tuple idcs,
                                     scalar_type value) {
  self->set_element(convert_index_tuple(self, idcs), value);
  return value;
}

//
// Implementation of python-side special functions
//    See https://docs.python.org/3/library/operator.html for details
//

static py::object Tensor___str__(const Tensor& self) {
  if (self.size() < 50000) {
    return py::str(Tensor_to_ndarray(self));
  } else {
    std::stringstream ss;
    ss << "Tensor shape (";
    for (size_t i = 0; i < self.ndim(); ++i) {
      ss << self.shape()[i];
      if (i != self.ndim() - 1) ss << ", ";
    }
    ss << ")";
    return py::cast(ss.str());
  }
}

static py::object Tensor___repr__(const Tensor& self) {
  // TODO extremely rudimentary information for now
  //      goal would be an unambiguous representation instead
  return Tensor___str__(self);
}

//
// Operations with a scalar
//
static ten_ptr Tensor_scalar__imul__(ten_ptr self, scalar_type number) {
  return self = self->scale(number);
}

static ten_ptr Tensor_scalar__mul__(const ten_ptr& self, scalar_type number) {
  return self->scale(number);
}

static ten_ptr Tensor_scalar__itruediv__(ten_ptr& self, scalar_type number) {
  return self = self->scale(1. / number);
}

static ten_ptr Tensor_scalar__truediv__(const ten_ptr& self, scalar_type number) {
  return self->scale(1. / number);
}

static ten_ptr Tensor_scalar__add__(const ten_ptr& self, scalar_type number) {
  if (number == 0) {
    return self;
  } else {
    // TODO I know there are more efficient ways to do this in libtensor,
    //      but they are not yet exported all the way up
    auto other = self->empty_like();
    other->fill(number);
    return self->add(other);
  }
}

static ten_ptr Tensor_scalar__sub__(const ten_ptr& self, scalar_type number) {
  if (number == 0) {
    return self;
  } else {
    // TODO I know there are more efficient ways to do this in libtensor,
    //      but they are not yet exported all the way up
    auto other = self->empty_like();
    other->fill(-number);
    return self->add(other);
  }
}

//
// Operations with another tensor
//

static ten_ptr Tensor__iadd__(ten_ptr self, const ten_ptr& other) {
  return self = self->add(other);
}

static ten_ptr Tensor__add__(const ten_ptr& self, const ten_ptr& other) {
  return self->add(other);
}

static ten_ptr Tensor__isub__(ten_ptr self, const ten_ptr& other) {
  return self = self->add(other->scale(-1.0));
}

static ten_ptr Tensor__sub__(const ten_ptr& self, const ten_ptr& other) {
  return self->add(other->scale(-1.0));
}

static ten_ptr Tensor__mul__(const ten_ptr& self, const ten_ptr& other) {
  return self->multiply(other);
}

static ten_ptr Tensor__truediv__(const ten_ptr& self, const ten_ptr& other) {
  return self->divide(other);
}

static ten_ptr Tensor__matmul__(const ten_ptr& self, const ten_ptr& other) {
  return self->tensordot(other, {{1}, {0}}).tensor_ptr;
}

void export_Tensor(py::module& m) {
  py::class_<Tensor, std::shared_ptr<Tensor>>(
        m, "Tensor",
        "Class representing the Tensor objects used for computations in adcc")
        .def(py::init(&make_tensor_zero),
             "Construct a Tensor object using a Symmetry object describing its symmetry "
             "properties.\n"
             "The returned object is not guaranteed to contain initialised memory. "
             "Python binding to :cpp:class:`libadcc::Tensor`")
        .def_property_readonly("ndim", &Tensor::ndim)
        .def_property_readonly("shape", &Tensor_shape)
        .def_property_readonly("size", &Tensor::size)
        .def_property_readonly("space", &Tensor::space)
        .def_property_readonly("subspaces", &Tensor::subspaces)
        .def_property("flags", &Tensor::flags, &Tensor::set_flags)
        //
        .def_property_readonly("needs_evaluation", &Tensor::needs_evaluation,
                               "Does the tensor need evaluation or is it fully evaluated "
                               "and resilient in memory.")
        .def("evaluate", &evaluate,
             "Ensure the tensor to be fully evaluated and resilient in memory. Usually "
             "happens automatically when needed. Might be useful for fine-tuning, "
             "however.")
        .def_property_readonly("mutable", &Tensor::is_mutable)
        .def("set_immutable", &Tensor::set_immutable,
             "Set the tensor as immutable, allowing some optimisations to be performed.")
        //
        .def("empty_like", &Tensor::zeros_like)  // TODO used to be empty_like
        .def("zeros_like", &Tensor::zeros_like)
        .def("ones_like", &Tensor::ones_like)
        .def("nosym_like", &Tensor::nosym_like)
        .def("set_random", &Tensor_set_random,
             "Set all tensor elements to random data, adhering to the internal "
             "symmetry.")
        .def("set_mask", &Tensor::set_mask,
             "Set all elements corresponding to an index mask, which is given by a "
             "string eg. 'iijkli' sets elements T_{iijkli}")
        .def("diagonal", &Tensor_diagonal)
        .def("copy", &Tensor::copy, "Returns a deep copy of the tensor.")
        .def("dot", &Tensor_dot)
        .def("dot", &Tensor_dot_list)
        .def_property_readonly("T", &Tensor_transpose_1)
        .def("transpose", &Tensor_transpose_1)
        .def("transpose", &Tensor_transpose_2)
        .def("symmetrise", &Tensor_symmetrise_1)
        .def("symmetrise", &Tensor_symmetrise_2)
        .def("antisymmetrise", &Tensor_antisymmetrise_1)
        .def("antisymmetrise", &Tensor_antisymmetrise_2)
        .def("to_ndarray", &Tensor_to_ndarray,
             "Export the tensor data to a standard np::ndarray by making a copy.")
        .def("set_from_ndarray", &Tensor_from_ndarray,
             "Set all tensor elements from a standard np::ndarray by making a copy. "
             "Provide an optional tolerance argument to increase the tolerance for the "
             "check for symmetry consistency.")
        .def("set_from_ndarray", &Tensor_from_ndarray_tol,
             "Set all tensor elements from a standard np::ndarray by making a copy. "
             "Provide an optional tolerance argument to increase the tolerance for the "
             "check for symmetry consistency.")
        .def("describe_symmetry", &Tensor::describe_symmetry,
             "Return a string providing a hopefully descriptive representation of the "
             "symmetry information stored inside the tensor.")
        .def("describe_expression", &Tensor::describe_expression,
             "Return a string providing a hopefully descriptive representation of the "
             "tensor expression stored inside the object.")
        .def("describe_expression",
             [](ten_ptr t) { return t->describe_expression("unoptimised"); })
        //
        .def("__getitem__", &Tensor__getitem__,
             "Get a tensor element or a slice of tensor elements.")
        .def("__setitem__", &Tensor__setitem__,
             "Set a tensor element or a slice of tensor elements. The operation will "
             "adhere symmetry, i.e. alter all elements equivalent by symmetry at once.")
        .def("is_allowed", &Tensor_is_allowed,
             " Is a particular index allowed by symmetry")
        .def("select_n_absmax", &Tensor_select_n_absmax,
             "Select the n absolute maximal elements.")
        .def("select_n_absmin", &Tensor_select_n_absmin,
             "Select the n absolute minimal elements.")
        .def("select_n_max", &Tensor_select_n_max, "Select the n maximal elements.")
        .def("select_n_min", &Tensor_select_n_min, "Select the n minimal elements.")
        //
        .def("__len__", [](ten_ptr self) { return self->shape()[0]; })
        .def("__repr__", &Tensor___repr__)
        .def("__str__", &Tensor___str__)
        //
        .def("__pos__", [](ten_ptr self) { return self; })               // + tensor
        .def("__neg__", [](ten_ptr self) { return self->scale(-1.0); })  // - tensor
        .def("__add__", &Tensor_scalar__add__)            // tensor + scalar
        .def("__sub__", &Tensor_scalar__sub__)            // tensor - scalar
        .def("__radd__", &Tensor_scalar__add__)           // scalar + tensor
        .def("__rsub__", &Tensor_scalar__sub__)           // scalar - tensor
        .def("__imul__", &Tensor_scalar__imul__)          // tensor *= scalar
        .def("__mul__", &Tensor_scalar__mul__)            // tensor * scalar
        .def("__rmul__", &Tensor_scalar__mul__)           // scalar * tensor
        .def("__itruediv__", &Tensor_scalar__itruediv__)  // tensor /= scalar
        .def("__truediv__", &Tensor_scalar__truediv__)    // tensor / scalar
                                                          //
        .def("__mul__", &Tensor__mul__,
             "Multiply two tensors elementwise.")  // tensor * tensor
        .def("__truediv__", &Tensor__truediv__,
             "Divide two tensors elementwise.")  // tensor / tensor
        .def("__iadd__", &Tensor__iadd__)        // tensor += tensor
        .def("__add__", &Tensor__add__)          // tensor + tensor
        .def("__isub__", &Tensor__isub__)        // tensor -= tensor
        .def("__sub__", &Tensor__sub__)          // tensor - tensor
        //
        .def("__matmul__", &Tensor__matmul__)  // tensor @ tensor
        //
        ;

  m.def("evaluate", &evaluate);
  m.def("tensordot", &tensordot_1, "a"_a, "b"_a, "axes"_a);
  m.def("tensordot", &tensordot_2, "a"_a, "b"_a, "axes"_a);
  m.def("tensordot", &tensordot_3, "a"_a, "b"_a);
  m.def("direct_sum", &direct_sum, "a"_a, "b"_a);
  m.def("trace", &Tensor_trace_1, "subscripts"_a, "tensor"_a);
  m.def("trace", &Tensor_trace_2, "tensor"_a);
  m.def("linear_combination_strict", &linear_combination_strict, "coefficients"_a,
        "tensors"_a);
}

}  // namespace libadcc
