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
#include <adcc/config.hh>
#include <adcc/exceptions.hh>
#include <sstream>
#include <type_traits>

namespace adcc {

namespace py_iface {

namespace py = pybind11;

py::tuple shape_tuple(const std::vector<size_t>& shape) {
  switch (shape.size()) {
    case 0:
      throw runtime_error("Encountered unexpected dimensionality 0.");
    case 1:
      return py::make_tuple(shape[0]);
    case 2:
      return py::make_tuple(shape[0], shape[1]);
    case 3:
      return py::make_tuple(shape[0], shape[1], shape[2]);
    case 4:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3]);
    case 5:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4]);
    case 6:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
    case 7:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                            shape[6]);
    case 8:
      return py::make_tuple(shape[0], shape[1], shape[2], shape[3], shape[4], shape[5],
                            shape[6], shape[7]);
    default:
      throw not_implemented_error(
            "shape_tuple only implemented up to dimensionality 8 so far.");
      // TensorImpl is only implemented up to 4 indices so far
      // libtensor only supports up to 8 indices at the moment
  }
}

template <typename T>
T* extract_array_data(const py::array_t<T>& array, std::vector<size_t> shape) {
  // Convert numpy array shape to std::vector
  const size_t array_dim = static_cast<size_t>(array.ndim());
  const auto array_shape = array.shape();
  std::vector<size_t> vec_shape(array_dim);
  for (size_t i = 0; i < array_dim; ++i) {
    vec_shape[i] = static_cast<const size_t&>(array_shape[i]);
  }

  if (vec_shape != shape) {
    std::stringstream ss;
    ss << "Inconsintent array shape. Expected (";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      ss << shape[i];
    }
    ss << "), but obtained (";
    for (size_t i = 0; i < vec_shape.size(); ++i) {
      if (i != 0) {
        ss << ",";
      }
      ss << vec_shape[i];
    }
    ss << ").";
    throw invalid_argument(ss.str());
  }
  return const_cast<T*>(array.data());
}

template <typename T>
py::array_t<T> make_array(T* data, std::vector<size_t> shape, const py::handle& owner) {
  using T_noconst = typename std::remove_const<T>::type;

  const bool any_shape_zero = [&shape] {
    for (auto& s : shape) {
      if (s == 0) return true;
    }
    return false;
  }();
  if (data == nullptr || any_shape_zero) {
    std::vector<size_t> newshape(shape.size(), 0);
    return py::array_t<T_noconst>(newshape);
  }

  // Construct the strides
  size_t accu = 1;
  std::vector<size_t> strides(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    strides[shape.size() - 1 - i] = sizeof(T_noconst) * accu;
    accu *= shape[shape.size() - 1 - i];
  }

  return py::array_t<T_noconst>(shape, strides, data, owner);
}

template <typename Listlike>
std::vector<std::shared_ptr<Tensor>> extract_tensors(const Listlike& in) {
  std::vector<std::shared_ptr<Tensor>> ret;
  for (py::handle elem : in) {
    ret.push_back(elem.cast<std::shared_ptr<Tensor>>());
  }
  return ret;
}

py::list pack_tensors(const std::vector<std::shared_ptr<Tensor>>& list) {
  py::list ret;
  for (auto& tensor_ptr : list) {
    ret.append(tensor_ptr);
  }
  return ret;
}

//
// Template instantiations
//
template std::vector<std::shared_ptr<Tensor>> extract_tensors<py::list>(
      const py::list& in);
template std::vector<std::shared_ptr<Tensor>> extract_tensors<py::tuple>(
      const py::tuple& in);

#define INSTANTIATE_MAKE(TYPE)                                                        \
  template py::array_t<TYPE> make_array<TYPE>(TYPE * data, std::vector<size_t> shape, \
                                              const py::handle& owner)

#define INSTANTIATE_EXTRACT(TYPE)                                         \
  template TYPE* extract_array_data<TYPE>(const py::array_t<TYPE>& array, \
                                          std::vector<size_t> shape)

INSTANTIATE_MAKE(bool);
INSTANTIATE_MAKE(const bool);
INSTANTIATE_MAKE(scalar_type);
INSTANTIATE_MAKE(const scalar_type);

INSTANTIATE_EXTRACT(scalar_type);
INSTANTIATE_EXTRACT(bool);

}  // namespace py_iface
}  // namespace adcc
