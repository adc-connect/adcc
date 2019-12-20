//
// Copyright (C) 2019 by the adcc authors
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

#include "caching_policy_hack.hh"
#include <adcc/CachingPolicy.hh>
#include <memory>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

/** This implements the trampoline for  CachingPolicy_i */
class PyCachingPolicy_i : public CachingPolicy_i {
 public:
  using CachingPolicy_i::CachingPolicy_i;
  virtual ~PyCachingPolicy_i() = default;

  bool should_cache(const std::string& tensor_label, const std::string& tensor_space,
                    const std::string& leading_order_contraction) override {
    PYBIND11_OVERLOAD_PURE(bool, CachingPolicy_i, should_cache, tensor_label,
                           tensor_space, leading_order_contraction);
  }
};

void export_CachingPolicy(py::module& m) {
  py::class_<CachingPolicy_i, std::shared_ptr<CachingPolicy_i>, PyCachingPolicy_i>(
        m, "CachingPolicy_i",
        "Should a particular tensor given by a label, its space string and the string of "
        "the spaces involved in the most expensive contraction be stored. Python binding "
        "to "
        ":cpp:class:`adcc::CachingPolicy_i`")
        .def(py::init<>())
        .def("should_cache", &CachingPolicy_i::should_cache,
             "Should a particular tensor given by a label, its space string and the "
             "string of "
             "the spaces involved in the most expensive contraction be stored.");
}

}  // namespace py_iface
}  // namespace adcc
