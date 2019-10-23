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

#pragma once
#include <adcc/CachingPolicy.hh>
#include <pybind11/pybind11.h>

namespace pybind11 {
namespace detail {
using adcc::CachingPolicy_i;
namespace py = pybind11;
// This workaround is straight from
// https://github.com/pybind/pybind11/issues/1546 Should not be needed any more
// once the above is merged

template <>
struct type_caster<std::shared_ptr<CachingPolicy_i>> {
  PYBIND11_TYPE_CASTER(std::shared_ptr<CachingPolicy_i>, _("CachingPolicy_i"));

  using CPiCaster =
        copyable_holder_caster<CachingPolicy_i, std::shared_ptr<CachingPolicy_i>>;

  bool load(pybind11::handle src, bool b) {
    CPiCaster bc;
    bool success = bc.load(src, b);
    if (!success) {
      return false;
    }

    auto py_obj  = py::reinterpret_borrow<py::object>(src);
    auto cpi_ptr = static_cast<std::shared_ptr<CachingPolicy_i>>(bc);

    // Construct a shared_ptr to the py::object
    // For the deleter note, that it's possible that when the shared_ptr
    // dies we won't have the GIL (global interpreter lock)
    // (if the last holder is in a non-Python thread), so we make
    // sure to acquire it in the deleter.
    auto py_obj_ptr =
          std::shared_ptr<py::object>{new object{py_obj}, [](py::object* py_object_ptr) {
                                        gil_scoped_acquire gil;
                                        delete py_object_ptr;
                                      }};

    value = std::shared_ptr<CachingPolicy_i>(py_obj_ptr, cpi_ptr.get());
    return true;
  }

  static handle cast(std::shared_ptr<CachingPolicy_i> base, return_value_policy rvp,
                     handle h) {
    return CPiCaster::cast(base, rvp, h);
  }
};

template <>
struct is_holder_type<CachingPolicy_i, std::shared_ptr<CachingPolicy_i>>
      : std::true_type {};
}  // namespace detail
}  // namespace pybind11

