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
#include <adcc/Timer.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {
namespace py = pybind11;

// TODO When the Timer functionality is used more wide-spread
//      (i.e. in different classes) it probably makes sense to integrate
//      this into the pybind11 automatic type conversion mechanism.
/** Convert a C++ timer to a python timer */
py::object convert_timer(const Timer& timer);

}  // namespace py_iface
}  // namespace adcc

