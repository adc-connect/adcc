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

#include <adcc/amplitude_vector_enforce_spin_kind.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

void export_amplitude_vector_enforce_spin_kind(py::module& m) {
  m.def("amplitude_vector_enforce_spin_kind", &amplitude_vector_enforce_spin_kind,
        "Apply the spin symmetrisation required to make the doubles and higher parts of "
        "an amplitude vector consist of components for a particular spin kind only.");
}

}  // namespace py_iface
}  // namespace adcc
