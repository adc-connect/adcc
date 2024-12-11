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

#include "../amplitude_vector_enforce_spin_kind.hh"
#include "../fill_ip_doubles_guesses.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace libadcc {

namespace py = pybind11;
using namespace pybind11::literals;

void export_adc_ip(py::module& m) {
  m.def("amplitude_vector_enforce_spin_kind", &amplitude_vector_enforce_spin_kind,
        "Apply the spin symmetrisation required to make the doubles and higher parts of "
        "an amplitude vector consist of components for a particular spin kind only.");

  m.def("fill_ip_doubles_guesses", &fill_ip_doubles_guesses, "guesses_d"_a, 
        "mospaces"_a, "d_o"_a, "d_v"_a, "a_spin"_a, "restricted"_a,
        "doublet"_a, "spin_change_twice"_a, "degeneracy_tolerance"_a,
        "Fill the passed vector of doubles blocks with doubles guesses using "
        "the O and V matrices., which are the two Fock matrices "
        "involved in the doubles block.\n\nguesses_d    Vectors of guesses, "
        "all elements are assumed to be initialised to zero and the symmetry "
        "is assumed to be properly set up.\nmospaces     Mospaces object"
        "\nd_o                   Matrix to construct guesses from (occ.)"
        "\nd_v                   Matrix to construct guesses from (virt.)"
        "\na_spin                If alpha ionization (false: beta)"
        "\nrestricted            Is this a restricted calculation"
        "\ndoublet               Doublet or quartet states (only in case of"
        "restricted calculation)"
        "\nspin_change_twice   Twice the value of the spin change to enforce "
        "in an excitation.\ndegeneracy_tolerance  Tolerance for two entries of "
        "the diagonal to be considered degenerate, i.e. identical."
        "\nReturns     The number of guess vectors which have been properly "
        "initialised (the others are invalid and should be discarded).");
}

}  // namespace libadcc