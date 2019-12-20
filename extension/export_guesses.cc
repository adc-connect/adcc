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

#include <adcc/guess_zero.hh>
#include <adcc/guesses_from_diagonal.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

void export_guesses(py::module& m) {

  py::class_<AdcGuessKind, std::shared_ptr<AdcGuessKind>>(
        m, "AdcGuessKind",
        "Class which collects information about the kind of guess vectors to be "
        "constructed. Python binding to :cpp:class:`adcc::AdcGuessKind`.")
        .def(py::init<>(), "Construct default AdcGuessKind.")
        .def(py::init<std::string, float, std::string>(),
             "Construct from irrep, spin_change and spin_block_symmetrisation.")
        .def_readonly("irrep", &AdcGuessKind::irrep,
                      " String describing the irreducible representation to consider")
        .def_readonly(
              "spin_change", &AdcGuessKind::spin_change,
              "The spin change to enforce in an excitation. Typical values are 0 and -1.")
        .def_readonly("spin_change_twice", &AdcGuessKind::spin_change_twice,
                      "Twice the change of spin_change.")
        .def_readonly(
              "spin_block_symmetrisation", &AdcGuessKind::spin_block_symmetrisation,
              "Symmetrisation to enforce between equivalent spin blocks, which all yield "
              "\n"
              "the desired spin_change. E.g. if spin_change == 0, then both the "
              "alpha->alpha \n"
              "and beta->beta blocks of the singles part of the excitation vector "
              "achieve \n"
              "this spin change. The symmetry specified with this parameter will then be "
              "\n"
              "imposed between the a-a and b-b blocks. \n"
              "Valid values are \"none\", \"symmetric\" and \"antisymmetric\", \n"
              "where \"none\" enforces no symmetry.")
        //
        ;

  m.def("guess_symmetries", &guess_symmetries,
        "Return a list of symmetry objects in order to construct the singles / doubles "
        "part of the AmplitudeVector.");

  m.def("guess_zero", &guess_zero,
        "Return an AmplitudeVector object filled with zeros, but where the symmetry has "
        "been properly set up to meet the requirements of the AdcGuessKind object");

  m.def("guesses_from_diagonal", &guesses_from_diagonal,
        "Obtain guesses by inspecting a block of the diagonal of the passed ADC matrix.\n"
        "The symmetry of the returned vectors is already set-up properly. Note that\n"
        "this routine may return fewer vectors than requested in case the requested\n"
        "number could not be found.\n"
        "\n"
        "matrix     AdcMatrix for which guesses are to be constructed\n"
        "kind       AdcGuessKind object describing the kind of guesses to be "
        "constructed\n"
        "block      The block of the diagonal to investigate. May be 's' (singles) or\n"
        "           'd' (doubles).\n"
        "n_guesses  The number of guesses to look for.\n"
        "degeneracy_tolerance  Tolerance for two entries of the diagonal to be "
        "considered\n"
        "                      degenerate, i.e. identical.");
}

}  // namespace py_iface
}  // namespace adcc
