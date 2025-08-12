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

#include "../Symmetry.hh"
#include "../make_symmetry.hh"
#include "util.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace libadcc {

namespace py = pybind11;

void export_Symmetry(py::module& m) {

  py::class_<Symmetry, std::shared_ptr<Symmetry>>(
        m, "Symmetry", "Container for Tensor symmetry information")
        .def(py::init<std::shared_ptr<const MoSpaces>, const std::string&>(),
             "Construct a Symmetry class from an MoSpaces object and the identifier for "
             "the space (e.g. o1o1, v1o1, o3v2o1v1, ...). Python binding to "
             ":cpp:class:`libadcc::Symmetry`.")
        .def(py::init<std::shared_ptr<const MoSpaces>, const std::string&,
                      std::map<std::string, std::pair<size_t, size_t>>>(),
             "Construct a Symmetry class from an MoSpaces object, a space string and a "
             "map to supply the number of orbitals for some additional axes.\nFor the "
             "additional axis the pair contains either two numbers (for the number of "
             "alpha and beta orbitals in this axis) or only one number and a zero (for "
             "an axis, which as only one spin kind, alpha or beta).\n\nThis is an "
             "advanced constructor. Use only if you know what you do.")
        //
        .def_property_readonly("mospaces", &Symmetry::mospaces_ptr,
                               "Return the MoSpaces object supplied on initialisation")
        .def_property_readonly("space", &Symmetry::space,
                               "Return the space supplied on initialisation.")
        .def_property_readonly("ndim", &Symmetry::ndim,
                               "Return the number of dimensions.")
        .def_property_readonly(
              "shape",
              [](std::shared_ptr<Symmetry> self) { return shape_tuple(self->shape()); },
              "Return the shape of tensors constructed from this symmetry.")
        .def_property_readonly("empty", &Symmetry::empty,
                               "Is the symmetry empty (i.e. noy symmetry setup)")
        .def("clear", &Symmetry::clear, "Clear the symmetry.")
        .def("describe", &Symmetry::describe, "Return a descriptive string.")
        //
        .def_property("irreps_allowed", &Symmetry::irreps_allowed,
                      &Symmetry::set_irreps_allowed,
                      "The list of irreducible representations, for which the tensor "
                      "shall be non-zero. If this is *not* set, i.e. an empty list, all "
                      "irreps will be allowed.")
        .def_property(
              "permutations", &Symmetry::permutations, &Symmetry::set_permutations,
              "The list of index permutations, which do not change the tensor.\n"
              "A minus may be used to indicate anti-symmetric\n"
              "permutations with respect to the first (reference) permutation.\n"
              "\n"
              "For example the list [\"ij\", \"ji\"] defines a symmetric matrix\n"
              "and [\"ijkl\", \"-jikl\", \"-ijlk\", \"klij\"] the symmetry of the ERI\n"
              "tensor. Not all permutations need to be given to fully describe\n"
              "the symmetry. Beware that the check for errors and conflicts\n"
              "is only rudimentary at the moment.")
        .def_property("spin_block_maps", &Symmetry::spin_block_maps,
                      &Symmetry::set_spin_block_maps,
                      "A list of tuples of the form (\"aaaa\", \"bbbb\", -1.0), i.e.\n"
                      "two spin blocks followed by a factor. This maps the second onto "
                      "the first\n"
                      "with a factor of -1.0 between them.")
        .def_property("spin_blocks_forbidden", &Symmetry::spin_blocks_forbidden,
                      &Symmetry::set_spin_blocks_forbidden,
                      "List of spin-blocks, which are marked forbidden (i.e. enforce "
                      "them to stay zero).\n"
                      "Blocks are given as a string in the letters 'a' and 'b', e.g. "
                      "[\"aaba\", \"abba\"]")
        //
        ;

  //
  // Factories for common cases
  //
  m.def("make_symmetry_orbital_energies", &make_symmetry_orbital_energies,
        "Return the Symmetry object like it would be set up for the passed subspace \n"
        "of the orbital energies tensor.\n"
        "\n"
        "  mospaces    MoSpaces object\n"
        "  space       space string (e.g. o1)");

  m.def("make_symmetry_orbital_coefficients", &make_symmetry_orbital_coefficients,
        "Return the Symmetry object like it would be set up for the passed subspace \n"
        "of the orbital coefficients tensor.\n"
        "\n"
        "  mospaces    MoSpaces object\n"
        "  space       Space string (e.g. o1b)\n"
        "  n_bas       Number of basis functions\n"
        "  blocks      Spin blocks to include. Valid are \"ab\", \"a\" and \"b\".");

  m.def("make_symmetry_eri", &make_symmetry_eri,
        "Return the Symmetry object like it would be set up for the passed subspace \n"
        "of the electron-repulsion tensor.\n"
        "\n"
        "  mospaces    MoSpaces object\n"
        "  space       Space string (e.g. o1v1o1v1)\n");

  m.def("make_symmetry_operator", &make_symmetry_operator,
        "Return the Symmetry object for an orbital subspace block of a one-particle "
        "operator\n"
        "\n"
        "  mospaces    MoSpaces object\n"
        "  space       Space string (e.g. o1v1)\n"
        "  symmetric   Is the tensor symmetric (only in effect if both subspaces\n"
        "              of the space string are identical). False disables\n"
        "              a setup of permutational symmetry.\n"
        "  cartesian_transformation\n"
        "              The cartesian function according to which the operator "
        "transforms.\n"
        "\n"
        "Valid cartesian_transformation values include:\n"
        "     \"1\"                   Totally symmetric (default)\n"
        "     \"x\", \"y\", \"z\"         Coordinate axis\n"
        "     \"xx\", \"xy\", \"yz\" ...  Products of two coordinate axis\n"
        "     \"Rx\", \"Ry\", \"Rz\"      Rotations about the coordinate axis\n");

  m.def("make_symmetry_operator_basis", &make_symmetry_operator_basis,
        "Return the symmetry object for an operator in the AO basis. The object will\n"
        "represent a block-diagonal matrix of the form\n"
        "    ( M 0 )\n"
        "    ( 0 M ).\n"
        "where M is an n_bas x n_bas block and is indentical in upper-left\n"
        "and lower-right.\n"
        "\n"
        "mospaces_ptr     MoSpaces pointer\n"
        "n_bas            Number of AO basis functions\n"
        "symmetric        Is the tensor symmetric (only in effect if both space\n"
        "                 axes identical). false disables a setup of permutational\n"
        "                 symmetry.\n");
}

}  // namespace libadcc
