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

#include "convert_timer.hh"
#include "hartree_fock_solution_hack.hh"
#include <adcc/ReferenceState.hh>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adcc {
namespace py_iface {

/** Exports adcc/ReferenceState.hh to python */
void export_ReferenceState(py::module& m) {

  py::class_<ReferenceState, std::shared_ptr<ReferenceState>>(
        m, "ReferenceState",
        "Class representing information about the reference state for adcc. Python "
        "binding to"
        ":cpp:class:`adcc::ReferenceState`.")
        .def(py::init<std::shared_ptr<const HartreeFockSolution_i>,
                      std::shared_ptr<const MoSpaces>, bool>(),
             "Setup a ReferenceStateject using an MoSpaces object.\n"
             "\n"
             "hfsoln_ptr        Pointer to the Interface to the host program,\n"
             "                  providing the HartreeFockSolution data, which\n"
             "                  will be provided by this object.\n"
             "mo_ptr            MoSpaces object containing info about the MoSpace setup\n"
             "                  and the point group symmetry.\n"
             "symmetry_check_on_import\n"
             "                  Should symmetry of the imported objects be checked\n"
             "                  explicitly during the import process. This massively "
             "slows\n"
             "                  down the import process and has a dramatic impact on "
             "memory\n"
             "                  usage and should thus only be used for debugging import "
             "routines\n"
             "                  from the host programs. Do not enable this unless you "
             "know\n"
             "                  that you really want to.\n")
        .def_property_readonly("restricted", &ReferenceState::restricted,
                               "Return whether the reference is restricted or not.")
        .def_property_readonly(
              "spin_multiplicity", &ReferenceState::spin_multiplicity,
              "Return the spin multiplicity of the reference state. 0 indicates "
              "that the spin cannot be determined or is not integer (e.g. UHF)")
        .def_property_readonly("has_core_occupied_space",
                               &ReferenceState::has_core_occupied_space,
                               "Is a core occupied space setup, such that a core-valence "
                               "separation can be applied.")
        .def_property_readonly("irreducible_representation",
                               &ReferenceState::irreducible_representation,
                               "Reference state irreducible representation")
        .def_property_readonly("mospaces", &ReferenceState::mospaces_ptr,
                               "The MoSpaces object supplied on initialisation")
        .def_property_readonly(
              "backend", &ReferenceState::backend,
              "The identifier of the back end used for the SCF calculation.")
        .def_property_readonly("n_orbs", &ReferenceState::n_orbs,
                               "Number of molecular orbitals")
        .def_property_readonly("n_orbs_alpha", &ReferenceState::n_orbs_alpha,
                               "Number of alpha orbitals")
        .def_property_readonly("n_orbs_beta", &ReferenceState::n_orbs_beta,
                               "Number of beta orbitals")
        .def_property_readonly("n_alpha", &ReferenceState::n_alpha,
                               "Number of alpha electrons")
        .def_property_readonly("n_beta", &ReferenceState::n_beta,
                               "Number of beta electrons")
        .def_property_readonly(
              "nuclear_total_charge",
              [](const ReferenceState& ref) { return ref.nuclear_multipole(0)[0]; })
        .def_property_readonly("nuclear_dipole",
                               [](const ReferenceState& ref) {
                                 py::array_t<scalar_type> ret(std::vector<ssize_t>{3});
                                 auto res = ref.nuclear_multipole(1);
                                 std::copy(res.begin(), res.end(), ret.mutable_data());
                                 return res;
                               })
        .def_property_readonly("conv_tol", &ReferenceState::conv_tol,
                               "SCF convergence tolererance")
        .def_property_readonly("energy_scf", &ReferenceState::energy_scf,
                               "Final total SCF energy")
        //
        .def("orbital_energies", &ReferenceState::orbital_energies,
             "Return the orbital energies corresponding to the provided space")
        .def("orbital_coefficients", &ReferenceState::orbital_coefficients,
             "Return the molecular orbital coefficients corresponding to the provided "
             "space (alpha and beta coefficients are returned)")
        .def("orbital_coefficients_alpha", &ReferenceState::orbital_coefficients_alpha,
             "Return the alpha molecular orbital coefficients corresponding to the "
             "provided space")
        .def("orbital_coefficients_beta", &ReferenceState::orbital_coefficients_beta,
             "Return the beta molecular orbital coefficients corresponding to the "
             "provided space")
        .def("fock", &ReferenceState::fock,
             "Return the Fock matrix block corresponding to the provided space.")
        .def("eri", &ReferenceState::eri,
             "Return the ERI (electron-repulsion integrals) tensor block corresponding "
             "to the provided space.")
        //
        .def("import_all", &ReferenceState::import_all,
             "Normally the class only imports the Fock matrix blocks and "
             "electron-repulsion integrals of a particular space combination when this "
             "is requested by a call to above fock() or eri() functions. This function "
             "call, however, instructs the class to immediately import *all* such "
             "blocks. Typically you do not want to do this.")
        .def_property("cached_fock_blocks", &ReferenceState::cached_fock_blocks,
                      &ReferenceState::set_cached_fock_blocks,
                      "Get or set the list of momentarily cached Fock matrix blocks\n"
                      "\n"
                      "Setting this property allows to drop fock matrix blocks if they "
                      "are no longer needed to save memory.")
        .def_property("cached_eri_blocks", &ReferenceState::cached_eri_blocks,
                      &ReferenceState::set_cached_eri_blocks,
                      "Get or set the list of momentarily cached ERI tensor blocks\n"
                      "\n"
                      "Setting this property allows to drop ERI tensor blocks if they "
                      "are no longer needed to save memory.")
        .def("flush_hf_cache", &ReferenceState::flush_hf_cache,
             "Tell the contained HartreeFockSolution_i object (which was passed upon "
             "construction), that a larger amount of import operations is done and that "
             "the next request for further imports will most likely take some time, such "
             "that intermediate caches can now be flushed to save some memory or other "
             "resources.")
        .def_property_readonly(
              "timer",
              [](const ReferenceState& self) { return convert_timer(self.timer()); },
              "Obtain the timer object of this class.")
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
