//
// Copyright (C) 2018 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include <adcc/ReferenceStateNew.hh>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace adcc {
namespace py_iface {

/** Exports adcc/ReferenceState.hh to python */
void export_ReferenceStateNew(py::module& m) {

  py::class_<ReferenceStateNew, std::shared_ptr<ReferenceStateNew>>(
        m, "ReferenceStateNew",
        "Class representing information about the reference state for adcc.")
        .def(py::init<std::shared_ptr<const HartreeFockSolution_i>,
                      std::shared_ptr<const AdcMemory>, std::vector<size_t>,
                      std::vector<size_t>, std::vector<size_t>, bool>(),
             "Setup a ReferenceState object using lists for orbitals spaces\n"
             "\n"
             "hfsoln_ptr        Pointer to the Interface to the host program,\n"
             "                  providing the HartreeFockSolution data, which\n"
             "                  will be provided by this object.\n"
             "adcmem_ptr        Pointer to the adc memory management object.\n"
             "core_orbitals     List of orbitals indices (in the full fock space, "
             "original\n"
             "                  ordering of the hf object), which defines the orbitals "
             "to\n"
             "                  be put into the core space, if any. The same number\n"
             "                  of alpha and beta orbitals should be selected. These "
             "will\n"
             "                  be forcibly occupied.\n"
             "frozen_core_orbitals\n"
             "                  List of orbital indices, which define the frozen core,\n"
             "                  i.e. those occupied orbitals, which do not take part in\n"
             "                  the ADC calculation. The same number of alpha and beta\n"
             "                  orbitals has to be selected.\n"
             "frozen_virtuals   List of orbital indices, which the frozen virtuals,\n"
             "                  i.e. those virtual orbitals, which do not take part\n"
             "                  in the ADC calculation. The same number of alpha and "
             "beta\n"
             "                  orbitals has to be selected.\n"
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
        //
        .def_property_readonly("restricted", &ReferenceStateNew::restricted,
                               "Return whether the reference is restricted or not.")
        .def_property_readonly(
              "spin_multiplicity", &ReferenceStateNew::spin_multiplicity,
              "Return the spin multiplicity of the reference state. 0 indicates "
              "that the spin cannot be determined or is not integer (e.g. UHF)")
        .def_property_readonly("core_occupied_space",
                               &ReferenceStateNew::has_core_occupied_space,
                               "Is a core occupied space setup, such that a core-valence "
                               "separation can be applied.")
        .def_property_readonly("irreducible_representation",
                               &ReferenceStateNew::irreducible_representation,
                               "Reference state irreducible representation")
        .def_property_readonly("mospaces", &ReferenceStateNew::mospaces_ptr,
                               "The MoSpaces object supplied on initialisation")
        .def_property_readonly("n_orbs", &ReferenceStateNew::n_orbs,
                               "Number of molecular orbitals")
        .def_property_readonly("n_orbs_alpha", &ReferenceStateNew::n_orbs_alpha,
                               "Number of alpha orbitals")
        .def_property_readonly("n_orbs_beta", &ReferenceStateNew::n_orbs_beta,
                               "Number of beta orbitals")
        .def_property_readonly("n_alpha", &ReferenceStateNew::n_alpha,
                               "Number of alpha electrons")
        .def_property_readonly("n_beta", &ReferenceStateNew::n_beta,
                               "Number of beta electrons")
        .def_property_readonly("conv_tol", &ReferenceStateNew::conv_tol,
                               "SCF convergence tolererance")
        .def_property_readonly("energy_scf", &ReferenceStateNew::energy_scf,
                               "Final total SCF energy")
        //
        .def("orbital_energies", &ReferenceStateNew::orbital_energies,
             "Return the orbital energies corresponding to the provided space")
        .def("orbital_coefficients", &ReferenceStateNew::orbital_coefficients,
             "Return the molecular orbital coefficients corresponding to the provided "
             "space (alpha and beta coefficients are returned)")
        .def("orbital_coefficients_alpha", &ReferenceStateNew::orbital_coefficients_alpha,
             "Return the alpha molecular orbital coefficients corresponding to the "
             "provided space")
        .def("orbital_coefficients_beta", &ReferenceStateNew::orbital_coefficients_beta,
             "Return the beta molecular orbital coefficients corresponding to the "
             "provided space")
        .def("fock", &ReferenceStateNew::fock,
             "Return the Fock matrix block corresponding to the provided space.")
        .def("eri", &ReferenceStateNew::eri,
             "Return the ERI (electron-repulsion integrals) tensor block corresponding "
             "to the provided space.")
        //
        .def("import_all", &ReferenceStateNew::import_all,
             "Normally the class only imports the Fock matrix blocks and "
             "electron-repulsion integrals of a particular space combination when this "
             "is requested by a call to above fock() or eri() functions. This function "
             "call, however, instructs the class to immediately import *all* such "
             "blocks. Typically you do not want to do this.")
        .def_property("cached_fock_blocks", &ReferenceStateNew::cached_fock_blocks,
                      &ReferenceStateNew::set_cached_fock_blocks,
                      "Get or set the list of momentarily cached Fock matrix blocks\n"
                      "\n"
                      "Setting this property allows to drop fock matrix blocks if they "
                      "are no longer needed to save memory.")
        .def_property("cached_eri_blocks", &ReferenceStateNew::cached_eri_blocks,
                      &ReferenceStateNew::set_cached_eri_blocks,
                      "Get or set the list of momentarily cached ERI tensor blocks\n"
                      "\n"
                      "Setting this property allows to drop ERI tensor blocks if they "
                      "are no longer needed to save memory.")
        .def("flush_hf_cache", &ReferenceStateNew::flush_hf_cache,
             "Tell the contained HartreeFockSolution_i object (which was passed upon "
             "construction), that a larger amount of import operations is done and that "
             "the next request for further imports will most likely take some time, such "
             "that intermediate caches can now be flushed to save some memory or other "
             "resources.")
        //
        .def("to_ctx", &ReferenceStateNew::to_ctx)
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
