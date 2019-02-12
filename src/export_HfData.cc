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

#include "util.hh"
#include <adcc/HfData.hh>
#include <adcc/exceptions.hh>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

// TODO It makes more sense to have a specialisiation of HfData for
//      numpy objects in here and expose that to python instead of
//      the bare HfData. This would also prevent the memory issues
//      we have at the moment (see notes in the various functions)

namespace {

template <typename T>
void copy_dict_to_elem(const py::dict& d, std::string key, T& elem,
                       bool required = false) {
  if (d.contains(py::cast(key))) {
    elem = d[py::cast(key)].cast<T>();
  } else if (required) {
    throw invalid_argument("Missing required key '" + key + "' from python dict");
  }
}

void copy_array_data_ptr(const py::dict& d, std::string key, const scalar_type*& data,
                         const std::vector<size_t>& expected_shape,
                         bool required = false) {
  if (d.contains(py::cast(key))) {
    py::array array = d[py::cast(key)];
    try {
      data = extract_array_data<scalar_type>(array, expected_shape);
    } catch (invalid_argument& e) {
      throw invalid_argument("Error parsing numpy array associated with key " + key +
                             ":" + std::string(e.what()));
    }
  } else if (required) {
    throw invalid_argument("Missing required key '" + key + "' from python dict.");
  }
}

}  // namespace

static std::shared_ptr<HfData> HfData_from_dict(const py::dict& d) {
  auto ret = std::make_shared<HfData>();
  copy_dict_to_elem(d, "n_alpha", ret->m_n_alpha, /*required*/ true);
  copy_dict_to_elem(d, "n_beta", ret->m_n_beta, /*required*/ true);
  copy_dict_to_elem(d, "n_beta", ret->m_n_beta, /*required*/ true);
  copy_dict_to_elem(d, "threshold", ret->m_threshold);
  copy_dict_to_elem(d, "restricted", ret->m_restricted, /*required*/ true);
  copy_dict_to_elem(d, "spin_multiplicity", ret->m_spin_multiplicity, /*required*/ true);
  copy_dict_to_elem(d, "energy_scf", ret->m_energy_scf);
  copy_dict_to_elem(d, "energy_nuclear_repulsion", ret->m_energy_nuclear_repulsion,
                    /* required */ true);
  copy_dict_to_elem(d, "energy_nuclear_attraction", ret->m_energy_nuclear_attraction);
  copy_dict_to_elem(d, "energy_coulomb", ret->m_energy_coulomb);
  copy_dict_to_elem(d, "energy_exchange_alpha", ret->m_energy_exchange_alpha);
  copy_dict_to_elem(d, "energy_exchange_beta", ret->m_energy_exchange_beta);
  copy_dict_to_elem(d, "energy_kinetic", ret->m_energy_kinetic);

  copy_dict_to_elem(d, "n_orbs_alpha", ret->m_n_orbs_alpha, /*required*/ true);
  copy_dict_to_elem(d, "n_orbs_beta", ret->m_n_orbs_beta, /*required*/ true);
  copy_dict_to_elem(d, "n_bas", ret->m_n_bas, /*required*/ true);

  // TODO Keep original numpy array as a pointer such that we have an owner
  //      for when returning the data.
  copy_array_data_ptr(d, "orbcoeff_fb", ret->m_orbcoeff_fb_ptr,
                      {ret->n_orbs(), ret->n_bas()}, /*required*/ true);
  copy_array_data_ptr(d, "orben_f", ret->m_orben_f_ptr, {ret->n_orbs()},
                      /*required*/ true);
  copy_array_data_ptr(d, "fock_ff", ret->m_fock_ff_ptr, {ret->n_orbs(), ret->n_orbs()},
                      /*required*/ true);
  copy_array_data_ptr(d, "eri_ffff", ret->m_eri_ffff_ptr,
                      {ret->n_orbs(), ret->n_orbs(), ret->n_orbs(), ret->n_orbs()},
                      /*required*/ false);
  copy_array_data_ptr(d, "eri_phys_asym_ffff", ret->m_eri_phys_asym_ffff_ptr,
                      {ret->n_orbs(), ret->n_orbs(), ret->n_orbs(), ret->n_orbs()},
                      /*required*/ false);

  if (ret->m_eri_phys_asym_ffff_ptr == nullptr && ret->m_eri_ffff_ptr == nullptr) {
    throw invalid_argument(
          "At least one of the keys 'eri_phys_asym_ffff' (to give the antisymmetric "
          "electron repulsion integrals in the physicists indexing convention) or "
          "eri_ffff (to give the electron repulsion integrals in chemists' convention) "
          "is required.");
  }

  return ret;
}

static py::dict HfData_to_dict(const HfData& self) {
  throw not_implemented_error("not yet implemented.");
  (void)self;
  // TODO
}

static py::array HfData_orbcoeff_fb(const HfData& self) {
  // TODO Right now produces a segfault -> return owner
  return make_array(self.m_orbcoeff_fb_ptr, {self.n_orbs(), self.n_bas()});
}

static py::array HfData_orben_f(const HfData& self) {
  // TODO Right now produces a segfault -> return owner
  return make_array(self.m_orben_f_ptr, {self.n_orbs()});
}

static py::array HfData_fock_ff(const HfData& self) {
  // TODO Right now produces a segfault -> return owner
  return make_array(self.m_orben_f_ptr, {self.n_orbs(), self.n_orbs()});
}

static py::array HfData_eri_ffff(const HfData& self) {
  // TODO Right now produces a segfault -> return owner
  return make_array(self.m_eri_ffff_ptr,
                    {self.n_orbs(), self.n_orbs(), self.n_orbs(), self.n_orbs()});
}

static py::array HfData_eri_phys_asym_ffff(const HfData& self) {
  // TODO Right now produces a segfault -> return owner
  return make_array(self.m_eri_phys_asym_ffff_ptr,
                    {self.n_orbs(), self.n_orbs(), self.n_orbs(), self.n_orbs()});
}

void export_HfData(py::module& m) {
  // TODO Make the constructor from python dictionary the only way to make this class
  py::class_<HfData, std::shared_ptr<HfData>, HartreeFockSolution_i>(
        m, "HfData", "Class supplying an HF solution as plain data to adcc.",
        py::dynamic_attr())
        .def_readwrite("n_alpha", &HfData::m_n_alpha)
        .def_readwrite("n_beta", &HfData::m_n_beta)
        .def_readwrite("threshold", &HfData::m_threshold)
        .def_readwrite("restricted", &HfData::m_restricted)
        .def_readwrite("energy_scf", &HfData::m_energy_scf)
        .def_readwrite("energy_nuclear_repulsion", &HfData::m_energy_nuclear_repulsion)
        .def_readwrite("energy_coulomb", &HfData::m_energy_coulomb)
        .def_readwrite("energy_exchange_alpha", &HfData::m_energy_exchange_alpha)
        .def_readwrite("energy_exchange_beta", &HfData::m_energy_exchange_beta)
        .def_readwrite("energy_nuclear_attraction", &HfData::m_energy_nuclear_attraction)
        .def_readwrite("energy_kinetic", &HfData::m_energy_kinetic)
        .def_readwrite("spin_multiplicity", &HfData::m_spin_multiplicity)
        .def_property_readonly("n_orbs_alpha", &HfData::n_orbs_alpha)
        .def_property_readonly("n_orbs_beta", &HfData::n_orbs_beta)
        .def_property_readonly("n_orbs", &HfData::n_orbs)
        .def_property_readonly("n_bas", &HfData::n_bas)
        .def_property_readonly("orbcoeff_fb", &HfData_orbcoeff_fb)
        .def_property_readonly("orben_f", &HfData_orben_f)
        .def_property_readonly("fock_ff", &HfData_fock_ff)
        .def_property_readonly("eri_ffff", &HfData_eri_ffff)
        .def_property_readonly("eri_phys_asym_ffff", &HfData_eri_phys_asym_ffff)
        .def("to_dict", &HfData_to_dict)
        .def_static("from_dict", &HfData_from_dict,
                    "Make a HfData object from a python dictionary. Unknown keys will be "
                    "silently ignored.");
}

}  // namespace py_iface
}  // namespace adcc
