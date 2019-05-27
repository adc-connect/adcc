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

#include "hartree_fock_solution_hack.hh"
#include "util.hh"
#include <adcc/HartreeFockSolution_i.hh>
#include <adcc/exceptions.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

/** This class implements the translation from the C++ to the python world */
class HartreeFockProvider : public HartreeFockSolution_i {
 public:
  using HartreeFockSolution_i::HartreeFockSolution_i;
  virtual ~HartreeFockProvider() = default;

  //
  // Implementation of the C++ interface
  //
  std::string backend() const override { return get_backend(); }
  size_t n_alpha() const override { return get_n_alpha(); }
  size_t n_beta() const override { return get_n_beta(); }
  size_t n_orbs_alpha() const override { return get_n_orbs_alpha(); }
  size_t n_orbs_beta() const override { return get_n_orbs_beta(); }
  size_t n_bas() const override { return get_n_bas(); }
  real_type conv_tol() const override { return get_conv_tol(); }
  bool restricted() const override { return get_restricted(); }
  size_t spin_multiplicity() const override { return get_spin_multiplicity(); }
  real_type energy_scf() const override { return get_energy_scf(); }

  //
  // Translate C++-like interface to python-like interface
  //
  void occupation_f(scalar_type* buffer, size_t size) const override {
    const ssize_t ssize  = static_cast<ssize_t>(size);
    const ssize_t n_orbs = static_cast<ssize_t>(this->n_orbs());
    if (ssize != n_orbs) {
      throw dimension_mismatch("Buffer size (==" + std::to_string(ssize) +
                               ") does not agree with n_orbs (" + std::to_string(n_orbs) +
                               ").");
    }
    py::memoryview memview(py::buffer_info(buffer, n_orbs));
    fill_occupation_f(py::array({n_orbs}, buffer, memview));
  }

  void orben_f(scalar_type* buffer, size_t size) const override {
    const ssize_t ssize  = static_cast<ssize_t>(size);
    const ssize_t n_orbs = static_cast<ssize_t>(this->n_orbs());
    if (ssize != n_orbs) {
      throw dimension_mismatch("Buffer size (==" + std::to_string(ssize) +
                               ") does not agree with n_orbs (" + std::to_string(n_orbs) +
                               ").");
    }
    py::memoryview memview(py::buffer_info(buffer, n_orbs));
    fill_orben_f(py::array({n_orbs}, buffer, memview));
  }

  void orbcoeff_fb(scalar_type* buffer, size_t size) const override {
    const ssize_t ssize  = static_cast<ssize_t>(size);
    const ssize_t n_orbs = static_cast<ssize_t>(this->n_orbs());
    const ssize_t n_bas  = static_cast<ssize_t>(this->n_bas());
    if (ssize != n_orbs * n_bas) {
      throw dimension_mismatch("Buffer size (==" + std::to_string(ssize) +
                               ") does not agree with n_orbs*n_bas (" +
                               std::to_string(n_orbs * n_bas) + ").");
    }
    py::memoryview memview(py::buffer_info(buffer, n_orbs * n_bas));
    fill_orbcoeff_fb(py::array({n_orbs, n_bas}, buffer, memview));
  }

  void fock_ff(size_t d1_start, size_t d1_end, size_t d2_start, size_t d2_end,
               size_t d1_stride, size_t d2_stride, scalar_type* buffer,
               size_t size) const override {
    const ssize_t ssize = static_cast<ssize_t>(size);
    if (d1_end > n_orbs()) {
      throw out_of_range("End of first dimension (d1_end, == " + std::to_string(d1_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d2_end > n_orbs()) {
      throw out_of_range("End of second dimension (d2_end, == " + std::to_string(d2_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    const ssize_t d1_length = static_cast<ssize_t>(d1_end - d1_start);
    const ssize_t d2_length = static_cast<ssize_t>(d2_end - d2_start);
    if (ssize != d1_length * d2_length) {
      throw dimension_mismatch(
            "Expected length from range (== " + std::to_string(d1_length * d2_length) +
            ") does not agree with buffer size (== " + std::to_string(ssize) + ").");
    }

    if (size == 0) return;
    py::memoryview memview(py::buffer_info(buffer, d1_length * d2_length));
    std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(scalar_type) * d1_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d2_stride)};
    py::tuple slices = py::make_tuple(
          py::slice(static_cast<ssize_t>(d1_start), static_cast<ssize_t>(d1_end), 1),
          py::slice(static_cast<ssize_t>(d2_start), static_cast<ssize_t>(d2_end), 1));
    fill_fock_ff(slices, py::array({d1_length, d2_length}, strides, buffer, memview));
  }

  void eri_ffff(size_t d1_start, size_t d1_end, size_t d2_start, size_t d2_end,
                size_t d3_start, size_t d3_end, size_t d4_start, size_t d4_end,
                size_t d1_stride, size_t d2_stride, size_t d3_stride, size_t d4_stride,
                scalar_type* buffer, size_t size) const override {
    const ssize_t ssize = static_cast<ssize_t>(size);

    if (d1_end > n_orbs()) {
      throw out_of_range("End of first dimension (d1_end, == " + std::to_string(d1_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d2_end > n_orbs()) {
      throw out_of_range("End of second dimension (d2_end, == " + std::to_string(d2_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d3_end > n_orbs()) {
      throw out_of_range("End of third dimension (d3_end, == " + std::to_string(d3_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d4_end > n_orbs()) {
      throw out_of_range("End of fourth dimension (d4_end, == " + std::to_string(d4_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }

    const ssize_t d1_length = static_cast<ssize_t>(d1_end - d1_start);
    const ssize_t d2_length = static_cast<ssize_t>(d2_end - d2_start);
    const ssize_t d3_length = static_cast<ssize_t>(d3_end - d3_start);
    const ssize_t d4_length = static_cast<ssize_t>(d4_end - d4_start);
    if (ssize != d1_length * d2_length * d3_length * d4_length) {
      throw dimension_mismatch(
            "Expected length from range (== " +
            std::to_string(d1_length * d2_length * d3_length * d4_length) +
            ") does not agree with buffer size (== " + std::to_string(ssize) + ").");
    }

    if (size == 0) return;
    py::memoryview memview(
          py::buffer_info(buffer, d1_length * d2_length * d3_length * d4_length));
    std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(scalar_type) * d1_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d2_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d3_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d4_stride)};
    py::tuple slices = py::make_tuple(
          py::slice(static_cast<ssize_t>(d1_start), static_cast<ssize_t>(d1_end), 1),
          py::slice(static_cast<ssize_t>(d2_start), static_cast<ssize_t>(d2_end), 1),
          py::slice(static_cast<ssize_t>(d3_start), static_cast<ssize_t>(d3_end), 1),
          py::slice(static_cast<ssize_t>(d4_start), static_cast<ssize_t>(d4_end), 1));
    fill_eri_ffff(slices, py::array({d1_length, d2_length, d3_length, d4_length}, strides,
                                    buffer, memview));
  }

  void eri_phys_asym_ffff(size_t d1_start, size_t d1_end, size_t d2_start, size_t d2_end,
                          size_t d3_start, size_t d3_end, size_t d4_start, size_t d4_end,
                          size_t d1_stride, size_t d2_stride, size_t d3_stride,
                          size_t d4_stride, scalar_type* buffer,
                          size_t size) const override {
    const ssize_t ssize = static_cast<ssize_t>(size);

    if (d1_end > n_orbs()) {
      throw out_of_range("End of first dimension (d1_end, == " + std::to_string(d1_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d2_end > n_orbs()) {
      throw out_of_range("End of second dimension (d2_end, == " + std::to_string(d2_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d3_end > n_orbs()) {
      throw out_of_range("End of third dimension (d3_end, == " + std::to_string(d3_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }
    if (d4_end > n_orbs()) {
      throw out_of_range("End of fourth dimension (d4_end, == " + std::to_string(d4_end) +
                         " overshoots n_orbs() == " + std::to_string(n_orbs()));
    }

    const ssize_t d1_length = static_cast<ssize_t>(d1_end - d1_start);
    const ssize_t d2_length = static_cast<ssize_t>(d2_end - d2_start);
    const ssize_t d3_length = static_cast<ssize_t>(d3_end - d3_start);
    const ssize_t d4_length = static_cast<ssize_t>(d4_end - d4_start);
    if (ssize != d1_length * d2_length * d3_length * d4_length) {
      throw dimension_mismatch(
            "Expected length from range (== " +
            std::to_string(d1_length * d2_length * d3_length * d4_length) +
            ") does not agree with buffer size (== " + std::to_string(ssize) + ").");
    }

    // Skip empty ranges
    if (size == 0) return;
    py::memoryview memview(
          py::buffer_info(buffer, d1_length * d2_length * d3_length * d4_length));
    std::vector<ssize_t> strides{static_cast<ssize_t>(sizeof(scalar_type) * d1_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d2_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d3_stride),
                                 static_cast<ssize_t>(sizeof(scalar_type) * d4_stride)};
    py::tuple slices = py::make_tuple(
          py::slice(static_cast<ssize_t>(d1_start), static_cast<ssize_t>(d1_end), 1),
          py::slice(static_cast<ssize_t>(d2_start), static_cast<ssize_t>(d2_end), 1),
          py::slice(static_cast<ssize_t>(d3_start), static_cast<ssize_t>(d3_end), 1),
          py::slice(static_cast<ssize_t>(d4_start), static_cast<ssize_t>(d4_end), 1));
    fill_eri_phys_asym_ffff(
          slices, py::array({d1_length, d2_length, d3_length, d4_length}, strides, buffer,
                            memview));
  }

  //
  // Interface for the python world
  //
  virtual size_t get_n_alpha() const           = 0;
  virtual size_t get_n_beta() const            = 0;
  virtual size_t get_n_orbs_alpha() const      = 0;
  virtual size_t get_n_orbs_beta() const       = 0;
  virtual size_t get_n_bas() const             = 0;
  virtual real_type get_conv_tol() const       = 0;
  virtual bool get_restricted() const          = 0;
  virtual size_t get_spin_multiplicity() const = 0;
  virtual real_type get_energy_scf() const     = 0;
  virtual std::string get_backend() const      = 0;

  virtual void fill_occupation_f(py::array out) const                         = 0;
  virtual void fill_orben_f(py::array out) const                              = 0;
  virtual void fill_orbcoeff_fb(py::array out) const                          = 0;
  virtual void fill_fock_ff(py::tuple, py::array out) const                   = 0;
  virtual void fill_eri_ffff(py::tuple slices, py::array out) const           = 0;
  virtual void fill_eri_phys_asym_ffff(py::tuple slices, py::array out) const = 0;
};

/** This implements the trampoline for C++ to call the python functions
 *  when needed */
class PyHartreeFockProvider : public HartreeFockProvider {
 public:
  using HartreeFockProvider::HartreeFockProvider;
  virtual ~PyHartreeFockProvider() = default;

  size_t get_n_alpha() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_alpha, );
  }
  size_t get_n_beta() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_beta, );
  }
  size_t get_n_orbs_alpha() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_orbs_alpha, );
  }
  size_t get_n_orbs_beta() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_orbs_beta, );
  }
  size_t get_n_bas() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_bas, );
  }
  real_type get_conv_tol() const override {
    PYBIND11_OVERLOAD_PURE(real_type, HartreeFockProvider, get_conv_tol, );
  }
  bool get_restricted() const override {
    PYBIND11_OVERLOAD_PURE(bool, HartreeFockProvider, get_restricted, );
  }
  size_t get_spin_multiplicity() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_spin_multiplicity, );
  }
  real_type get_energy_scf() const override {
    PYBIND11_OVERLOAD_PURE(real_type, HartreeFockProvider, get_energy_scf, );
  }
  void fill_occupation_f(py::array out) const override {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_occupation_f, out);
  }
  void fill_orben_f(py::array out) const override {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_orben_f, out);
  }
  void fill_orbcoeff_fb(py::array out) const override {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_orbcoeff_fb, out);
  }
  void fill_fock_ff(py::tuple slices, py::array out) const override {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_fock_ff, slices, out);
  }
  void fill_eri_ffff(py::tuple slices, py::array out) const {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_eri_ffff, slices, out);
  }
  void fill_eri_phys_asym_ffff(py::tuple slices, py::array out) const {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_eri_phys_asym_ffff, slices,
                           out);
  }
  bool has_eri_phys_asym_ffff() const override {
    PYBIND11_OVERLOAD(bool, HartreeFockProvider, has_eri_phys_asym_ffff, );
  }
  std::string get_backend() const override {
    PYBIND11_OVERLOAD_PURE(std::string, HartreeFockProvider, get_backend, );
  }
  void flush_cache() const override {
    PYBIND11_OVERLOAD(void, HartreeFockProvider, flush_cache, );
  }
};

static py::array_t<scalar_type> HartreeFockSolution_i_occupation_f(
      const HartreeFockSolution_i& self) {
  py::array_t<scalar_type> ret(self.n_orbs());
  self.occupation_f(ret.mutable_data(), self.n_orbs());
  return ret;
}

static py::array_t<scalar_type> HartreeFockSolution_i_orben_f(
      const HartreeFockSolution_i& self) {
  py::array_t<scalar_type> ret(self.n_orbs());
  self.orben_f(ret.mutable_data(), self.n_orbs());
  return ret;
}

static py::array_t<scalar_type> HartreeFockSolution_i_orbcoeff_fb(
      const HartreeFockSolution_i& self) {
  py::array_t<scalar_type> ret({self.n_orbs(), self.n_bas()});
  self.orbcoeff_fb(ret.mutable_data(), self.n_orbs() * self.n_bas());
  return ret;
}

static py::array_t<scalar_type> HartreeFockSolution_i_fock_ff(
      const HartreeFockSolution_i& self) {
  py::array_t<scalar_type> ret({self.n_orbs(), self.n_orbs()});
  self.fock_ff(0, self.n_orbs(), 0, self.n_orbs(),
               static_cast<size_t>(ret.strides(0)) / sizeof(scalar_type),
               static_cast<size_t>(ret.strides(1)) / sizeof(scalar_type),
               ret.mutable_data(), self.n_orbs() * self.n_orbs());
  return ret;
}

/** Exports adcc/HartreeFockSolution_i.hh and HartreeFockProvider to python */
void export_HartreeFockProvider(py::module& m) {
  py::class_<adcc::HartreeFockSolution_i, std::shared_ptr<adcc::HartreeFockSolution_i>>
        hfdata_i(m, "HartreeFockSolution_i",
                 "Interface class representing the data expected in adcc from an "
                 "interfacing HF / SCF program.");
  hfdata_i.def_property_readonly("n_alpha", &HartreeFockSolution_i::n_alpha)
        .def_property_readonly("n_beta", &HartreeFockSolution_i::n_beta)
        .def_property_readonly("conv_tol", &HartreeFockSolution_i::conv_tol)
        .def_property_readonly("restricted", &HartreeFockSolution_i::restricted)
        .def_property_readonly("energy_scf", &HartreeFockSolution_i::energy_scf)
        .def_property_readonly("spin_multiplicity",
                               &HartreeFockSolution_i::spin_multiplicity)
        .def_property_readonly("n_orbs_alpha", &HartreeFockSolution_i::n_orbs_alpha)
        .def_property_readonly("n_orbs_beta", &HartreeFockSolution_i::n_orbs_beta)
        .def_property_readonly("n_orbs", &HartreeFockSolution_i::n_orbs)
        .def_property_readonly("n_bas", &HartreeFockSolution_i::n_bas)
        .def_property_readonly("backend", &HartreeFockSolution_i::backend)
        //
        .def_property_readonly("orben_f", &HartreeFockSolution_i_orben_f)
        .def_property_readonly("occupation_f", &HartreeFockSolution_i_occupation_f)
        .def_property_readonly("orbcoeff_fb", &HartreeFockSolution_i_orbcoeff_fb)
        .def_property_readonly("fock_ff", &HartreeFockSolution_i_fock_ff)
        //
        ;

  py::class_<HartreeFockProvider, std::shared_ptr<HartreeFockProvider>,
             PyHartreeFockProvider>(m, "HartreeFockProvider", hfdata_i,
                                    "Class for providing the data expected from adcc "
                                    "directly via overloading python functions.")
        .def(py::init<>())
        // Getter functions to be overwritten
        .def("get_n_alpha", &HartreeFockProvider::get_n_alpha,
             "Overwrite in python to return the number of alpha electrons.")
        .def("get_n_beta", &HartreeFockProvider::get_n_beta,
             "Overwrite in python to return the number fo beta electrons.")
        .def("get_conv_tol", &HartreeFockProvider::get_conv_tol,
             "Overwrite in python to return the scf threshold.")
        .def("get_restricted", &HartreeFockProvider::get_restricted,
             "Overwrite in python to return whether the calculation is restricted (true) "
             "or unrestricted(false)")
        .def("get_energy_scf", &HartreeFockProvider::get_energy_scf,
             "Overwrite to return the final SCF energy")
        .def("get_spin_multiplicity", &HartreeFockProvider::get_spin_multiplicity,
             "Overwrite to return the spin multiplicity of the calculation")
        .def("get_n_orbs_alpha", &HartreeFockProvider::get_n_orbs_alpha,
             "Overwrite to return the number of alpha orbitals")
        .def("get_n_orbs_beta", &HartreeFockProvider::get_n_orbs_beta,
             "Overwrite to return the number of beta orbitals")
        .def("get_n_bas", &HartreeFockProvider::get_n_bas,
             "Overwrite to return the number of basis functions")
        //
        .def("fill_occupation_f", &HartreeFockProvider::fill_orben_f,
             "Overwrite to fill the passed numpy array with the occupation numbers")
        .def("fill_orben_f", &HartreeFockProvider::fill_orben_f,
             "Overwrite to fill the passed numpy array with the orbital energies")
        .def("fill_orbcoeff_fb", &HartreeFockProvider::fill_orbcoeff_fb,
             "Overwrite to fill the passed numpy array with orbital coefficients")
        .def("fill_fock_ff", &HartreeFockProvider::fill_fock_ff,
             "Overwrite to fill the passed numpy array with part of the fock matrix")
        .def("fill_eri_ffff", &HartreeFockProvider::fill_eri_ffff,
             "Overwrite to fill the passed numpy array with a part of the eri tensor (in "
             "chemist's notation)")
        .def("fill_eri_phys_asym_ffff", &HartreeFockProvider::fill_eri_phys_asym_ffff,
             "Overwrite to fill the passed numpy array with part of the anti-symmetrised "
             "eri tensor (in physicist's notation)")
        .def("has_eri_phys_asym_ffff", &HartreeFockProvider::has_eri_phys_asym_ffff,
             "Does the overwriting class have a routine for directly providing the eri "
             "tensor in anti-symmetrised form (i.e. is fill_eri_phys_asym_ffff "
             "implemented).")
        .def("flush_cache", &HartreeFockProvider::flush_cache,
             "This function is called to signal that potential cached data could now be "
             "flushed to save memory or other resources.\nThis is called by adcc as soon "
             "as the import process is largely finished and can be used to purge e.g. "
             "intermediates for the computation of eri tensor data.")
        //
        ;
}

}  // namespace py_iface
}  // namespace adcc
