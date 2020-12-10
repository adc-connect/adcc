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

#include "../HartreeFockSolution_i.hh"
#include "../exceptions.hh"
#include "hartree_fock_solution_hack.hh"
#include "util.hh"
#include <pybind11/pybind11.h>

namespace libadcc {

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
  size_t n_orbs_alpha() const override { return get_n_orbs_alpha(); }
  size_t n_bas() const override { return get_n_bas(); }
  real_type conv_tol() const override { return get_conv_tol(); }
  bool restricted() const override { return get_restricted(); }
  size_t spin_multiplicity() const override { return get_spin_multiplicity(); }
  real_type energy_scf() const override { return get_energy_scf(); }

  //
  // Translate C++-like interface to python-like interface
  //
  void nuclear_multipole(size_t order, scalar_type* buffer, size_t size) const override {
    py::array_t<scalar_type> ret = get_nuclear_multipole(order);
    if (static_cast<ssize_t>(size) != ret.size()) {
      throw dimension_mismatch("Array size (==" + std::to_string(ret.size()) +
                               ") does not agree with buffer size (" +
                               std::to_string(size) + ").");
    }
    std::copy(ret.data(), ret.data() + size, buffer);
  }

  void occupation_f(scalar_type* buffer, size_t size) const override {
    const ssize_t ssize  = static_cast<ssize_t>(size);
    const ssize_t n_orbs = static_cast<ssize_t>(this->n_orbs());
    if (ssize != n_orbs) {
      throw dimension_mismatch("Buffer size (==" + std::to_string(ssize) +
                               ") does not agree with n_orbs (" + std::to_string(n_orbs) +
                               ").");
    }
    auto memview = py::memoryview::from_memory(buffer, n_orbs * sizeof(scalar_type));
    fill_occupation_f(py::array(std::vector<ssize_t>{n_orbs}, buffer, memview));
  }

  void orben_f(scalar_type* buffer, size_t size) const override {
    const ssize_t ssize  = static_cast<ssize_t>(size);
    const ssize_t n_orbs = static_cast<ssize_t>(this->n_orbs());
    if (ssize != n_orbs) {
      throw dimension_mismatch("Buffer size (==" + std::to_string(ssize) +
                               ") does not agree with n_orbs (" + std::to_string(n_orbs) +
                               ").");
    }
    auto memview = py::memoryview::from_memory(buffer, n_orbs * sizeof(scalar_type));
    fill_orben_f(py::array(std::vector<ssize_t>{n_orbs}, buffer, memview));
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
    auto memview =
          py::memoryview::from_memory(buffer, n_orbs * n_bas * sizeof(scalar_type));
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
    if (d1_length == 0 || d2_length == 0) return;

    const ssize_t sd1_stride = static_cast<ssize_t>(d1_stride);
    const ssize_t sd2_stride = static_cast<ssize_t>(d2_stride);
    const ssize_t slength = (d1_length - 1) * sd1_stride + (d2_length - 1) * sd2_stride;
    if (ssize < slength) {
      throw dimension_mismatch(
            "Expected length from range and strides (== " + std::to_string(slength) +
            ") overshoots buffer size (== " + std::to_string(ssize) + ").");
    }
    auto memview = py::memoryview::from_memory(
          buffer, d1_length * d2_length * sizeof(scalar_type));
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
    if (d1_length == 0 || d2_length == 0 || d3_length == 0 || d4_length == 0) return;

    const ssize_t sd1_stride = static_cast<ssize_t>(d1_stride);
    const ssize_t sd2_stride = static_cast<ssize_t>(d2_stride);
    const ssize_t sd3_stride = static_cast<ssize_t>(d3_stride);
    const ssize_t sd4_stride = static_cast<ssize_t>(d4_stride);
    const ssize_t slength = (d1_length - 1) * sd1_stride + (d2_length - 1) * sd2_stride +
                            (d3_length - 1) * sd3_stride + (d4_length - 1) * sd4_stride;
    if (ssize < slength) {
      throw dimension_mismatch(
            "Expected length from range and strides (== " + std::to_string(slength) +
            ") overshoots buffer size (== " + std::to_string(ssize) + ").");
    }

    auto memview = py::memoryview::from_memory(
          buffer, d1_length * d2_length * d3_length * d4_length * sizeof(scalar_type));
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
    if (d1_length == 0 || d2_length == 0 || d3_length == 0 || d4_length == 0) return;

    const ssize_t sd1_stride = static_cast<ssize_t>(d1_stride);
    const ssize_t sd2_stride = static_cast<ssize_t>(d2_stride);
    const ssize_t sd3_stride = static_cast<ssize_t>(d3_stride);
    const ssize_t sd4_stride = static_cast<ssize_t>(d4_stride);
    const ssize_t slength = (d1_length - 1) * sd1_stride + (d2_length - 1) * sd2_stride +
                            (d3_length - 1) * sd3_stride + (d4_length - 1) * sd4_stride;
    if (ssize < slength) {
      throw dimension_mismatch(
            "Expected length from range and strides (== " + std::to_string(slength) +
            ") overshoots buffer size (== " + std::to_string(ssize) + ").");
    }

    auto memview = py::memoryview::from_memory(
          buffer, d1_length * d2_length * d3_length * d4_length * sizeof(scalar_type));
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
  virtual size_t get_n_orbs_alpha() const                                    = 0;
  virtual size_t get_n_bas() const                                           = 0;
  virtual py::array_t<scalar_type> get_nuclear_multipole(size_t order) const = 0;
  virtual real_type get_conv_tol() const                                     = 0;
  virtual bool get_restricted() const                                        = 0;
  virtual size_t get_spin_multiplicity() const                               = 0;
  virtual real_type get_energy_scf() const                                   = 0;
  virtual std::string get_backend() const                                    = 0;

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

  size_t get_n_orbs_alpha() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_orbs_alpha, );
  }
  size_t get_n_bas() const override {
    PYBIND11_OVERLOAD_PURE(size_t, HartreeFockProvider, get_n_bas, );
  }
  py::array_t<scalar_type> get_nuclear_multipole(size_t order) const override {
    PYBIND11_OVERLOAD_PURE(py::array_t<scalar_type>, HartreeFockProvider,
                           get_nuclear_multipole, order);
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
  void fill_eri_ffff(py::tuple slices, py::array out) const override {
    PYBIND11_OVERLOAD_PURE(void, HartreeFockProvider, fill_eri_ffff, slices, out);
  }
  void fill_eri_phys_asym_ffff(py::tuple slices, py::array out) const override {
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

static size_t count_electrons(const HartreeFockSolution_i& self, bool count_beta) {
  const py::array_t<scalar_type> occupation = HartreeFockSolution_i_occupation_f(self);
  const size_t first                        = count_beta ? self.n_orbs_alpha() : 0;
  const size_t last = count_beta ? self.n_orbs() : self.n_orbs_alpha();

  size_t ret = 0;
  for (size_t i = first; i < last; ++i) {
    if (std::fabs(occupation.at(i) - 1.0) < 1e-12) {
      ret += 1;
    } else if (std::fabs(occupation.at(i)) > 1e-12) {
      throw invalid_argument("Occupation value " + std::to_string(occupation.at(i)) +
                             "for orbital " + std::to_string(i) +
                             " is invalid, since neither zero nor one.");
    }
  }
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

void export_HartreeFockProvider(py::module& m) {
  py::class_<HartreeFockSolution_i, std::shared_ptr<HartreeFockSolution_i>> hfdata_i(
        m, "HartreeFockSolution_i",
        "Interface class representing the data expected in adcc from an "
        "interfacing HF / SCF program. Python binding to "
        ":cpp:class:`HartreeFockSolution_i`");
  hfdata_i
        // TODO n_alpha and n_beta are kind of expensive like this
        //      and maybe should be removed for this reason
        .def_property_readonly("n_alpha",
                               [](const HartreeFockSolution_i& self) {
                                 return count_electrons(self, /* count_beta = */ false);
                               })
        .def_property_readonly("n_beta",
                               [](const HartreeFockSolution_i& self) {
                                 return count_electrons(self, /* count_beta = */ true);
                               })
        .def_property_readonly("conv_tol", &HartreeFockSolution_i::conv_tol)
        .def_property_readonly("restricted", &HartreeFockSolution_i::restricted)
        .def_property_readonly("energy_scf", &HartreeFockSolution_i::energy_scf)
        .def_property_readonly("spin_multiplicity",
                               &HartreeFockSolution_i::spin_multiplicity)
        .def_property_readonly("n_orbs_alpha", &HartreeFockSolution_i::n_orbs_alpha)
        .def_property_readonly("n_orbs_beta", &HartreeFockSolution_i::n_orbs_alpha)
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
             PyHartreeFockProvider>(
        m, "HartreeFockProvider", hfdata_i,
        "Abstract class defining the interface for passing data from the host program to "
        "adcc. All functions of this class need to be overwritten explicitly from "
        "python.\nIn the remaining documentation we denote with `nf` the value returned "
        "by `get_n_orbs_alpha()` and with `nb` the value returned by "
        "`get_nbas()`.")
        .def(py::init<>())
        //
        .def("get_conv_tol", &HartreeFockProvider::get_conv_tol,
             "Returns the tolerance value used for SCF convergence. Should be roughly "
             "equivalent to the l2 norm of the Pulay error.")
        .def("get_restricted", &HartreeFockProvider::get_restricted,
             "Return *True* for a restricted SCF calculation, *False* otherwise.")
        .def("get_energy_scf", &HartreeFockProvider::get_energy_scf,
             "Returns the final total SCF energy (sum of electronic and nuclear terms.")
        .def("get_spin_multiplicity", &HartreeFockProvider::get_spin_multiplicity,
             "Returns the spin multiplicity of the HF ground state. A value of 0* (for "
             "unknown) should be supplied for unrestricted calculations.")
        .def("get_n_orbs_alpha", &HartreeFockProvider::get_n_orbs_alpha,
             "Returns the number of HF *spin* orbitals of alpha spin. It is assumed the "
             "same number of beta spin orbitals are used. This value is abbreviated by "
             "`nf` in the documentation.")
        .def("get_n_bas", &HartreeFockProvider::get_n_bas,
             "Returns the number of *spatial* one-electron basis functions. This value "
             "is abbreviated by `nb` in the documentation.")
        .def("get_nuclear_multipole", &HartreeFockProvider::get_nuclear_multipole,
             "Returns the nuclear multipole of the requested order. For `0` returns the "
             "total nuclear charge as an array of size 1, for `1` returns the nuclear "
             "dipole moment as an array of size 3.")
        //
        .def("fill_occupation_f", &HartreeFockProvider::fill_orben_f,
             "Fill the passed numpy array of size `(2 * nf, )` with the occupation "
             "number for each SCF orbital.")
        .def("fill_orben_f", &HartreeFockProvider::fill_orben_f,
             "Fill the passed numpy array of size `(2 * nf, )` with the SCF orbital "
             "energies.")
        .def("fill_orbcoeff_fb", &HartreeFockProvider::fill_orbcoeff_fb,
             "Fill the passed numpy array of size `(2 * nf, nb)` with the SCF orbital "
             "coefficients, i.e. the uniform transform from the one-particle basis to "
             "the molecular orbitals.")
        .def("fill_fock_ff", &HartreeFockProvider::fill_fock_ff,
             "Fill the passed numpy array `arg1` with a part of the Fock matrix in the "
             "molecular orbital basis. The block to store is specified by the provided "
             "tuple of ranges `arg0`, which gives the range of indices to place into the "
             "buffer along each of the axis. The index counting is done in spin "
             "orbitals, so the full range in each axis is `range(0, 2 * nf)`. The "
             "implementation should not assume that the alpha-beta and beta-alpha blocks "
             "are not accessed even though they are zero by spin symmetry.")
        .def("fill_eri_ffff", &HartreeFockProvider::fill_eri_ffff,
             "Fill the passed numpy array `arg1` with a part of the electron-repulsion "
             "integral tensor in the molecular orbital basis. "
             "The indexing convention is the chemist's notation, i.e. the index tuple "
             "`(i,j,k,l)` refers to the integral :math:`(ij|kl)`. "
             "The block to store is specified by the provided "
             "tuple of ranges `arg0`, which gives the range of indices to place into the "
             "buffer along each of the axis. The index counting is done in spin "
             "orbitals, so the full range in each axis is `range(0, 2 * nf)`.")
        .def("fill_eri_phys_asym_ffff", &HartreeFockProvider::fill_eri_phys_asym_ffff,
             "Fill the passed numpy array `arg1` with a part of the **antisymmetrised** "
             "electron-repulsion integral tensor in the molecular orbital basis. "
             "The indexing convention is the physicist's notation, i.e. the index tuple "
             "`(i,j,k,l)` refers to the integral :math:`\\langle ij||kl \\rangle`. "
             "The block to store is specified by the provided "
             "tuple of ranges `arg0`, which gives the range of indices to place into the "
             "buffer along each of the axis. The index counting is done in spin "
             "orbitals, so the full range in each axis is `range(0, 2 * nf)`.")
        .def("has_eri_phys_asym_ffff", &HartreeFockProvider::has_eri_phys_asym_ffff,
             "Returns whether `fill_eri_phys_asym_ffff` function is implemented and "
             "should be used(*True*) or whether antisymmetrisation should be done inside "
             "adcc starting from the `fill_eri_ffff` function (*False*)")
        .def("flush_cache", &HartreeFockProvider::flush_cache,
             "This function is called to signal that potential cached data could now be "
             "flushed to save memory or other resources.\nThis can be used to purge e.g. "
             "intermediates for the computation of electron-repulsion integral tensor "
             "data.")
        //
        ;
}

}  // namespace libadcc
