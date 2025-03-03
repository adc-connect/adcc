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
#include "../HartreeFockSolution_i.hh"
#include "../exceptions.hh"
#include <array>
#include <vector>

namespace libadcc {
namespace tests {

/** Mock of the abstract interface which allows to test some
 *  of the features
 *
 * \note This class is *not* fully functional and not logically consistent.
 *  */
struct HFSolutionMock : public HartreeFockSolution_i {
  size_t exposed_n_orbs_alpha            = 7;
  size_t exposed_restricted              = true;
  size_t exposed_n_bas                   = 7;
  std::vector<size_t> exposed_occupation = {1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0};
  std::vector<double> exposed_orben      = {
        -2.0233397420161026e+01, -1.2657145550718050e+00, -6.2926710042596645e-01,
        -4.4166801612481776e-01, -3.8764506899048456e-01, 6.0283939648336604e-01,
        7.6591832861501685e-01,  -2.0233397420161026e+01, -1.2657145550718050e+00,
        -6.2926710042596645e-01, -4.4166801612481776e-01, -3.8764506899048456e-01,
        6.0283939648336604e-01,  7.6591832861501685e-01};

  size_t n_orbs_alpha() const override { return exposed_n_orbs_alpha; }
  real_type conv_tol() const override { return 1e-10; }
  bool restricted() const override { return exposed_restricted; }
  size_t spin_multiplicity() const override { return restricted() ? 1 : 0; }

  size_t n_bas() const override { return exposed_n_bas; }
  void nuclear_multipole(size_t /*order*/, scalar_type* /*buffer*/,
                         size_t /*size*/) const override {
    throw not_implemented_error("Not implemented.");
  }
  real_type energy_scf() const override { return 0; }
  void occupation_f(scalar_type* buffer, size_t size) const override {
    if (exposed_occupation.size() != size) {
      throw runtime_error(
            "Size mismatch between exposed_occupation and occupation_f buffer size.");
    }
    std::copy(exposed_occupation.begin(), exposed_occupation.end(), buffer);
  }
  void orbcoeff_fb(scalar_type* /* buffer */, size_t /* size */) const override {
    throw not_implemented_error("Not implemented.");
  }
  void orben_f(scalar_type* buffer, size_t size) const override {
    if (exposed_orben.size() != size) {
      throw runtime_error(
            "Size mismatch between exposed_occupation and occupation_f buffer size.");
    }
    std::copy(exposed_orben.begin(), exposed_orben.end(), buffer);
  }
  void fock_ff(size_t /*d1_start*/, size_t /*d1_end*/, size_t /*d2_start*/,
               size_t /*d2_end*/, size_t /* d1_stride */, size_t /* d2_stride */,
               scalar_type* /*buffer*/, size_t /*size*/) const override {
    throw not_implemented_error("Not implemented.");
  }

  void eri_ffff(size_t /*d1_start*/, size_t /*d1_end*/, size_t /*d2_start*/,
                size_t /*d2_end*/, size_t /*d3_start*/, size_t /*d3_end*/,
                size_t /*d4_start*/, size_t /*d4_end*/, size_t /*d1_stride*/,
                size_t /*d2_stride*/, size_t /*d3_stride*/, size_t /*d4_stride*/,
                scalar_type* /*buffer*/, size_t /*size*/) const override {
    throw not_implemented_error("Not implemented.");
  }

  void eri_phys_asym_ffff(size_t /*d1_start*/, size_t /*d1_end*/, size_t /*d2_start*/,
                          size_t /*d2_end*/, size_t /*d3_start*/, size_t /*d3_end*/,
                          size_t /*d4_start*/, size_t /*d4_end*/, size_t /*d1_stride*/,
                          size_t /*d2_stride*/, size_t /*d3_stride*/,
                          size_t /*d4_stride*/, scalar_type* /*buffer*/,
                          size_t /*size*/) const override {
    throw not_implemented_error("Not implemented.");
  }

  std::string backend() const override { return "mock"; }
};

}  // namespace tests
}  // namespace libadcc
