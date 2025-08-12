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

#include "amplitude_vector_enforce_spin_kind.hh"
#include "TensorImpl.hh"
#include "exceptions.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/core/short_orbit.h>
#pragma GCC visibility pop

namespace libadcc {

namespace lt = libtensor;

void amplitude_vector_enforce_spin_kind(std::shared_ptr<Tensor> doubles_tensor,
                                        std::string block, std::string spin_kind) {
  // Nothing to do for singles block
  if (block == "s") return;

  if (block != "d") {
    throw not_implemented_error("Not implemented for block != 'd'");
  }

  //
  // Note: This is not necessary for triplets at all, since
  //       there the antisymmetric mapping setup in the guesses
  //       is already sufficient to separate them from the singlets
  //       and quintets
  //
  //       Even for the singlets this is only to separate the singlets
  //       from the quintets, since the triplet contribution is already
  //       gone due to the antisymmetry setup.
  //
  //       This routine assumes that the incoming guess vectors are already
  //       properly setup, otherwise the symmetry implicitly assumed here
  //       does not hold. So: Make sure to use guess_zero and
  //       guesses_from_diagonal with the appropriate parameters to set up
  //       your guesses.
  //
  if (spin_kind == "triplet") {
    // If the doubles part is anti-symmetric with respect to
    // the spin blocks, i.e. guesses with
    // AdcGuessKind.spin_block_symmetrisation == "antisymmetric"
    // are constructed, we are all set here.
    // TODO Enforce this via an assertion or an explicit check.
    return;
  }

  if (spin_kind != "singlet") {
    throw not_implemented_error(
          "Only implemented for spin_kind == 'singlet' and spin_kind == "
          "'triplet'.");
  }

  auto& u2 = asbt4(doubles_tensor);
  lt::block_tensor_ctrl<4, scalar_type> ctrl(u2);
  const lt::symmetry<4, scalar_type>& sym = ctrl.req_const_symmetry();

  // Extract the number of blocks per dimension
  const lt::block_index_space<4>& bis = sym.get_bis();
  lt::dimensions<4> bidims(bis.get_block_index_dims());

  // Setup i1 to point to 0,0,0,0 and i2 to the half of the
  // full number of blocks, i.e. to the alpha blocks in each
  // dimension only.
  lt::index<4> i1, i2;
  for (size_t i = 0; i < 4; i++) i2[i] = bidims[i] / 2 - 1;

  // Index range over all alpha-alpha-alpha-alpha blocks
  // in all point group symmetries
  const lt::index_range<4> index_range_alpha(i1, i2);

  // This dimensions object contains the number of alpha blocks per dimension
  lt::dimensions<4> bidims_alpha(index_range_alpha);

  // Iterate over all alpha-alpha-alpha-alpha blocks
  lt::abs_index<4> ai(bidims_alpha);
  do {
    // This gives the block index tuple
    const lt::index<4>& ii = ai.get_index();

    // Construct the orbit corresponding to this index (i.e. the iterator
    // running over all elements equivalent by symmetry
    // Ignore spin-forbidden, i.e. zero orbits
    lt::short_orbit<4, scalar_type> orbi(sym, ii, /* compute_if_allowed_orbit = */ true);
    if (!orbi.is_allowed()) continue;

    // get_acindex -> get absolute canonical index
    // Continue if our current index is larger than the canonical index
    if (orbi.get_acindex() < ai.get_abs_index()) continue;

    // TODO This might be wrong ... think about it and talk to Adrian
    //      the point is that orbi might have other strides than bidims_alpha
    //      Continue if the canonical index is already past the
    //      alpha-alpha-alpha-alpha block
    if (orbi.get_acindex() > bidims_alpha.get_size()) continue;

    // Get the index tuple of the canonical block of (alpha, alpha, alpha, alpha)
    const lt::index<4>& ci = orbi.get_cindex();

    // set i1 to (alpha, beta, alpha, beta) equivalent of the canonical index
    //  pinned by ai and orbi
    lt::index<4> i1(ci);
    i1[1] += bidims_alpha[1];
    i1[3] += bidims_alpha[3];

    // set i2 to (alpha, beta, beta, alpha)
    lt::index<4> i2(ci);
    i2[1] += bidims_alpha[1];
    i2[2] += bidims_alpha[2];

    //
    // What the following code does is that it keeps the spin projection (S^2)
    // properly, provided that the symmetry setup is done as in
    // contrib/adc_pp/adc_guess_d.C. It assumes the coefficients as setup in
    // contrib/adc_pp/adc_guess_d.C adc_guess_d::build_guesses in order to preserve
    // S^2 value setup in the guess.
    //

    lt::orbit<4, scalar_type> orb1(sym, i1, false), orb2(sym, i2, false);
    const lt::index<4>& ci1 = orb1.get_cindex();  // Canonical block of (a, b, a, b)
    const lt::index<4>& ci2 = orb2.get_cindex();  // Canonical block of (a, b, b, a)
    bool zero1              = ctrl.req_is_zero_block(ci1);
    bool zero2              = ctrl.req_is_zero_block(ci2);
    if (zero1 && zero2) {
      // Set (alpha, alpha, alpha, alpha) to zero
      // This effectively filters out the quintet components with zero blocks
      // in (alpha, beta, alpha, beta) and (alpha, beta, beta, alpha) erroneously
      // introduced due to numerical errors.
      ctrl.req_zero_block(ci);
      continue;
    }

    // Get block corresponding to canonical index of (alpha, alpha, alpha, alpha)
    lt::dense_tensor_wr_i<4, scalar_type>& blk = ctrl.req_block(ci);

    if (!zero1) {  // (alpha, beta, alpha, beta) is not zero
      lt::dense_tensor_rd_i<4, scalar_type>& blk1 = ctrl.req_const_block(ci1);
      lt::tod_copy<4>(blk1, orb1.get_transf(i1)).perform(/* assign= */ true, blk);
      ctrl.ret_const_block(ci1);
    }
    if (!zero2) {  // (alpha, beta, beta, alpha) is not zero
      lt::dense_tensor_rd_i<4, scalar_type>& blk2 = ctrl.req_const_block(ci2);

      // Assign if (alpha, beta, alpha, beta) is zero, else +=
      lt::tod_copy<4>(blk2, orb2.get_transf(i2)).perform(zero1, blk);
      ctrl.ret_const_block(ci2);
    }
    ctrl.ret_block(ci);

  } while (ai.inc());
}

}  // namespace libadcc
