//
// Copyright (C) 2020 by the adcc authors
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

#include "fill_pp_doubles_guesses.hh"
#include "TensorImpl.hh"
#include "guess/adc_guess_d.hh"

namespace libadcc {

size_t fill_pp_doubles_guesses(std::vector<std::shared_ptr<Tensor>> guesses_d,
                               std::shared_ptr<const MoSpaces> mospaces,
                               std::shared_ptr<Tensor> df1, std::shared_ptr<Tensor> df2,
                               int spin_change_twice, scalar_type degeneracy_tolerance) {

  size_t n_guesses = guesses_d.size();
  if (n_guesses == 0) return 0;

  // Make a copy of the singles symmetry
  libtensor::block_tensor_ctrl<4, scalar_type> ctrl(asbt4(guesses_d[0]));
  libtensor::symmetry<4, scalar_type> sym_s(ctrl.req_const_symmetry().get_bis());
  libtensor::so_copy<4, scalar_type>(ctrl.req_const_symmetry()).perform(sym_s);

  // Make ab pointers object
  auto make_ab = [](const MoSpaces& mo, const std::string& space) {
    const std::vector<char>& block_spin = mo.map_block_spin.at(space);
    std::vector<bool> ab;
    for (size_t i = 0; i < block_spin.size(); ++i) {
      ab.push_back(block_spin[i] == 'b');
    }
    return ab;
  };

  const std::vector<std::string> spaces_d = guesses_d[0]->subspaces();
  std::vector<std::vector<bool>> abvectors;
  for (size_t i = 0; i < 4; ++i) {
    abvectors.push_back(make_ab(*mospaces, spaces_d[i]));
  }
  libtensor::sequence<4, std::vector<bool>*> ab_d;
  for (size_t i = 0; i < 4; ++i) {
    ab_d[i] = &abvectors[i];
  }

  // Make singles list data structure
  std::list<std::pair<libtensor::btensor<4, double>*, double>> guesspairs;
  for (size_t i = 0; i < n_guesses; i++) {
    guesspairs.emplace_back(&(asbt4(guesses_d[i])), 0.0);
  }

  if (spin_change_twice != -2 && spin_change_twice != 0) {
    throw not_implemented_error("spin_change == -1 has not been tested.");
  }

  return adc_guess_d(guesspairs, asbt2(df1), asbt2(df2), sym_s, ab_d, spin_change_twice,
                     degeneracy_tolerance);
}

}  // namespace libadcc
