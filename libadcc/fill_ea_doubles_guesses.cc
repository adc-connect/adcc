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

#include "fill_ea_doubles_guesses.hh"
#include "TensorImpl.hh"
#include "guess/ea_adc_guess_d.hh"

namespace libadcc {

size_t fill_ea_doubles_guesses(std::vector<std::shared_ptr<Tensor>> guesses_d,
                               std::shared_ptr<const MoSpaces> mospaces,
                               std::shared_ptr<Tensor> d_o, 
                               std::shared_ptr<Tensor> d_v,
                               bool a_spin, bool restricted, bool doublet,
                               int spin_change_twice, 
                               scalar_type degeneracy_tolerance) {

  size_t n_guesses = guesses_d.size();
  if (n_guesses == 0) return 0;

  // Make a copy of the singles symmetry
  libtensor::block_tensor_ctrl<3, scalar_type> ctrl(asbt3(guesses_d[0]));
  libtensor::symmetry<3, scalar_type> sym_s(ctrl.req_const_symmetry().get_bis());
  libtensor::so_copy<3, scalar_type>(ctrl.req_const_symmetry()).perform(sym_s);

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
  for (size_t i = 0; i < 3; ++i) {
    abvectors.push_back(make_ab(*mospaces, spaces_d[i]));
  }
  libtensor::sequence<3, std::vector<bool>*> ab_d;
  for (size_t i = 0; i < 3; ++i) {
    ab_d[i] = &abvectors[i];
  }

  // Make singles list data structure
  std::list<std::pair<libtensor::btensor<3, double>*, double>> guesspairs;
  for (size_t i = 0; i < n_guesses; i++) {
    guesspairs.emplace_back(&(asbt3(guesses_d[i])), 0.0);
  }

  if (abs(spin_change_twice) != 1){
    throw not_implemented_error("spin_change ==" + 
      std::to_string(spin_change_twice) + " has not been tested.");
  }

  return ea_adc_guess_d(guesspairs, asbt1(d_o), asbt1(d_v), sym_s, a_spin, 
                        restricted, doublet, ab_d, spin_change_twice, 
                        degeneracy_tolerance);
}

}  // namespace libadcc