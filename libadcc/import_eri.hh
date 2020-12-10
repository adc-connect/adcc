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

#pragma once
#include "HartreeFockSolution_i.hh"
#include "MoIndexTranslation.hh"
#include "Tensor.hh"

namespace libadcc {

/** Import anti-symmetrised electron repulsion tensor directly from the
 * HartreeFockSolution_i object */
std::shared_ptr<Tensor> import_eri_asym_direct(const HartreeFockSolution_i& hf,
                                               const MoIndexTranslation& idxtrans,
                                               bool symmetry_check_on_import);

/** Import anti-symmetrised electron repulsion tensor by first importing the normal one
 * in chemist's indexing convention and then performing the antisymmetrisation
 */
std::shared_ptr<Tensor> import_eri_chem_then_asym_fast(const HartreeFockSolution_i& hf,
                                                       const MoIndexTranslation& idxtrans,
                                                       bool symmetry_check_on_import);

/** Import anti-symmetrised electron repulsion tensor by first importing the normal one
 * in chemist's indexing convention and then performing the antisymmetrisation and the
 * index reordering on the fly */
std::shared_ptr<Tensor> import_eri_chem_then_asym(const HartreeFockSolution_i& hf,
                                                  const MoIndexTranslation& idxtrans,
                                                  bool symmetry_check_on_import);

}  // namespace libadcc
