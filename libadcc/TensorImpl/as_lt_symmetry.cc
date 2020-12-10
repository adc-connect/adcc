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

#include "as_lt_symmetry.hh"
#include "../exceptions.hh"
#include "as_bispace.hh"

// Change visibility of libtensor singletons to public
#pragma GCC visibility push(default)
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/scalar_transf.h>
#include <libtensor/symmetry/se_label.h>
#include <libtensor/symmetry/se_part.h>
#include <libtensor/symmetry/se_perm.h>
#pragma GCC visibility pop

namespace libadcc {
namespace lt = libtensor;

namespace {
template <size_t N>
void transfer_permuational_symmetry(
      const Symmetry& sym, std::shared_ptr<lt::symmetry<N, scalar_type>> libtensym_ptr) {
  auto perms = sym.permutations_parsed();

  // Construct a reference sequence
  lt::sequence<N, size_t> seq_ref;
  for (size_t i = 0; i < N; ++i) seq_ref[i] = i;

  for (const auto& perm_fac : perms) {
    // Construct libtensor sequence
    lt::sequence<N, size_t> seq_idcs;
    for (size_t i = 0; i < N; ++i) seq_idcs[i] = perm_fac.first[i];

    lt::permutation<N> perm = lt::permutation_builder<N>(seq_ref, seq_idcs).get_perm();
    lt::scalar_transf<scalar_type> transf(perm_fac.second);
    libtensym_ptr->insert(lt::se_perm<N, scalar_type>(perm, transf));
  }
}

template <size_t N>
void transfer_pointgroup_symmetry(
      const Symmetry& sym, std::shared_ptr<lt::symmetry<N, scalar_type>> libtensym_ptr) {
  const MoSpaces& mospaces                  = *sym.mospaces_ptr();
  const std::vector<std::string>& subspaces = sym.subspaces();
  const std::vector<AxisInfo>& axes         = sym.axes();

  // Build block_index_dims
  lt::index<N> start;
  lt::index<N> end;
  for (size_t i = 0; i < N; ++i) {
    // The -1 is because libtensor ranges are inclusive on both ends.
    end[i] = axes[i].block_starts.size() - 1;
  }
  lt::dimensions<N> block_index_dims(lt::index_range<N>(start, end));
  lt::se_label<N, scalar_type> slabel(block_index_dims, mospaces.point_group);

  std::vector<bool> dimdone(N, false);
  for (size_t i = 0; i < N; ++i) {
    if (dimdone[i]) continue;
    lt::mask<N> m;
    for (size_t j = 0; j < N; ++j) m[j] = subspaces[i] == subspaces[j];

    if (mospaces.map_block_irrep.find(subspaces[i]) == mospaces.map_block_irrep.end()) {
      // Skip the rest for extra axes, which do not have PG symmetry
      continue;
    }

    for (size_t ib = 0; ib < block_index_dims[i]; ++ib) {
      const std::string irrep = mospaces.map_block_irrep.at(subspaces[i])[ib];
      slabel.get_labeling().assign(m, ib, mospaces.libtensor_irrep_index(irrep));
    }
    for (size_t j = 0; j < N; ++j) dimdone[j] = dimdone[j] | m[j];
  }

  std::set<size_t> tgts;
  for (auto irrep : sym.irreps_allowed()) {
    tgts.insert(mospaces.libtensor_irrep_index(irrep));
  }
  slabel.set_rule(tgts);
  libtensym_ptr->insert(slabel);
}

template <size_t N>
void transfer_spin_symmetry(const Symmetry& sym,
                            std::shared_ptr<lt::symmetry<N, scalar_type>> libtensym_ptr) {
  const std::vector<AxisInfo>& axes = sym.axes();

  // Build partitioning dimensions (keeping in mind that some axes
  // might have no spin symmetry) and setup se_part object
  lt::index<N> start;
  lt::index<N> end;
  for (size_t idim = 0; idim < N; ++idim) {
    end[idim] = 0;  // Default to not having spin on this axis

    if (axes[idim].has_spin()) {
      size_t n_alpha_blocks = 0;
      size_t n_beta_blocks  = 0;

      const std::vector<size_t>& block_starts = axes[idim].block_starts;
      for (size_t istart = 0; istart < axes[idim].block_starts.size(); ++istart) {
        if (block_starts[istart] < axes[idim].n_orbs_alpha) {
          n_alpha_blocks += 1;
        } else {
          n_beta_blocks += 1;
        }
      }
      if (n_alpha_blocks + n_beta_blocks != block_starts.size()) {
        throw runtime_error("Internal error when distributing blocks to alpha and beta.");
      }

      // Libtensor can only properly partition an axis if the number
      // of alpha blocks and the number of beta blocks is identical.
      // If that's not the case, we ignore the symmetry request.
      if (n_alpha_blocks != n_beta_blocks) continue;

      end[idim] = 1;  // We allow for spin on this axis.
    }
  }

  // If no axis ends up being partitioned there is no point in inserting
  // a symmetry object at all.
  if (start == end) return;

  lt::dimensions<N> partdims(lt::index_range<N>(start, end));
  lt::se_part<N, scalar_type> spart(libtensym_ptr->get_bis(), partdims);

  auto parse_spinblock = [](const std::string& block) {
    // Build block-index from aabb string or so,
    // where a maps to 0 and b to 1.
    lt::index<N> ret;
    for (size_t i = 0; i < N; ++i) {
      if (block[i] == 'b') {
        ret[i] = 1;  // Beta block
      } else if (block[i] == 'a' || block[i] == 'x') {
        ret[i] = 0;  // Alpha block or unspecified
      } else {
        throw runtime_error("Internal error: Invalid spin block specification");
      }
    }
    return ret;
  };

  for (auto from_to_fac : sym.spin_block_maps()) {
    const lt::index<N> block1 = parse_spinblock(std::get<0>(from_to_fac));
    const lt::index<N> block2 = parse_spinblock(std::get<1>(from_to_fac));
    lt::scalar_transf<scalar_type> transf(std::get<2>(from_to_fac));
    spart.add_map(block1, block2, transf);
  }

  for (auto block : sym.spin_blocks_forbidden()) {
    spart.mark_forbidden(parse_spinblock(block));
  }

  libtensym_ptr->insert(spart);
}
}  // namespace

template <size_t N>
std::shared_ptr<lt::symmetry<N, scalar_type>> as_lt_symmetry(const Symmetry& sym) {
  if (sym.ndim() != N) {
    throw invalid_argument("N (== " + std::to_string(N) +
                           ") and dimensionality of Symmetry object (== " +
                           std::to_string(sym.ndim()) + " does not agree.");
  }

  // Build libtensor block-index-space
  lt::block_index_space<N> block_index_space = as_bispace<N>(sym.axes()).get_bis();
  auto ret_ptr = std::make_shared<lt::symmetry<N, scalar_type>>(block_index_space);

  if (sym.has_permutations()) {
    transfer_permuational_symmetry(sym, ret_ptr);
  }
  if (sym.has_irreps_allowed()) {
    transfer_pointgroup_symmetry(sym, ret_ptr);
  }
  if (sym.has_spin_block_maps() || sym.has_spin_blocks_forbidden()) {
    transfer_spin_symmetry(sym, ret_ptr);
  }

  return ret_ptr;
}

//
// Explicit instantiation
//

#define INSTANTIATE(DIM)                                                          \
  template std::shared_ptr<libtensor::symmetry<DIM, scalar_type>> as_lt_symmetry( \
        const Symmetry& sym);

INSTANTIATE(1)
INSTANTIATE(2)
INSTANTIATE(3)
INSTANTIATE(4)

#undef INSTANTIATE

}  // namespace libadcc
