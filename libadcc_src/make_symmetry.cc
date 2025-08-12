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

#include "make_symmetry.hh"
#include "exceptions.hh"

namespace libadcc {
namespace {

/** Return the irreps of the passed cartesian transformation in the passed point group */
std::vector<std::string> irreps_of_cartesian_transformation(
      const std::string& point_group, const std::string& cartesian_transformation,
      const std::string& irrep_totsym) {

  // Handle the special cases first
  if (cartesian_transformation == "1") {
    return {irrep_totsym};  // Totally symmetric is always totally symmetric ^^
  }
  if (point_group == "C1") {
    return {irrep_totsym};  // C1 only has one irrep
  }

  using pg    = std::string;
  using trafo = std::string;
  using irrep = std::string;
  std::map<pg, std::map<irrep, std::vector<trafo>>> map{
        {
              "C2v",
              {
                    {"A1", {"z", "xx", "yy", "zz"}},  //
                    {"A2", {"xy", "Rz"}},             //
                    {"B1", {"x", "Ry", "xz"}},        //
                    {"B2", {"y", "Rx", "yz"}}         //
              },
        },
        {
              "D2",
              {
                    {"A", {"xx", "yy", "zz"}},  //
                    {"B1", {"z", "Rz", "xy"}},  //
                    {"B2", {"x", "Rx", "yz"}},  //
                    {"B3", {"y", "Ry", "xz"}}   //
              },

        }};

  const auto itpg = map.find(point_group);
  if (itpg == map.end()) {
    throw invalid_argument("Invalid point group or point group not implemented: " +
                           point_group);
  }

  // Go over all irreps in the point group and check if the cartesian_transformation
  // is part of it. If yes then add it to the returned list.
  std::vector<std::string> ret;
  for (const auto& irrep_trafos : itpg->second) {
    const std::string& irrep = irrep_trafos.first;
    for (const std::string& trafos : irrep_trafos.second) {
      if (trafos == cartesian_transformation) {
        ret.push_back(irrep);
      }
    }
  }

  if (ret.empty()) {
    throw invalid_argument("Could not find cartesian_transformation " +
                           cartesian_transformation + "in point group " + point_group +
                           ".");
  }
  return ret;
}

}  // namespace

std::shared_ptr<Symmetry> make_symmetry_orbital_energies(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space) {
  auto sym = std::make_shared<Symmetry>(mospaces_ptr, space);
  if (sym->ndim() != 1) {
    throw invalid_argument("Expect exactly a one-dimensional space string, not " + space +
                           ".");
  }

  // Setup spin symmetry
  if (mospaces_ptr->restricted) {
    sym->set_spin_block_maps({{"a", "b", 1.0}});
  }

  return sym;
}

std::shared_ptr<Symmetry> make_symmetry_orbital_coefficients(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
      size_t n_bas, const std::string& blocks) {
  if (space.back() != 'b') {
    throw invalid_argument(
          "Expect exactly a two-dimensional space string like 'o1b', not " + space + ".");
  }
  if (blocks != "ab" && blocks != "a" && blocks != "b" && blocks != "abstack") {
    throw invalid_argument(
          "Invalid argument to 'blocks' parameter. Only valid values are 'ab' (both "
          "alpha and beta coefficients in a block-diagonal fashion), "
          "'a' (only alpha coefficients), 'b' (only beta coefficients) and 'abstack'"
          "(alpha and beta coefficients stacked on top of another)");
  }

  // Setup extra "b" axis in a way that it twice n_bas in length
  // (alpha and beta coefficients)
  std::map<std::string, std::pair<size_t, size_t>> extra_axis{{"b", {n_bas, n_bas}}};

  if (blocks == "a" || blocks == "b" || blocks == "abstack") {
    // Cut the second spin block in the axis (i.e. only have either alpha
    // or beta spin along the "b" axis)
    extra_axis = {{"b", {n_bas, 0}}};
  }

  auto sym = std::make_shared<Symmetry>(mospaces_ptr, space, extra_axis);
  if (sym->ndim() != 2) {
    throw invalid_argument("Expect exactly a two-dimensional space string, not " + space +
                           ".");
  }

  // Set point-group symmetry:
  //    I (mfh) am not entirely sure why this always has to be the totally symmetric irrep
  //    I would imagine this to be the irrep of the groundstate
  //    ... but this is how it's done elsewhere
  sym->set_irreps_allowed({mospaces_ptr->irrep_totsym()});

  // Setup spin symmetry (spin block mapping)
  if (mospaces_ptr->restricted) {
    // Note: Libtensor assumes for the spin block mappings that there are
    //       an even number of blocks along the spin axis. We explicitly check
    //       for this in in as_lt_symmetry and ignore the spin symmetry if this
    //       is not the case, but it has to be properly tested if this does
    //       the trick. Notice that e.g. adcman does it like it is coded here
    //       and ignores all symmetry between the spin blocks in the case of
    //       an unrestricted reference. Technically speaking this is not
    //       completely necessary and thus the if(restricted) should be dropped.

    if (blocks == "ab") {
      // ab and ba are forbidden, eventually "aa" and "bb" mapped on top.
      sym->set_spin_blocks_forbidden({"ab", "ba"});
      if (mospaces_ptr->restricted) {
        sym->set_spin_block_maps({{"aa", "bb", 1.0}});
      }
    } else if (blocks == "a") {
      // Only alpha spin is allowed
      sym->set_spin_blocks_forbidden({"bx"});
    } else if (blocks == "b") {
      // Only beta spin is allowed
      sym->set_spin_blocks_forbidden({"ax"});
    } else if (blocks == "abstack") {
      if (mospaces_ptr->restricted) {
        sym->set_spin_block_maps({{"ax", "bx", 1.0}});
      }
    }
  }  // spin symmetry
  return sym;
}

std::shared_ptr<Symmetry> make_symmetry_eri(std::shared_ptr<const MoSpaces> mospaces_ptr,
                                            const std::string& space) {
  auto sym = std::make_shared<Symmetry>(mospaces_ptr, space);

  const std::vector<std::string>& ss = sym->subspaces();
  const MoSpaces& mo                 = *mospaces_ptr;
  if (sym->ndim() != 4) {
    throw invalid_argument("Expect exactly a four-dimensional space string, not " +
                           space + ".");
  }

  // adcc uses anti-symmetrised repulsion integrals in the physicist's
  // convention, i.e. the integral <ij || kl> = < ij | kl > - < ij | lk>
  // is anti-symmetric in the first two indices and the last two
  // indices but symmetric if the first two are swapped with the last two.
  //
  // These symmetries of course only apply if we deal with the same MO subspace
  // blocks in the respective indices.
  std::vector<std::string> permutations{"ijkl"};
  if (ss[0] == ss[1]) permutations.push_back("-jikl");
  if (ss[2] == ss[3]) permutations.push_back("-ijlk");
  if (ss[0] == ss[2] && ss[1] == ss[3]) permutations.push_back("klij");
  if (permutations.size() > 1) {
    sym->set_permutations(permutations);
  }

  // Set point-group symmetry: Repulsion integrals are totally symmetric
  sym->set_irreps_allowed({mo.irrep_totsym()});

  // Set spin symmetry:
  if (mospaces_ptr->restricted) {
    // Note: Libtensor assumes for the spin block mappings that there are
    //       an even number of blocks along the spin axis. We explicitly check
    //       for this in in as_lt_symmetry and ignore the spin symmetry if this
    //       is not the case, but it has to be properly tested if this does
    //       the trick. Notice that e.g. adcman does it like it is coded here
    //       and ignores all symmetry between the spin blocks in the case of
    //       an unrestricted reference. Technically speaking this is not
    //       completely necessary and as for example the next statement of
    //       forbidden blocks should actually be shifted one up out of the if.
    sym->set_spin_blocks_forbidden({"aaab", "aaba", "aabb", "abaa", "abbb",  //
                                    "bbba", "bbab", "bbaa", "babb", "baaa"});
    sym->set_spin_block_maps({
          {"aaaa", "bbbb", 1.},
          {"abab", "baba", 1.},
          {"abba", "baab", 1.},
    });
  }
  return sym;
}

std::shared_ptr<Symmetry> make_symmetry_eri_symm(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space) {
  auto sym = std::make_shared<Symmetry>(mospaces_ptr, space);

  const std::vector<std::string>& ss = sym->subspaces();
  const MoSpaces& mo                 = *mospaces_ptr;
  if (sym->ndim() != 4) {
    throw invalid_argument("Expect exactly a four-dimensional space string, not " +
                           space + ".");
  }

  // We setup the symmetry of the ERI <ij | kl> in Physicists' indexing convention.
  // This one is symmetric in the 1st and 3rd, under exchange of shell pairs
  // and if first two and last two are swapped.
  // These symmetries only apply if they connect identical MO subspaces.
  std::vector<std::string> permutations{"ijkl"};
  if (ss[0] == ss[2]) permutations.push_back("kjil");
  if (ss[0] == ss[1] && ss[2] == ss[3]) permutations.push_back("jilk");
  if (ss[0] == ss[2] && ss[1] == ss[3]) permutations.push_back("klij");
  if (permutations.size() > 1) {
    sym->set_permutations(permutations);
  }

  // Set point-group symmetry: Repulsion integrals are totally symmetric
  sym->set_irreps_allowed({mo.irrep_totsym()});

  // Set spin symmetry:
  if (mospaces_ptr->restricted) {
    // Note: Libtensor assumes for the spin block mappings that there are
    //       an even number of blocks along the spin axis. We explicitly check
    //       for this in in as_lt_symmetry and ignore the spin symmetry if this
    //       is not the case, but it has to be properly tested if this does
    //       the trick. Notice that e.g. adcman does it like it is coded here
    //       and ignores all symmetry between the spin blocks in the case of
    //       an unrestricted reference. Technically speaking this is not
    //       completely necessary and as for example the next statement of
    //       forbidden blocks should actually be shifted one up out of the if.
    sym->set_spin_blocks_forbidden({"aaab", "aaba", "aabb", "abba", "abaa", "abbb",  //
                                    "bbba", "bbab", "bbaa", "baab", "babb", "baaa"});
    sym->set_spin_block_maps({
          {"aaaa", "bbbb", 1.},
          {"abab", "baba", 1.},
          {"abab", "aaaa", 1.},  // That's only true for restricted!
    });
  }
  return sym;
}

std::shared_ptr<Symmetry> make_symmetry_operator(
      std::shared_ptr<const MoSpaces> mospaces_ptr, const std::string& space,
      bool symmetric, const std::string& cartesian_transformation) {
  auto sym = std::make_shared<Symmetry>(mospaces_ptr, space);

  const std::vector<std::string>& ss = sym->subspaces();
  if (sym->ndim() != 2) {
    throw invalid_argument("Expect exactly a two-dimensional space string, not " + space +
                           ".");
  }

  // Operator is a symmetric matrix iff symmetric and both spaces equal
  if (symmetric && ss[0] == ss[1]) {
    sym->set_permutations({"ij", "ji"});
  }

  // Point-group symmetry
  const std::vector<std::string> irreps = irreps_of_cartesian_transformation(
        mospaces_ptr->point_group, cartesian_transformation,
        mospaces_ptr->irrep_totsym());
  sym->set_irreps_allowed(irreps);

  // A one-electron (spin-free) Operator is aa / bb block-diagonal only.
  if (mospaces_ptr->restricted) {
    // Note: Libtensor assumes for the spin block mappings that there are
    //       an even number of blocks along the spin axis. We explicitly check
    //       for this in in as_lt_symmetry and ignore the spin symmetry if this
    //       is not the case, but it has to be properly tested if this does
    //       the trick. Notice that e.g. adcman does it like it is coded here
    //       and ignores all symmetry between the spin blocks in the case of
    //       an unrestricted reference. Technically speaking this is not
    //       completely necessary and as for example the next statement of
    //       forbidden blocks should actually be shifted one up out of the if.
    sym->set_spin_blocks_forbidden({"ab", "ba"});
    sym->set_spin_block_maps({{"aa", "bb", 1.}});
  }
  return sym;
}

std::shared_ptr<Symmetry> make_symmetry_operator_basis(
      std::shared_ptr<const MoSpaces> mospaces_ptr, size_t n_bas, bool symmetric) {

  // Setup symmetry in a way that the "b" axis has "b" alphas and "b" betas
  using map_type = std::map<std::string, std::pair<size_t, size_t>>;
  auto sym =
        std::make_shared<Symmetry>(mospaces_ptr, "bb", map_type{{"b", {n_bas, n_bas}}});
  if (symmetric) sym->set_permutations({"ij", "ji"});

  // A one-electron (spin-free) Operator is aa / bb block-diagonal only
  // and the basis functions are spin-free, thus aa == bb.
  sym->set_spin_blocks_forbidden({"ab", "ba"});
  sym->set_spin_block_maps({{"aa", "bb", 1.}});
  return sym;
}

}  // namespace libadcc
