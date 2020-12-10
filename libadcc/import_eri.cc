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

#include "import_eri.hh"
#include "make_symmetry.hh"

namespace libadcc {
namespace {
/** Import (non-antisymmetrised) electron repulsion tensor in the physicists' indexing
 *  convention by importing from the HartreeFockSolution_i ERI interface
 *  (which is in chemist's indexing convention) using appropriately permuted strides. */
std::shared_ptr<Tensor> import_eri_symm_from_chem(const HartreeFockSolution_i& hf,
                                                  const MoIndexTranslation& idxtrans,
                                                  bool symmetry_check_on_import) {
  std::shared_ptr<Symmetry> sym_ptr =
        make_symmetry_eri_symm(idxtrans.mospaces_ptr(), idxtrans.space());
  std::shared_ptr<Tensor> eri = make_tensor_zero(sym_ptr);

  auto gen_translate_chem = [&idxtrans, &hf](
                                  const std::vector<std::pair<size_t, size_t>>& block,
                                  scalar_type* buffer) {
    // Strides, which should be used to write data to buffer
    // and size of the buffer.
    const std::vector<size_t> blength{
          block[0].second - block[0].first, block[1].second - block[1].first,
          block[2].second - block[2].first, block[3].second - block[3].first};
    const std::vector<size_t> strides{blength[1] * blength[2] * blength[3],
                                      blength[2] * blength[3], blength[3], 1};
    const size_t buffer_size = strides[0] * blength[0];

    std::vector<RangeMapping> mappings = idxtrans.map_range_to_hf_provider(block);
    for (RangeMapping& map : mappings) {
      const SimpleRange& bidx  = map.from();  // Indices of the MO subspace block
      const SimpleRange& hfidx = map.to();    // Indices translated to HF provider

      // Compute offset and remaining buffer size
      size_t offset = 0;
      for (size_t i = 0; i < 4; ++i) {
        offset += (bidx.axis(i).start() - block[i].first) * strides[i];
      }
      const size_t size = buffer_size - offset;

      // eri_ffff returns the (non-antisymmetrised) ERIs in Chemists' notation,
      // so the integral <01|23> = (02|13) needs to be imported
      hf.eri_ffff(hfidx.axis(0).start(), hfidx.axis(0).end(),      //
                  hfidx.axis(2).start(), hfidx.axis(2).end(),      //
                  hfidx.axis(1).start(), hfidx.axis(1).end(),      //
                  hfidx.axis(3).start(), hfidx.axis(3).end(),      //
                  strides[0], strides[2], strides[1], strides[3],  //
                  buffer + offset, size);
    }
  };
  eri->import_from(gen_translate_chem, hf.conv_tol(), symmetry_check_on_import);
  hf.flush_cache();
  return eri;
}
}  // namespace

/** Import anti-symmetrised electron repulsion tensor directly from the
 * HartreeFockSolution_i object */
std::shared_ptr<Tensor> import_eri_asym_direct(const HartreeFockSolution_i& hf,
                                               const MoIndexTranslation& idxtrans,
                                               bool symmetry_check_on_import) {
  std::shared_ptr<Symmetry> sym_ptr =
        make_symmetry_eri(idxtrans.mospaces_ptr(), idxtrans.space());
  std::shared_ptr<Tensor> eri = make_tensor_zero(sym_ptr);

  auto eri_generator = [&idxtrans, &hf](
                             const std::vector<std::pair<size_t, size_t>>& block,
                             scalar_type* buffer) {
    // Strides, which should be used to write data to buffer
    // and size of the buffer.
    const std::vector<size_t> blength{
          block[0].second - block[0].first, block[1].second - block[1].first,
          block[2].second - block[2].first, block[3].second - block[3].first};
    const std::vector<size_t> strides{blength[1] * blength[2] * blength[3],
                                      blength[2] * blength[3], blength[3], 1};
    const size_t buffer_size = strides[0] * blength[0];

    std::vector<RangeMapping> mappings = idxtrans.map_range_to_hf_provider(block);
    for (RangeMapping& map : mappings) {
      const SimpleRange& bidx  = map.from();  // Indices of the MO subspace block
      const SimpleRange& hfidx = map.to();    // Indices translated to HF provider

      // Compute offset and remaining buffer size
      size_t offset = 0;
      for (size_t i = 0; i < 4; ++i) {
        offset += (bidx.axis(i).start() - block[i].first) * strides[i];
      }
      const size_t size = buffer_size - offset;
      hf.eri_phys_asym_ffff(hfidx.axis(0).start(), hfidx.axis(0).end(),      //
                            hfidx.axis(1).start(), hfidx.axis(1).end(),      //
                            hfidx.axis(2).start(), hfidx.axis(2).end(),      //
                            hfidx.axis(3).start(), hfidx.axis(3).end(),      //
                            strides[0], strides[1], strides[2], strides[3],  //
                            buffer + offset, size);
    }
  };
  eri->import_from(eri_generator, hf.conv_tol(), symmetry_check_on_import);
  hf.flush_cache();
  return eri;
}

/** Import anti-symmetrised electron repulsion tensor by first importing the normal one
 * in chemist's indexing convention and then performing the antisymmetrisation
 */
std::shared_ptr<Tensor> import_eri_chem_then_asym_fast(const HartreeFockSolution_i& hf,
                                                       const MoIndexTranslation& idxtrans,
                                                       bool symmetry_check_on_import) {
  // First import the (non-antisymmetrised) ERI in physicists' indexing convention.
  // This is done from the chemists' indexing convention by using strides.
  std::shared_ptr<Tensor> eri_symm =
        import_eri_symm_from_chem(hf, idxtrans, symmetry_check_on_import);

  // Antisymmetrise permuting either 0th and 1st or 2nd and 3rd index
  // Multiply by 2, because the antisymmetrisation below includes a factor (1/2)
  const std::vector<std::string>& ss = idxtrans.subspaces();
  if (ss[0] == ss[1]) {
    return evaluate(eri_symm->scale(2.0)->antisymmetrise({{0, 1}}));
  } else if (ss[2] == ss[3]) {
    return evaluate(eri_symm->scale(2.0)->antisymmetrise({{2, 3}}));
  } else {
    throw invalid_argument(
          "import_eri_chem_then_asym_fast only supported if subspaces[0] == subspaces[1] "
          "or subspaces[2] == subspaces[3].");
  }
  return eri_symm;
}

/** Import anti-symmetrised electron repulsion tensor by first importing the normal one
 * in chemist's indexing convention and then performing the antisymmetrisation and the
 * index reordering on the fly */
std::shared_ptr<Tensor> import_eri_chem_then_asym(const HartreeFockSolution_i& hf,
                                                  const MoIndexTranslation& idxtrans,
                                                  bool symmetry_check_on_import) {
  std::shared_ptr<Symmetry> sym_ptr =
        make_symmetry_eri(idxtrans.mospaces_ptr(), idxtrans.space());
  std::shared_ptr<Tensor> eri = make_tensor_zero(sym_ptr);

  // Dummy generator to import translating from chem -> phys and antisymmetrising
  // at the same time.
  auto gen_translate_chem_asym =
        [&idxtrans, &hf](const std::vector<std::pair<size_t, size_t>>& block,
                         scalar_type* buffer) {
          std::vector<RangeMapping> mappings = idxtrans.map_range_to_hf_provider(block);

          // Strides, which should be used to write data to buffer in physicists' order
          const std::vector<size_t> blength{
                block[0].second - block[0].first, block[1].second - block[1].first,
                block[2].second - block[2].first, block[3].second - block[3].first};
          const std::vector<size_t> strides{blength[1] * blength[2] * blength[3],
                                            blength[2] * blength[3], blength[3], 1};
          const size_t buffer_size = strides[0] * blength[0];

          for (RangeMapping& map : mappings) {
            const SimpleRange& bidx  = map.from();  // Indices of the MO subspace block
            const SimpleRange& hfidx = map.to();    // Indices translated to HF provider

            // Compute offset and remaining buffer size
            size_t offset = 0;
            for (size_t i = 0; i < 4; ++i) {
              offset += (bidx.axis(i).start() - block[i].first) * strides[i];
            }
            const size_t size = buffer_size - offset;

            // In which spinblock are we?
            std::string spins = idxtrans.spin_of(bidx.starts());
            if (idxtrans.spin_of(bidx.lasts()) != spins) {
              throw runtime_error("Internal error: RangeMapping is over spin blocks.");
            }

            // We want to compute <01||23> = <01|23> - <10|23> or in chemist's notation
            //                             = (02|13) - (12|03)
            // Sometimes (02|13) is zero by spin-symmetry and sometimes (12|03) is zero
            // by spin-symmetry and thus do not need to be computed. This is governed
            // here.
            const bool need_0213 = ((spins[0] == spins[2] && spins[1] == spins[3]));
            const bool need_1203 = ((spins[1] == spins[2] && spins[0] == spins[3]));

            if (need_0213) {
              // Into buffer we write the first ERI block, keeping in mind that we have
              // to permute the strides and axes in order to translate from chemist's to
              // physicist's notation.
              hf.eri_ffff(hfidx.axis(0).start(), hfidx.axis(0).end(),      // axis 0
                          hfidx.axis(2).start(), hfidx.axis(2).end(),      // axis 2
                          hfidx.axis(1).start(), hfidx.axis(1).end(),      // axis 1
                          hfidx.axis(3).start(), hfidx.axis(3).end(),      // axis 3
                          strides[0], strides[2], strides[1], strides[3],  //
                          buffer + offset, size);
            } else {
              // Fill buffer with zeros at the locations the above hf.eri_ffff
              // call would set.
              for (size_t i0 = 0; i0 < bidx.axis(0).length(); ++i0) {
                for (size_t i1 = 0; i1 < bidx.axis(1).length(); ++i1) {
                  for (size_t i2 = 0; i2 < bidx.axis(2).length(); ++i2) {
                    for (size_t i3 = 0; i3 < bidx.axis(3).length(); ++i3) {
                      const size_t idx = strides[0] * i0 +  //
                                         strides[1] * i1 +  //
                                         strides[2] * i2 +  //
                                         strides[3] * i3;
                      buffer[offset + idx] = 0.0;
                    }  // i3
                  }    // i2
                }      // i1
              }        // i0
            }

            if (need_1203) {
              // Compute the strides and size of the other which is needed to import
              // the second ERI block.
              const std::vector<size_t> o_strides{
                    bidx.axis(1).length() * bidx.axis(2).length() * bidx.axis(3).length(),
                    bidx.axis(2).length() * bidx.axis(3).length(), bidx.axis(3).length(),
                    1};
              std::vector<scalar_type> other(o_strides[0] * bidx.axis(0).length());

              // Into the "other" buffer, we import the second ERI block
              hf.eri_ffff(hfidx.axis(1).start(), hfidx.axis(1).end(),  // axis 1
                          hfidx.axis(2).start(), hfidx.axis(2).end(),  // axis 2
                          hfidx.axis(0).start(), hfidx.axis(0).end(),  // axis 0
                          hfidx.axis(3).start(), hfidx.axis(3).end(),  // axis 3
                          o_strides[1], o_strides[2], o_strides[0], o_strides[3],  //
                          other.data(), other.size());

              // In-place form the right result in the buffer:
              for (size_t i0 = 0; i0 < bidx.axis(0).length(); ++i0) {
                for (size_t i1 = 0; i1 < bidx.axis(1).length(); ++i1) {
                  for (size_t i2 = 0; i2 < bidx.axis(2).length(); ++i2) {
                    for (size_t i3 = 0; i3 < bidx.axis(3).length(); ++i3) {
                      const size_t idx = strides[0] * i0 +  //
                                         strides[1] * i1 +  //
                                         strides[2] * i2 +  //
                                         strides[3] * i3;
                      const size_t o_idx = o_strides[0] * i0 +  //
                                           o_strides[1] * i1 +  //
                                           o_strides[2] * i2 +  //
                                           o_strides[3] * i3;
                      buffer[offset + idx] = buffer[offset + idx] - other[o_idx];
                    }  // i3
                  }    // i2
                }      // i1
              }        // i0

            }  // need_1203
          }    // map : mappings
        };
  eri->import_from(gen_translate_chem_asym, hf.conv_tol(), symmetry_check_on_import);
  hf.flush_cache();
  return eri;
}

}  // namespace libadcc
