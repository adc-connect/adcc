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

#include "../MoIndexTranslation.hh"
#include "HFSolutionMock.hh"
#include <catch2/catch.hpp>

namespace libadcc {
namespace tests {

TEST_CASE("Test MoIndexTranslation", "[MoIndexTranslation]") {
  using vecsize = std::vector<size_t>;

  HFSolutionMock hf_h2o;
  auto adcmem_ptr = std::make_shared<AdcMemory>();

  //
  // Block size 16
  //
  SECTION("Block size 16") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    std::shared_ptr<MoSpaces> mospaces_ptr(new MoSpaces(tested, adcmem_ptr, {}, {}, {}));

    SECTION("Space o1o1") {
      MoIndexTranslation motrans(mospaces_ptr, "o1o1");
      CHECK(motrans.space() == "o1o1");
      CHECK(motrans.subspaces() == std::vector<std::string>{"o1", "o1"});
      CHECK(motrans.ndim() == 2);
      CHECK(motrans.shape() == vecsize{10, 10});

      CHECK_THROWS(motrans.full_index_of({1, 2}) == vecsize{1, 2});
      CHECK_THROWS(motrans.full_index_of({1, 5}) == vecsize{1, 7});
      CHECK_THROWS(motrans.full_index_of({7, 5}) == vecsize{9, 7});

      CHECK(motrans.block_index_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_of({1, 5}) == vecsize{0, 1});
      CHECK(motrans.block_index_of({7, 5}) == vecsize{1, 1});

      CHECK(motrans.block_index_spatial_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({1, 5}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({7, 5}) == vecsize{0, 0});

      CHECK(motrans.inblock_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.inblock_index_of({1, 5}) == vecsize{1, 0});
      CHECK(motrans.inblock_index_of({7, 5}) == vecsize{2, 0});

      CHECK(motrans.spin_of({1, 2}) == "aa");
      CHECK(motrans.spin_of({1, 5}) == "ab");
      CHECK(motrans.spin_of({7, 5}) == "bb");

      std::vector<vecsize> cases{{1, 2}, {1, 5}, {7, 5}};
      for (auto idx : cases) {
        const auto result = motrans.split(idx);
        CHECK(result.first == motrans.block_index_of(idx));
        CHECK(result.second == motrans.inblock_index_of(idx));

        CHECK(motrans.combine(result.first, result.second) == idx);
        CHECK(motrans.combine(motrans.spin_of(idx), motrans.block_index_spatial_of(idx),
                              result.second) == idx);
      }

      CHECK(motrans.hf_provider_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.hf_provider_index_of({1, 5}) == vecsize{1, 7});
      CHECK(motrans.hf_provider_index_of({7, 5}) == vecsize{9, 7});

      // Map a single block
      std::vector<RangeMapping> ref_1{{
            {{0, 4}, {0, 3}},  // from
            {{0, 4}, {0, 3}}   // to
      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 4}, {0, 3}}) == ref_1);

      // Map over a block split
      std::vector<RangeMapping> ref_2{{
                                            {{0, 5}, {0, 3}},  // from
                                            {{0, 5}, {0, 3}}   // to
                                      },
                                      {
                                            {{5, 6}, {0, 3}},  // from
                                            {{7, 8}, {0, 3}}   // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 6}, {0, 3}}) == ref_2);

      // Map over two block splits
      std::vector<RangeMapping> ref_3{{
                                            {{3, 5}, {4, 5}},  // from
                                            {{3, 5}, {4, 5}}   // to
                                      },
                                      {
                                            {{3, 5}, {5, 6}},  // from
                                            {{3, 5}, {7, 8}}   // to
                                      },
                                      {
                                            {{5, 6}, {4, 5}},  // from
                                            {{7, 8}, {4, 5}}   // to
                                      },
                                      {
                                            {{5, 6}, {5, 6}},  // from
                                            {{7, 8}, {7, 8}}   // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{3, 6}, {4, 6}}) == ref_3);
    }  // o1o1

    SECTION("Space o1v1o1v1") {
      MoIndexTranslation motrans(mospaces_ptr, "o1v1o1v1");

      CHECK(motrans.space() == "o1v1o1v1");
      CHECK(motrans.subspaces() == std::vector<std::string>{"o1", "v1", "o1", "v1"});
      CHECK(motrans.ndim() == 4);
      CHECK(motrans.shape() == vecsize{10, 4, 10, 4});

      CHECK_THROWS(motrans.full_index_of({1, 1, 2, 1}) == vecsize{1, 6, 2, 6});
      CHECK_THROWS(motrans.full_index_of({1, 1, 5, 3}) == vecsize{1, 5, 7, 13});
      CHECK_THROWS(motrans.full_index_of({7, 3, 5, 3}) == vecsize{9, 13, 7, 13});

      CHECK(motrans.block_index_of({1, 1, 2, 1}) == vecsize{0, 0, 0, 0});
      CHECK(motrans.block_index_of({1, 1, 5, 3}) == vecsize{0, 0, 1, 1});
      CHECK(motrans.block_index_of({7, 3, 5, 3}) == vecsize{1, 1, 1, 1});

      CHECK(motrans.block_index_spatial_of({1, 1, 2, 1}) == vecsize{0, 0, 0, 0});
      CHECK(motrans.block_index_spatial_of({1, 1, 5, 3}) == vecsize{0, 0, 0, 0});
      CHECK(motrans.block_index_spatial_of({7, 3, 5, 3}) == vecsize{0, 0, 0, 0});

      CHECK(motrans.inblock_index_of({1, 1, 2, 1}) == vecsize{1, 1, 2, 1});
      CHECK(motrans.inblock_index_of({1, 1, 5, 3}) == vecsize{1, 1, 0, 1});
      CHECK(motrans.inblock_index_of({7, 3, 5, 3}) == vecsize{2, 1, 0, 1});

      CHECK(motrans.spin_of({1, 1, 2, 1}) == "aaaa");
      CHECK(motrans.spin_of({1, 1, 5, 3}) == "aabb");
      CHECK(motrans.spin_of({7, 3, 5, 3}) == "bbbb");

      std::vector<vecsize> cases{{1, 1, 2, 1}, {1, 1, 5, 3}, {7, 3, 5, 3}};
      for (auto idx : cases) {
        const auto result = motrans.split(idx);
        CHECK(result.first == motrans.block_index_of(idx));
        CHECK(result.second == motrans.inblock_index_of(idx));

        CHECK(motrans.combine(result.first, result.second) == idx);
        CHECK(motrans.combine(motrans.spin_of(idx), motrans.block_index_spatial_of(idx),
                              result.second) == idx);
      }

      CHECK(motrans.hf_provider_index_of({1, 1, 2, 1}) == vecsize{1, 6, 2, 6});
      CHECK(motrans.hf_provider_index_of({1, 1, 5, 3}) == vecsize{1, 6, 7, 13});
      CHECK(motrans.hf_provider_index_of({7, 3, 5, 3}) == vecsize{9, 13, 7, 13});

      // Map a single block
      std::vector<RangeMapping> ref_1{{
            {{0, 4}, {0, 2}, {0, 3}, {0, 1}},  // from
            {{0, 4}, {5, 7}, {0, 3}, {5, 6}}   // to
      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 4}, {0, 2}, {0, 3}, {0, 1}}) == ref_1);

      // Map over two block split
      std::vector<RangeMapping> ref_2{{
                                            {{0, 5}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{0, 5}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{0, 5}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{0, 5}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{7, 8}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{7, 8}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 6}, {0, 2}, {3, 5}, {1, 4}}) == ref_2);

      // Map over three block splits
      std::vector<RangeMapping> ref_3{{
                                            {{0, 5}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{0, 5}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{0, 5}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{0, 5}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      },
                                      {
                                            {{0, 5}, {2, 3}, {3, 5}, {1, 2}},   // from
                                            {{0, 5}, {12, 13}, {3, 5}, {6, 7}}  // to
                                      },
                                      {
                                            {{0, 5}, {2, 3}, {3, 5}, {2, 4}},     // from
                                            {{0, 5}, {12, 13}, {3, 5}, {12, 14}}  // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{7, 8}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{7, 8}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      },
                                      {
                                            {{5, 6}, {2, 3}, {3, 5}, {1, 2}},   // from
                                            {{7, 8}, {12, 13}, {3, 5}, {6, 7}}  // to
                                      },
                                      {
                                            {{5, 6}, {2, 3}, {3, 5}, {2, 4}},     // from
                                            {{7, 8}, {12, 13}, {3, 5}, {12, 14}}  // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 6}, {0, 3}, {3, 5}, {1, 4}}) == ref_3);
    }  // o1v1o1v1
  }    // Block size 16

  //
  // Block size 4
  //
  SECTION("Block size 4") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 4;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    std::shared_ptr<MoSpaces> mospaces_ptr(new MoSpaces(tested, adcmem_ptr, {}, {}, {}));

    SECTION("Space o1o1") {
      MoIndexTranslation motrans(mospaces_ptr, "o1o1");
      CHECK(motrans.space() == "o1o1");
      CHECK(motrans.subspaces() == std::vector<std::string>{"o1", "o1"});
      CHECK(motrans.ndim() == 2);
      CHECK(motrans.shape() == vecsize{10, 10});

      CHECK_THROWS(motrans.full_index_of({1, 2}) == vecsize{1, 2});
      CHECK_THROWS(motrans.full_index_of({1, 5}) == vecsize{1, 7});
      CHECK_THROWS(motrans.full_index_of({7, 5}) == vecsize{9, 7});

      CHECK(motrans.block_index_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_of({1, 4}) == vecsize{0, 1});
      CHECK(motrans.block_index_of({1, 5}) == vecsize{0, 2});
      CHECK(motrans.block_index_of({7, 5}) == vecsize{2, 2});

      CHECK(motrans.block_index_spatial_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({1, 4}) == vecsize{0, 1});
      CHECK(motrans.block_index_spatial_of({1, 5}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({7, 5}) == vecsize{0, 0});

      CHECK(motrans.inblock_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.inblock_index_of({1, 3}) == vecsize{1, 0});
      CHECK(motrans.inblock_index_of({1, 5}) == vecsize{1, 0});
      CHECK(motrans.inblock_index_of({7, 5}) == vecsize{2, 0});

      CHECK(motrans.spin_of({1, 2}) == "aa");
      CHECK(motrans.spin_of({1, 3}) == "aa");
      CHECK(motrans.spin_of({1, 5}) == "ab");
      CHECK(motrans.spin_of({7, 5}) == "bb");

      std::vector<vecsize> cases{{1, 2}, {1, 3}, {1, 5}, {7, 5}, {9, 8}};
      for (auto idx : cases) {
        const auto result = motrans.split(idx);
        CHECK(result.first == motrans.block_index_of(idx));
        CHECK(result.second == motrans.inblock_index_of(idx));

        CHECK(motrans.combine(result.first, result.second) == idx);
        CHECK(motrans.combine(motrans.spin_of(idx), motrans.block_index_spatial_of(idx),
                              result.second) == idx);
      }

      CHECK(motrans.hf_provider_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.hf_provider_index_of({1, 3}) == vecsize{1, 3});
      CHECK(motrans.hf_provider_index_of({1, 5}) == vecsize{1, 7});
      CHECK(motrans.hf_provider_index_of({7, 5}) == vecsize{9, 7});

      // Map over no range split, but over a block split
      std::vector<RangeMapping> ref_1{{
            {{0, 5}, {0, 3}},  // from
            {{0, 5}, {0, 3}}   // to
      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 5}, {0, 3}}) == ref_1);
    }  // o1o1

    SECTION("Space o1v1o1v1") {
      MoIndexTranslation motrans(mospaces_ptr, "o1v1o1v1");

      CHECK(motrans.space() == "o1v1o1v1");
      CHECK(motrans.subspaces() == std::vector<std::string>{"o1", "v1", "o1", "v1"});
      CHECK(motrans.ndim() == 4);
      CHECK(motrans.shape() == vecsize{10, 4, 10, 4});

      CHECK_THROWS(motrans.full_index_of({1, 1, 3, 1}) == vecsize{1, 6, 2, 6});
      CHECK_THROWS(motrans.full_index_of({1, 1, 5, 3}) == vecsize{1, 5, 7, 13});
      CHECK_THROWS(motrans.full_index_of({7, 3, 5, 3}) == vecsize{9, 13, 7, 13});

      CHECK(motrans.block_index_of({1, 1, 3, 1}) == vecsize{0, 0, 1, 0});
      CHECK(motrans.block_index_of({1, 1, 5, 3}) == vecsize{0, 0, 2, 1});
      CHECK(motrans.block_index_of({7, 3, 5, 3}) == vecsize{2, 1, 2, 1});

      CHECK(motrans.block_index_spatial_of({1, 1, 3, 1}) == vecsize{0, 0, 1, 0});
      CHECK(motrans.block_index_spatial_of({1, 1, 5, 3}) == vecsize{0, 0, 0, 0});
      CHECK(motrans.block_index_spatial_of({7, 3, 5, 3}) == vecsize{0, 0, 0, 0});

      CHECK(motrans.inblock_index_of({1, 1, 3, 1}) == vecsize{1, 1, 0, 1});
      CHECK(motrans.inblock_index_of({1, 1, 5, 3}) == vecsize{1, 1, 0, 1});
      CHECK(motrans.inblock_index_of({7, 3, 5, 3}) == vecsize{2, 1, 0, 1});

      CHECK(motrans.spin_of({1, 1, 3, 1}) == "aaaa");
      CHECK(motrans.spin_of({1, 1, 5, 3}) == "aabb");
      CHECK(motrans.spin_of({7, 3, 5, 3}) == "bbbb");

      std::vector<vecsize> cases{{1, 1, 3, 1}, {1, 1, 5, 3}, {7, 3, 5, 3}};
      for (auto idx : cases) {
        const auto result = motrans.split(idx);
        CHECK(result.first == motrans.block_index_of(idx));
        CHECK(result.second == motrans.inblock_index_of(idx));

        CHECK(motrans.combine(result.first, result.second) == idx);
        CHECK(motrans.combine(motrans.spin_of(idx), motrans.block_index_spatial_of(idx),
                              result.second) == idx);
      }

      CHECK(motrans.hf_provider_index_of({1, 1, 3, 1}) == vecsize{1, 6, 3, 6});
      CHECK(motrans.hf_provider_index_of({1, 1, 5, 3}) == vecsize{1, 6, 7, 13});
      CHECK(motrans.hf_provider_index_of({7, 3, 5, 3}) == vecsize{9, 13, 7, 13});

      // Map a single block
      std::vector<RangeMapping> ref_1{{
            {{0, 4}, {0, 2}, {0, 3}, {0, 1}},  // from
            {{0, 4}, {5, 7}, {0, 3}, {5, 6}}   // to
      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 4}, {0, 2}, {0, 3}, {0, 1}}) == ref_1);

      // Map over two range splits and a few block splits
      std::vector<RangeMapping> ref_2{{
                                            {{0, 5}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{0, 5}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{0, 5}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{0, 5}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {1, 2}},  // from
                                            {{7, 8}, {5, 7}, {3, 5}, {6, 7}}   // to
                                      },
                                      {
                                            {{5, 6}, {0, 2}, {3, 5}, {2, 4}},   // from
                                            {{7, 8}, {5, 7}, {3, 5}, {12, 14}}  // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 6}, {0, 2}, {3, 5}, {1, 4}}) == ref_2);
    }  // o1v1o1v1
  }    // Block size 4

  SECTION("Scattered CVS space") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    std::shared_ptr<MoSpaces> mospaces_ptr(
          new MoSpaces(tested, adcmem_ptr, {3, 8}, {}, {}));

    SECTION("Space o1o1") {
      MoIndexTranslation motrans(mospaces_ptr, "o1o1");
      CHECK(motrans.space() == "o1o1");
      CHECK(motrans.subspaces() == std::vector<std::string>{"o1", "o1"});
      CHECK(motrans.ndim() == 2);
      CHECK(motrans.shape() == vecsize{8, 8});

      CHECK_THROWS(motrans.full_index_of({1, 2}) == vecsize{1, 2});
      CHECK_THROWS(motrans.full_index_of({1, 5}) == vecsize{1, 9});
      CHECK_THROWS(motrans.full_index_of({7, 5}) == vecsize{11, 9});

      CHECK(motrans.block_index_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_of({1, 3}) == vecsize{0, 0});
      CHECK(motrans.block_index_of({1, 5}) == vecsize{0, 1});
      CHECK(motrans.block_index_of({7, 5}) == vecsize{1, 1});

      CHECK(motrans.block_index_spatial_of({1, 2}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({1, 3}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({1, 5}) == vecsize{0, 0});
      CHECK(motrans.block_index_spatial_of({7, 5}) == vecsize{0, 0});

      CHECK(motrans.inblock_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.inblock_index_of({1, 3}) == vecsize{1, 3});
      CHECK(motrans.inblock_index_of({1, 5}) == vecsize{1, 1});
      CHECK(motrans.inblock_index_of({7, 5}) == vecsize{3, 1});

      CHECK(motrans.spin_of({1, 2}) == "aa");
      CHECK(motrans.spin_of({1, 3}) == "aa");
      CHECK(motrans.spin_of({1, 5}) == "ab");
      CHECK(motrans.spin_of({7, 5}) == "bb");

      std::vector<vecsize> cases{{1, 2}, {1, 5}, {7, 5}};
      for (auto idx : cases) {
        const auto result = motrans.split(idx);
        CHECK(result.first == motrans.block_index_of(idx));
        CHECK(result.second == motrans.inblock_index_of(idx));

        CHECK(motrans.combine(result.first, result.second) == idx);
        CHECK(motrans.combine(motrans.spin_of(idx), motrans.block_index_spatial_of(idx),
                              result.second) == idx);
      }

      CHECK(motrans.hf_provider_index_of({1, 2}) == vecsize{1, 2});
      CHECK(motrans.hf_provider_index_of({1, 5}) == vecsize{1, 9});
      CHECK(motrans.hf_provider_index_of({7, 5}) == vecsize{11, 9});

      // Map to two blocks
      std::vector<RangeMapping> ref_1{{
                                            {{0, 3}, {0, 3}},  // from
                                            {{0, 3}, {0, 3}}   // to
                                      },
                                      {
                                            {{3, 4}, {0, 3}},  // from
                                            {{4, 5}, {0, 3}}   // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 4}, {0, 3}}) == ref_1);

      // Map to four blocks
      std::vector<RangeMapping> ref_2{{
                                            {{0, 3}, {0, 3}},  // from
                                            {{0, 3}, {0, 3}}   // to
                                      },
                                      {
                                            {{3, 4}, {0, 3}},  // from
                                            {{4, 5}, {0, 3}}   // to
                                      },
                                      {
                                            {{4, 5}, {0, 3}},  // from
                                            {{7, 8}, {0, 3}}   // to
                                      },
                                      {
                                            {{5, 6}, {0, 3}},  // from
                                            {{9, 10}, {0, 3}}  // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{0, 6}, {0, 3}}) == ref_2);

      // Map to six blocks
      std::vector<RangeMapping> ref_3{{
                                            {{3, 4}, {4, 5}},  // from
                                            {{4, 5}, {7, 8}}   // to
                                      },
                                      {
                                            {{3, 4}, {5, 6}},  // from
                                            {{4, 5}, {9, 10}}  // to
                                      },
                                      {
                                            {{4, 5}, {4, 5}},  // from
                                            {{7, 8}, {7, 8}}   // to
                                      },
                                      {
                                            {{4, 5}, {5, 6}},  // from
                                            {{7, 8}, {9, 10}}  // to
                                      },
                                      {
                                            {{5, 6}, {4, 5}},  // from
                                            {{9, 10}, {7, 8}}  // to
                                      },
                                      {
                                            {{5, 6}, {5, 6}},   // from
                                            {{9, 10}, {9, 10}}  // to
                                      }};
      CHECK(motrans.map_range_to_hf_provider({{3, 6}, {4, 6}}) == ref_3);
    }  // o1o1
  }    // Scattered CVS space
}  // MoIndexTranslation
}  // namespace tests
}  // namespace libadcc
