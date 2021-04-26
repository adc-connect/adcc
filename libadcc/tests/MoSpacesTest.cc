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

#include "../MoSpaces.hh"
#include "HFSolutionMock.hh"
#include <catch2/catch.hpp>

namespace libadcc {
namespace tests {

TEST_CASE("Test MoSpaces", "[MoSpaces]") {
  HFSolutionMock hf_h2o;
  auto adcmem_ptr = std::make_shared<AdcMemory>();

  SECTION("Block size 16") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "v1"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 5);
    CHECK(mo.n_orbs_beta("o1") == 5);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 3);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 6, 12, 13});

    CHECK(mo.map_block_start.size() == 3);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 5, 7, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 5});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 3);
    CHECK(mo.map_block_irrep["f"] == std::vector<std::string>{"A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 3);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});

    CHECK(mo.n_orbs("f") == mo.n_orbs_alpha("f") + mo.n_orbs_beta("f"));
    CHECK(mo.n_orbs("o1") == mo.n_orbs_alpha("o1") + mo.n_orbs_beta("o1"));
    CHECK(mo.n_orbs("v1") == mo.n_orbs_alpha("v1") + mo.n_orbs_beta("v1"));
    CHECK(mo.n_orbs_alpha() == mo.n_orbs_alpha("f"));
    CHECK(mo.n_orbs_beta() == mo.n_orbs_beta("f"));
    CHECK(mo.n_orbs() == mo.n_orbs("f"));
  }  // block size 16

  SECTION("Block size 4") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 4;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "v1"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 5);
    CHECK(mo.n_orbs_beta("o1") == 5);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 3);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 6, 12, 13});

    CHECK(mo.map_block_start.size() == 3);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 3, 5, 7, 10, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 3, 5, 8});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 3);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 3);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'a', 'b', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
  }  // block size 4

  SECTION("Block size 16, 1 core (lowest)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {0, 7}, {}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o2"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o2", "v1"});
    CHECK(mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 4);
    CHECK(mo.n_orbs_beta("o1") == 4);
    CHECK(mo.n_orbs_alpha("o2") == 1);
    CHECK(mo.n_orbs_beta("o2") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{1, 2, 3, 4, 8, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["o2"] == std::vector<size_t>{0, 7});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 6, 12, 13});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 5, 7, 8, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 4});
    CHECK(mo.map_block_start["o2"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o2"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o2"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
  }  // 1 core, lowest

  SECTION("Block size 16, 1 core (scattered)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {3, 8}, {}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o2"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o2", "v1"});
    CHECK(mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 4);
    CHECK(mo.n_orbs_beta("o1") == 4);
    CHECK(mo.n_orbs_alpha("o2") == 1);
    CHECK(mo.n_orbs_beta("o2") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{3, 0, 1, 2, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 4, 7, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["o2"] == std::vector<size_t>{3, 8});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 6, 12, 13});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 5, 7, 8, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 4});
    CHECK(mo.map_block_start["o2"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o2"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o2"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
  }  // 1 core, scattered

  SECTION("1 frozen virtual, highest") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {}, {6, 13});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1", "v2"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "v1", "v2"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 5);
    CHECK(mo.n_orbs_beta("o1") == 5);
    CHECK(mo.n_orbs_alpha("v1") == 1);
    CHECK(mo.n_orbs_beta("v1") == 1);
    CHECK(mo.n_orbs_alpha("v2") == 1);
    CHECK(mo.n_orbs_beta("v2") == 1);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 12});
    CHECK(mo.map_index_hf_provider["v2"] == std::vector<size_t>{6, 13});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 5, 6, 7, 12, 13});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 5});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v2"] == std::vector<size_t>{0, 1});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v2"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v2"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen virtual (highest)

  SECTION("1 frozen virtual, scattered") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {}, {5, 11});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1", "v2"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "v1", "v2"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 5);
    CHECK(mo.n_orbs_beta("o1") == 5);
    CHECK(mo.n_orbs_alpha("v1") == 1);
    CHECK(mo.n_orbs_beta("v1") == 1);
    CHECK(mo.n_orbs_alpha("v2") == 1);
    CHECK(mo.n_orbs_beta("v2") == 1);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 12, 13, 11});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 7, 8, 9, 10, 12});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{6, 13});
    CHECK(mo.map_index_hf_provider["v2"] == std::vector<size_t>{5, 11});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 5, 6, 7, 12, 13});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 5});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v2"] == std::vector<size_t>{0, 1});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v2"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v2"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen virtual (scattered)

  SECTION("Block size 16, 1 frozen core (lowest)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {0, 7}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o3"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o3", "v1"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 4);
    CHECK(mo.n_orbs_beta("o1") == 4);
    CHECK(mo.n_orbs_alpha("o3") == 1);
    CHECK(mo.n_orbs_beta("o3") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{1, 2, 3, 4, 8, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["o3"] == std::vector<size_t>{0, 7});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 6, 12, 13});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 5, 7, 8, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 4});
    CHECK(mo.map_block_start["o3"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o3"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o3"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen core, lowest

  SECTION("Block size 16, 1 frozen core (scattered)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {}, {6, 8}, {});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o3"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o3", "v1"});
    CHECK(!mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 4);
    CHECK(mo.n_orbs_beta("o1") == 4);
    CHECK(mo.n_orbs_alpha("o3") == 1);
    CHECK(mo.n_orbs_beta("o3") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 2);
    CHECK(mo.n_orbs_beta("v1") == 2);

    CHECK(mo.map_index_hf_provider.size() == 4);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{6, 0, 1, 2, 3, 4, 5, 8, 7, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] ==
          std::vector<size_t>{0, 1, 2, 3, 7, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["o3"] == std::vector<size_t>{6, 8});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{4, 5, 12, 13});

    CHECK(mo.map_block_start.size() == 4);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 5, 7, 8, 12});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 4});
    CHECK(mo.map_block_start["o3"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 2});

    CHECK(mo.map_block_irrep.size() == 4);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o3"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 4);
    CHECK(mo.map_block_spin["f"] == std::vector<char>{'a', 'a', 'a', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o3"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen core, scattered

  SECTION("Block size 16, 1 frozen core, 1 core, 1 frozen virtual (logical)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    MoSpaces mo(tested, adcmem_ptr, {1, 8}, {0, 7}, {6, 13});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o2", "o3"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1", "v2"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o2", "o3", "v1", "v2"});
    CHECK(mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 3);
    CHECK(mo.n_orbs_beta("o1") == 3);
    CHECK(mo.n_orbs_alpha("o2") == 1);
    CHECK(mo.n_orbs_beta("o2") == 1);
    CHECK(mo.n_orbs_alpha("o3") == 1);
    CHECK(mo.n_orbs_beta("o3") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 1);
    CHECK(mo.n_orbs_beta("v1") == 1);
    CHECK(mo.n_orbs_alpha("v2") == 1);
    CHECK(mo.n_orbs_beta("v2") == 1);

    CHECK(mo.map_index_hf_provider.size() == 6);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});
    CHECK(mo.map_index_hf_provider["o1"] == std::vector<size_t>{2, 3, 4, 9, 10, 11});
    CHECK(mo.map_index_hf_provider["o2"] == std::vector<size_t>{1, 8});
    CHECK(mo.map_index_hf_provider["o3"] == std::vector<size_t>{0, 7});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{5, 12});
    CHECK(mo.map_index_hf_provider["v2"] == std::vector<size_t>{6, 13});

    CHECK(mo.map_block_start.size() == 6);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 2, 5, 6, 7, 8, 9, 12, 13});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 3});
    CHECK(mo.map_block_start["o2"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["o3"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v2"] == std::vector<size_t>{0, 1});

    CHECK(mo.map_block_irrep.size() == 6);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o2"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o3"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v2"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 6);
    CHECK(mo.map_block_spin["f"] ==
          std::vector<char>{'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o2"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o3"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v2"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen core, 1 core, 1 frozen virtual, logical

  SECTION("Block size 16, 1 frozen core, 1 core, 1 frozen virtual (scattered)") {
    const HFSolutionMock& tested = hf_h2o;
    const size_t block_size      = 16;
    adcmem_ptr->initialise("/tmp", block_size, "standard");
    //                               core   fzcore   fzvirt
    MoSpaces mo(tested, adcmem_ptr, {3, 9}, {1, 8}, {4, 10});

    CHECK(mo.point_group == "C1");
    CHECK(mo.irreps == std::vector<std::string>{"A"});
    CHECK(mo.subspaces_occupied == std::vector<std::string>{"o1", "o2", "o3"});
    CHECK(mo.subspaces_virtual == std::vector<std::string>{"v1", "v2"});
    CHECK(mo.subspaces == std::vector<std::string>{"o1", "o2", "o3", "v1", "v2"});
    CHECK(mo.has_core_occupied_space());

    CHECK(mo.n_orbs_alpha("f") == 7);
    CHECK(mo.n_orbs_beta("f") == 7);
    CHECK(mo.n_orbs_alpha("o1") == 3);
    CHECK(mo.n_orbs_beta("o1") == 3);
    CHECK(mo.n_orbs_alpha("o2") == 1);
    CHECK(mo.n_orbs_beta("o2") == 1);
    CHECK(mo.n_orbs_alpha("o3") == 1);
    CHECK(mo.n_orbs_beta("o3") == 1);
    CHECK(mo.n_orbs_alpha("v1") == 1);
    CHECK(mo.n_orbs_beta("v1") == 1);
    CHECK(mo.n_orbs_alpha("v2") == 1);
    CHECK(mo.n_orbs_beta("v2") == 1);

    CHECK(mo.map_index_hf_provider.size() == 6);
    CHECK(mo.map_index_hf_provider["f"] ==
          std::vector<size_t>{1, 3, 0, 2, 5, 6, 4, 8, 9, 7, 11, 12, 13, 10});
    CHECK(mo.map_index_hf_provider["o1"] == std::vector<size_t>{0, 2, 5, 7, 11, 12});
    CHECK(mo.map_index_hf_provider["o2"] == std::vector<size_t>{3, 9});
    CHECK(mo.map_index_hf_provider["o3"] == std::vector<size_t>{1, 8});
    CHECK(mo.map_index_hf_provider["v1"] == std::vector<size_t>{6, 13});
    CHECK(mo.map_index_hf_provider["v2"] == std::vector<size_t>{4, 10});

    CHECK(mo.map_block_start.size() == 6);
    CHECK(mo.map_block_start["f"] == std::vector<size_t>{0, 1, 2, 5, 6, 7, 8, 9, 12, 13});
    CHECK(mo.map_block_start["o1"] == std::vector<size_t>{0, 3});
    CHECK(mo.map_block_start["o2"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["o3"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v1"] == std::vector<size_t>{0, 1});
    CHECK(mo.map_block_start["v2"] == std::vector<size_t>{0, 1});

    CHECK(mo.map_block_irrep.size() == 6);
    CHECK(mo.map_block_irrep["f"] ==
          std::vector<std::string>{"A", "A", "A", "A", "A", "A", "A", "A", "A", "A"});
    CHECK(mo.map_block_irrep["o1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o2"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["o3"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v1"] == std::vector<std::string>{"A", "A"});
    CHECK(mo.map_block_irrep["v2"] == std::vector<std::string>{"A", "A"});

    CHECK(mo.map_block_spin.size() == 6);
    CHECK(mo.map_block_spin["f"] ==
          std::vector<char>{'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'});
    CHECK(mo.map_block_spin["o1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o2"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["o3"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v1"] == std::vector<char>{'a', 'b'});
    CHECK(mo.map_block_spin["v2"] == std::vector<char>{'a', 'b'});
  }  // 1 frozen core, 1 core, 1 frozen virtual, scattered
}  // MoSpaces

}  // namespace tests
}  // namespace libadcc
