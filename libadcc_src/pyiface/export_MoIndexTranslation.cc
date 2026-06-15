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
#include "util.hh"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace libadcc {

namespace py = pybind11;

static std::vector<size_t> parse_tuple(size_t ndim, const py::tuple& tuple) {
  if (tuple.size() != ndim) {
    throw py::value_error(
          "Number of elements passed in index tuple (== " + std::to_string(tuple.size()) +
          ") and dimensionality (== " + std::to_string(ndim) + ") do not agree.");
  }

  std::vector<size_t> ret(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    ret[i] = tuple[i].cast<size_t>();
  }
  return ret;
}

static py::tuple convert_range_to_tuples(const SimpleRange& range) {
  py::tuple ret(range.size());
  for (size_t i = 0; i < range.size(); ++i) {
    ret[i] = py::make_tuple(range[i].first, range[i].second);
  }
  return ret;
}

static py::list MoIndexTranslation_map_range_to_hf_provider(
      std::shared_ptr<MoIndexTranslation> self, py::tuple ranges) {
  if (ranges.size() != self->ndim()) {
    throw py::value_error("Number of elements passed in the index range tuple (== " +
                          std::to_string(ranges.size()) + ") and dimensionality (== " +
                          std::to_string(self->ndim()) + ") do not agree.");
  }

  // Convert input tuples
  std::vector<std::pair<size_t, size_t>> parsed;
  for (size_t i = 0; i < self->ndim(); ++i) {
    try {
      auto elem = ranges[i].cast<py::tuple>();
      if (elem.size() != 2) {
        throw py::value_error(
              "The range argument passed to map_range_to_hf_provider needs to be given "
              "as a tuple of pairs.");
      }
      parsed.emplace_back(elem[0].cast<size_t>(), elem[1].cast<size_t>());
    } catch (const py::cast_error& c) {
      throw py::cast_error(
            "The range argument passed to map_range_to_hf_provider needs to be given as "
            "a tuple of pairs of indices defining, for each axis, the half-open index "
            "range [start, end)");
    }
  }

  // Call inner function and convert back to native python datastructures
  std::vector<RangeMapping> res = self->map_range_to_hf_provider(parsed);

  py::list ret;
  for (const RangeMapping& map : res) {
    py::dict d;
    d["from"] = convert_range_to_tuples(map.from());
    d["to"]   = convert_range_to_tuples(map.to());
    ret.append(d);
  }
  return ret;
}

void export_MoIndexTranslation(py::module& m) {

  py::class_<MoIndexTranslation, std::shared_ptr<MoIndexTranslation>>(
        m, "MoIndexTranslation",
        "Helper object to extract information from indices into orbitals subspaces and "
        "to map them between different indexing conventions (full MO space, MO "
        "subspaces, indexing convention in the HF Provider / SCF host program, ... "
        "Python binding to :cpp:class:`libadcc::MoIndexTranslation`.")
        .def(py::init<std::shared_ptr<const MoSpaces>, const std::string&>(),
             "Construct a MoIndexTranslation class from an MoSpaces object and the "
             "identifier for "
             "the space (e.g. o1o1, v1o1, o3v2o1v1, ...)")
        .def(py::init<std::shared_ptr<const MoSpaces>, const std::vector<std::string>&>(),
             "Construct a MoIndexTranslation class from an MoSpaces object and the "
             "list of identifiers for the space (e.g. [\"o1\", \"o1\"] ...)")
        .def_property_readonly("subspaces", &MoIndexTranslation::subspaces)
        .def_property_readonly("mospaces", &MoIndexTranslation::mospaces_ptr,
                               "Return the MoSpaces object supplied on initialisation")
        .def_property_readonly("space", &MoIndexTranslation::space,
                               "Return the space supplied on initialisation.")
        .def_property_readonly("ndim", &MoIndexTranslation::ndim,
                               "Return the number of dimensions.")
        .def_property_readonly(
              "shape",
              [](std::shared_ptr<MoIndexTranslation> self) {
                return shape_tuple(self->shape());
              },
              "Return the length along each dimension.")
        //
        .def(
              "full_index_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                return shape_tuple(self->full_index_of(parse_tuple(self->ndim(), tpl)));
              },
              "Map an index given in the space, which was passed upon construction, to "
              "the corresponding index in the full MO index range (the ffff space).")
        .def(
              "block_index_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                return shape_tuple(self->block_index_of(parse_tuple(self->ndim(), tpl)));
              },
              "Get the block index of an index, i.e. get the index which points to the "
              "block of the tensor in which the element with the passed index is "
              "contained in.")
        .def(
              "block_index_spatial_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                return shape_tuple(
                      self->block_index_spatial_of(parse_tuple(self->ndim(), tpl)));
              },
              "Get the spatial block index of an index\n"
              "\n"
              "The spatial block index is the result of block_index_of modulo the spin "
              "blocks,\n"
              "i.e. it maps an index onto the index of the *spatial* blocks only, such "
              "that\n"
              "the resulting value is identical for two index where the MOs only differ\n"
              "by spin. For example the 1st core alpha and the 1st core beta orbital "
              "will\n"
              "map to the same value upon a call of this function.")
        .def(
              "inblock_index_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                return shape_tuple(
                      self->inblock_index_of(parse_tuple(self->ndim(), tpl)));
              },
              "Get the in-block index, i.e. the index within the tensor block.")
        .def(
              "spin_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                return self->spin_of(parse_tuple(self->ndim(), tpl));
              },
              "Get the spin block of each of the index components as a string.")
        .def(
              "split",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                auto splitted = self->split(parse_tuple(self->ndim(), tpl));
                return py::make_tuple(shape_tuple(splitted.first),
                                      shape_tuple(splitted.second));
              },
              "Split an index into block index and in-block index")
        .def(
              "split_spin",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple tpl) {
                auto splitted = self->split_spin(parse_tuple(self->ndim(), tpl));
                return py::make_tuple(std::get<0>(splitted),
                                      shape_tuple(std::get<1>(splitted)),
                                      shape_tuple(std::get<2>(splitted)));
              },
              "Split an index into a spin block descriptor, a spatial block index and an "
              "in-block index.")
        .def(
              "combine",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple bidx,
                 py::tuple ibidx) {
                return shape_tuple(self->combine(parse_tuple(self->ndim(), bidx),
                                                 parse_tuple(self->ndim(), ibidx)));
              },
              "Combine a block index and an in-block index into the appropriate index. "
              "Effectively undoes the effect of 'split'.")
        .def(
              "combine",
              [](std::shared_ptr<MoIndexTranslation> self, std::string spin_block,
                 py::tuple bidx, py::tuple ibidx) {
                return shape_tuple(self->combine(spin_block,
                                                 parse_tuple(self->ndim(), bidx),
                                                 parse_tuple(self->ndim(), ibidx)));
              },
              "Combine a spin block (given as a string of 'a's or 'b's), a spatial-only "
              "block index and an in-block index into the appropriate index. Essentially "
              "undoes the effect of 'spin_of', 'block_index_spatial_of' and "
              "'inblock_index_of'.")
        .def(
              "hf_provider_index_of",
              [](std::shared_ptr<MoIndexTranslation> self, py::tuple index) {
                return shape_tuple(
                      self->hf_provider_index_of(parse_tuple(self->ndim(), index)));
              },
              "Map an index (given in the space passed upon construction) to the "
              "indexing "
              "convention of the host program provided to adcc as the HF provider.")
        //
        .def("map_range_to_hf_provider", &MoIndexTranslation_map_range_to_hf_provider,
             "Map a range of indices to host program indices, i.e. the indexing "
             "convention\n"
             "used in the HfProvider, which provides the SCF data to adcc.\n"
             "\n"
             "Since the mapping between subspace and host program indices might not be "
             "contiguous,\n"
             "a list of pairs of ranges is returned. In each pair, the first entry "
             "represents a\n"
             "range of indices (indexed in the MO subspace) and the second entry "
             "represents\n"
             "the equivalent range of indices in the Hartree-Fock provider these are "
             "mapped to.\n"
             "\n"
             "  ranges    Tuple of pairs of indices: One index pair for each dimension. "
             "Each\n"
             "            pair describes the range of indices along one axis, which "
             "should be\n"
             "            mapped to the indexing convention of the HfProvider. The range "
             "should\n"
             "            be thought of as a half-open interval [start, end), where "
             "start and\n"
             "            end are the indexed passed as a pair to the function.")
        //
        ;
}

}  // namespace libadcc
