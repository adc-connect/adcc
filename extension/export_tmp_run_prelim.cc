//
// Copyright (C) 2018 by the adcc authors
//
// This file is part of adcc.
//
// adcc is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// adcc is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with adcc. If not, see <http://www.gnu.org/licenses/>.
//

#include <adcc/tmp_run_prelim.hh>
#include <pybind11/pybind11.h>

namespace adcc {
namespace py_iface {

namespace py = pybind11;

using amplitudes_t = std::vector<std::shared_ptr<Tensor>>;
static py::list guesses_to_list(const std::vector<amplitudes_t>& guesses) {
  py::list ret;
  for (auto& guess : guesses) {
    py::list py_guess_vec;
    for (auto& part : guess) {
      py_guess_vec.append(part);
    }
    ret.append(py_guess_vec);
  }
  return ret;
}

static py::list TmpResult_guesses_singlet(const TmpResult& self) {
  return guesses_to_list(self.guesses_singlet);
}

static py::list TmpResult_guesses_triplet(const TmpResult& self) {
  return guesses_to_list(self.guesses_triplet);
}

static py::list TmpResult_guesses_state(const TmpResult& self) {
  return guesses_to_list(self.guesses_state);
}

/** Exports adcc/tmp_run_prelim.hh to python */
void export_tmp_run_prelim(py::module& m) {

  py::class_<TmpResult, std::shared_ptr<TmpResult>>(
        m, "TmpResult", "Temporary class to hold ADCman results")
        .def(py::init<>())
        .def_readonly("reference", &TmpResult::reference_ptr,
                      "The reference state to use for the ADC calculation.")
        .def_readonly("mp", &TmpResult::mp_ptr, "The MP results already obtained.")
        .def_readonly("intermediates", &TmpResult::intermediates_ptr,
                      "The ADC intermediates already calculated.")
        .def_readonly("ctx", &TmpResult::ctx_ptr,
                      "The full context as returned from the preliminary adcman run.")
        .def_readonly("have_singlet_and_triplet", &TmpResult::have_singlet_and_triplet,
                      "Are guesses_singlet and guesses_triplet populated (True) or "
                      "guesses_states (False).")
        .def_property_readonly(
              "guesses_singlet", &TmpResult_guesses_singlet,
              "The ADC guess vectors for computing singlet states. (only "
              "non-empty if have_singlet_and_triplet == True)")
        .def_property_readonly(
              "guesses_triplet", &TmpResult_guesses_triplet,
              "The ADC guess vectors for computing triplet states. (only "
              "non-empty if have_singlet_and_triplet == True)")
        .def_property_readonly(
              "guesses_state", &TmpResult_guesses_state,
              "The ADC guess vectors for computing states of unspecified spin "
              "symmetry (only non-empty if have_singlet_and_triplet == False).")
        .def_readonly(
              "method", &TmpResult::method,
              "The method for which the guess vectors in this class represent guesses")
        //
        ;

  m.def("tmp_run_prelim", &tmp_run_prelim,
        "Run adcman to get a preliminary state for starting one's own solver "
        "implementation.");
}

}  // namespace py_iface
}  // namespace adcc
