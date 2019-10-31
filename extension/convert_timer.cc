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

#include "convert_timer.hh"

namespace adcc {
namespace py_iface {
namespace py = pybind11;

py::object convert_timer(const Timer& timer) {
  // Determine the shift between the C++ and the python clocks
  py::object pynow         = py::module::import("time").attr("perf_counter");
  const double clock_shift = pynow().cast<double>() - Timer::now();

  // Shift the data and convert to python
  py::dict shifted_intervals;
  for (const auto& kv : timer.intervals) {
    py::list intlist;
    for (const auto& p : kv.second) {
      intlist.append(py::make_tuple(clock_shift + p.first, clock_shift + p.second));
    }
    shifted_intervals[py::cast(kv.first)] = intlist;
  }
  py::dict shifted_start_times;
  for (const auto& kv : timer.start_times) {
    shifted_start_times[py::cast(kv.first)] = kv.second + clock_shift;
  }

  py::object pyTimer            = py::module::import("adcc.timings").attr("Timer");
  py::object ret                = pyTimer();
  ret.attr("intervals")         = shifted_intervals;
  ret.attr("start_times")       = shifted_start_times;
  ret.attr("time_construction") = timer.time_construction + clock_shift;
  return ret;
}

}  // namespace py_iface
}  // namespace adcc
