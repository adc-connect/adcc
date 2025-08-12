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

#include "Timer.hh"
#include "exceptions.hh"

namespace libadcc {

double Timer::now() {
  using namespace std::chrono;
  // fix reference on first use, else just get current time point (jetzt)
  // and return number of seconds since the reference
  static auto reference = high_resolution_clock::now();
  const auto jetzt      = high_resolution_clock::now();
  return duration_cast<duration<double>>(jetzt - reference).count();
}

double Timer::stop(const std::string& task) {
  const double end = now();

  auto itstart = start_times.find(task);
  if (itstart == start_times.end()) {
    throw invalid_argument("Task " + task + " not running.");
  }
  const double start = itstart->second;
  start_times.erase(itstart);

  // Add an interval if a list already exists,
  // else create a list with one element.
  auto itinter = intervals.find(task);
  if (itinter != intervals.end()) {
    itinter->second.emplace_back(start, end);
  } else {
    intervals[task] = {{start, end}};
  }
  return end - start;
}

void Timer::start(const std::string& task) {
  auto itstart = start_times.find(task);
  if (itstart != start_times.end()) {
    throw invalid_argument("Task " + task + " already running.");
  }
  start_times[task] = now();
}

}  // namespace libadcc
