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

#pragma once
#include <chrono>
#include <map>
#include <string>
#include <vector>

namespace libadcc {
/**
 *  \addtogroup Utilities
 */
///@{

/** Minimalistic timer class: Just records the start and stop times for tasks,
 *  i.e. the intervals they ran */
class Timer {
 public:
  /** A float representing the current time on the clock used for measurement.
   * For consistency, this function should always be used to obtain a representation
   * of "now". The unit is seconds. */
  static double now();

  /** Construct an empty timer object*/
  Timer() : time_construction(now()), intervals{}, start_times{} {}

  /** Start a particular task */
  void start(const std::string& task);

  /** Stop the task again */
  double stop(const std::string& task);

  /** Time when this class was constructed */
  double time_construction;

  /** The intervals stored for a particular key */
  std::map<std::string, std::vector<std::pair<double, double>>> intervals;

  /** The start time stored for each key */
  std::map<std::string, double> start_times;
};

/** RAI-like class for timing tasks. Construction starts the timer, destruction
 *  ends it automatically. It is assumed that the passed task has a longer lifetime
 *  than the RecordTime object
 */
class RecordTime {
 public:
  /** Construct the class and start the timer */
  RecordTime(Timer& timer_, const std::string& task_) : timer(timer_), task(task_) {
    timer.start(task);
  }
  ~RecordTime() { timer.stop(task); }
  RecordTime(const RecordTime&) = delete;
  RecordTime& operator=(const RecordTime&) = delete;

  Timer& timer;      //!< Timer object, which is managed
  std::string task;  //!< String describing the task
};

///@}
}  // namespace libadcc
