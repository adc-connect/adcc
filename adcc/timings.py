#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import time

from contextlib import contextmanager
from os.path import join


def strtime_short(span):
    """
    Return a 5-character string identifying the timespan
    """
    if span < 1:
        return "{:3d}ms".format(int(span * 1000))
    if span < 60:
        return "{:4.1f}s".format(span)
    if span < 120:
        return "{:4d}s".format(int(span))
    if span < 3600:
        return "{:4.1f}m".format(span / 60)
    if span < 86400:
        return "{:4.1f}h".format(span / 3600)
    else:
        return "{:4.1f}d".format(span / 3600 / 24)


def strtime(span):
    """
    Return a moderately long string, providing a human-interpretable
    representation of the provided timespan
    """
    if span < 1:
        return "{:6.3f}ms".format(span * 1000)
    if span < 120:
        full = int(span)
        return "{:3d}s {:3d}ms".format(full, int((span - full) * 1000))
    if span < 3600:
        full = int(span / 60)
        return "{:2d}m {:2d}s".format(full, int(span - full * 60))
    if span < 86400:
        full = int(span / 3600)
        return "{:2d}h {:2d}m".format(full, int(span / 60 - full * 60))
    else:
        full = int(span / 86400)
        return "{:3d}d {:2d}h".format(full, int(span / 3600 - full * 24))


class Timer:
    # TODO More flexible: Time on a subtree
    # TODO Describe function to print a nice table
    def __init__(self):
        self.time_construction = time.perf_counter()
        self.intervals = {}
        self.start_times = {}

    def attach(self, other, subtree=""):
        """
        Attach the timing results from another timer,
        i.e. merge both timers together.
        """
        for k, v in other.intervals.items():
            kfull = join(subtree, k)
            if kfull not in self.intervals:
                self.intervals[kfull] = []
            self.intervals[kfull].extend(v)

        for k, v in other.start_times.items():
            kfull = join(subtree, k)
            if kfull not in self.start_times:
                self.start_times[kfull] = []
            self.start_times[kfull].extend(v)

    def stop(self, task, now=None):
        """Stop a task and return runtime of it."""
        if now is None:
            now = time.perf_counter()
        if task in self.start_times:
            start = self.start_times.pop(task)
            if task in self.intervals:
                self.intervals[task].append((start, now))
            else:
                self.intervals[task] = [(start, now)]
            return now - start
        return 0

    def restart(self, task):
        """
        Start a task if it is currently not running
        or stop and restart otherwise.
        """
        now = time.perf_counter()
        if self.is_running(task):
            self.stop(task, now)
        self.start_times[task] = now

    @contextmanager
    def record(self, task):
        """
        Context manager to automatically start and stop a time
        recording as long as context is active.

        Parameters
        ----------
        task : str
            The string describing the task
        """
        self.restart(task)
        try:
            yield self
        finally:
            self.stop(task)

    def is_running(self, task):
        return task in self.start_times

    @property
    def tasks(self):
        """The list of all tasks known to this object"""
        all_tasks = set(self.start_times.keys())
        all_tasks.update(set(self.intervals.keys()))
        return sorted(list(all_tasks))

    @property
    def lifetime(self):
        """Get total time since this class has been constructed"""
        return time.perf_counter() - self.time_construction

    def total(self, task):
        """Get total runtime on a task in seconds"""
        if task not in self.intervals:
            if task not in self.start_times:
                raise ValueError("Unknown task: " + task)
            return self.current(task)
        else:
            cur = 0
            if task in self.start_times:
                cur = time.perf_counter() - self.start_times[task]
            cumul = sum(end - start for start, end in self.intervals[task])
            return cur + cumul

    def current(self, task):
        """Get current time on a task without stopping it"""
        if self.is_running(task):
            return time.perf_counter() - self.start_times[task]
        else:
            raise ValueError("Task not currently running: " + task)

    def describe(self):
        maxlen = max(len(key) for key in self.tasks)

        # This is very dummy ... in the future we would like to have
        # a nice little table, which also respects the key hierachy
        # and displays cumulative sums for each level and which
        # tasks care of duplicated time intervals (e.g. if two tasks
        # run at the same time)
        # Colour: Use one for each level
        text = "Timer " + strtime_short(self.lifetime) + " lifetime:\n"
        for key in self.tasks:
            fmt = "  {:<" + str(maxlen) + "s} {:>20s}\n"
            text += fmt.format(key, strtime(self.total(key)))
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("Timer()")
        else:
            pp.text(self.describe())


Timer.start = Timer.restart
