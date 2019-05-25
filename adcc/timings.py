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
    # TODO
    # The functionality can be improved by having a tree
    # structure where tree times can be totaled or something like that
    #
    # It should be made usable as a context manager
    #
    def __init__(self):
        self.time_construction = time.clock()
        self.intervals = {}
        self.start_times = {}

    def stop(self, task, now=None):
        """Stop a task and return runtime of it."""
        if now is None:
            now = time.clock()
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
        now = time.clock()
        if self.is_running(task):
            self.stop(task, now)
        self.start_times[task] = now

    def is_running(self, task):
        return task in self.start_times

    @property
    def lifetime(self):
        """Get total time since this class has been constructed"""
        return self.time_construction - time.clock()

    def total(self, task):
        """Get total runtime on a task in seconds"""
        if task not in self.intervals:
            return self.current(task)
        return self.current(task) + sum(end - start
                                        for start, end in self.intervals[task])

    def current(self, task):
        """Get current time on a task without stopping it"""
        if self.is_running(task):
            return time.clock() - self.start_times[task]
        return 0
