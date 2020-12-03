#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2019 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
import sys
import time
import numpy as np

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


def strtime(span, colour=False):
    """
    Return a moderately long string, providing a human-interpretable
    representation of the provided timespan
    """
    if colour:
        cms = "\033[0m"
        cs = "\033[38;5;246m"
        cm = "\033[38;5;239m"
        ch = "\033[38;5;250m"
        cd = "\033[38;5;242m"
        cr = "\033[0m"
    else:
        cr = cd = ch = cm = cs = cms = ""

    if span < 1:
        return cms + "{:6.3f}ms".format(span * 1000) + cr
    if span < 120:
        full = int(span)
        return (cs + "{:3d}s ".format(full) + cms
                + "{:3d}ms".format(int((span - full) * 1000)) + cr)
    if span < 3600:
        full = int(span / 60)
        return (cm + "{:2d}m ".format(full) + cs
                + "{:2d}s".format(int(span - full * 60)) + cr)
    if span < 86400:
        full = int(span / 3600)
        return (ch + "{:2d}h ".format(full) + cm
                + "{:2d}m".format(int(span / 60 - full * 60)) + cr)
    else:
        full = int(span / 86400)
        return (cd + "{:3d}d ".format(full) + ch
                + "{:2d}h".format(int(span / 3600 - full * 24)) + cr)


class Timer:
    # TODO More flexible: Time on a subtree
    # TODO Describe function to print a nice table
    def __init__(self):
        self.time_construction = time.perf_counter()
        self.raw_data = {}  # Raw data of time intervals [start, end]
        self.start_times = {}

    def attach(self, other, subtree=""):
        """
        Attach the timing results from another timer,
        i.e. merge both timers together.
        """
        for k, v in other.raw_data.items():
            kfull = join(subtree, k)
            if kfull not in self.raw_data:
                self.raw_data[kfull] = []
            self.raw_data[kfull].extend(v)

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
            if task in self.raw_data:
                self.raw_data[task].append((start, now))
            else:
                self.raw_data[task] = [(start, now)]
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
        all_tasks.update(set(self.raw_data.keys()))
        return sorted(list(all_tasks))

    @property
    def lifetime(self):
        """Get total time since this class has been constructed"""
        return time.perf_counter() - self.time_construction

    def intervals(self, task):
        """Get all time intervals recorded for a particular task"""
        if task not in self.raw_data:
            if task not in self.start_times:
                raise ValueError("Unknown task: " + task)
            return self.current(task)
        else:
            intervals = [end - start for start, end in self.raw_data[task]]
            if task in self.start_times:
                intervals.append(time.perf_counter() - self.start_times[task])
            return np.array(intervals)

    def total(self, task):
        """Get total runtime on a task in seconds"""
        return np.sum(self.intervals(task))

    def best(self, task):
        """Get the best time of all the intervals recorded for a task"""
        return np.min(self.intervals(task))

    def median(self, task):
        return np.median(self.intervals(task))

    def average(self, task):
        return np.average(self.intervals(task))

    def current(self, task):
        """Get current time on a task without stopping it"""
        if self.is_running(task):
            return time.perf_counter() - self.start_times[task]
        else:
            raise ValueError("Task not currently running: " + task)

    def describe(self, colour=sys.stdout.isatty()):
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
            text += fmt.format(key, strtime(self.total(key), colour=colour))
        return text

    def _repr_pretty_(self, pp, cycle):
        if cycle:
            pp.text("Timer()")
        else:
            pp.text(self.describe())


Timer.start = Timer.restart


def timed_call(f):
    """
    Decorator to automatically time function calls.
    The timer object is available under the function attribute _timer
    """
    def decorated(*args, **kwargs):
        if not hasattr(decorated, "_timer"):
            setattr(decorated, "_timer", Timer())
        with getattr(decorated, "_timer").record(f.__name__):
            return f(*args, **kwargs)
    decorated.__doc__ = f.__doc__
    return decorated


def timed_member_call(timer="timer"):
    """
    Decorator to automatically time calls to instance member functions.
    The name of the instance attribute where timings are stored is the
    timer argument to this function.
    """
    def decorator(f):
        def wrapped(self, *args, **kwargs):
            if not hasattr(self, timer):
                setattr(self, timer, Timer())
            with getattr(self, timer).record(f.__name__):
                return f(self, *args, **kwargs)
        wrapped.__doc__ = f.__doc__
        return wrapped
    return decorator
