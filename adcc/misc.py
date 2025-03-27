#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
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
import warnings
import numpy as np
import inspect
from functools import wraps
from packaging.version import parse

from .timings import Timer


def cached_property(f):
    """
    Decorator for a cached property. From
    https://stackoverflow.com/questions/6428723/python-are-property-fields-being-cached-automatically
    """
    def get(self):
        try:
            return self._property_cache[f]
        except AttributeError:
            self._property_cache = {}
            x = self._property_cache[f] = f(self)
            return x
        except KeyError:
            x = self._property_cache[f] = f(self)
            return x

    get.__doc__ = f.__doc__
    # TODO: find more elegant solution for this
    if hasattr(f, "__excitation_property"):
        get.__excitation_property = f.__excitation_property

    return property(get)


def cached_member_function(timer: str = "timer",
                           separate_timings_by_args: bool = True):
    """
    Decorates a member function being called with
    one or more arguments and stores the results
    in field `_function_cache` of the class instance.
    If the class has a timer (defined under the provided name)
    the timings of the (first) call will be measured.

    Parameters
    ----------
    timer: str, optional
        Name of the member variable the :class:`Timer` instance can be found on the
        class instance (default: 'timer'). If the timer is not found no timings
        are measured.
    separate_timings_by_args: bool, optional
        If set the arguments passed to the decorated functions will be included
        in the key under which the timings are stored, i.e., a distinct timer task
        will be generated for each set of arguments. (default: True)
    """
    def decorator(function):
        fname = function.__name__

        # get the function signature and ensure that we don't have any
        # keyword only arguments:
        # func(..., *, kwarg=None, ...) or func(..., **kwargs).
        # If we want to support them we need to add them in a well defined
        # order to the cache key (sort them by name)
        func_signature = inspect.signature(function)
        bad_arg_types = (
            inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD
        )
        if any(arg.kind in bad_arg_types for arg in
               func_signature.parameters.values()):
            raise ValueError("Member functions with keyword only arguments can not "
                             "be wrapped with the cached_member_function.")

        @wraps(function)
        def wrapper(self, *args, **kwargs):
            # convert all arguments to poisitonal arguments and add default
            # arguments for not provided arguments
            bound_args = func_signature.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            assert not bound_args.kwargs
            args = bound_args.args[1:]  # remove self from args

            try:
                fun_cache = self._function_cache[fname]
            except AttributeError:
                self._function_cache = {}
                fun_cache = self._function_cache[fname] = {}
            except KeyError:
                fun_cache = self._function_cache[fname] = {}

            try:
                return fun_cache[args]
            except KeyError:
                # Record with a timer if possible
                instance_timer = getattr(self, timer, None)
                if isinstance(instance_timer, Timer):
                    timer_task = fname
                    if separate_timings_by_args:
                        descr = '_'.join([str(a) for a in args])
                        timer_task += f"/{descr}"

                    with instance_timer.record(timer_task):
                        # try to evaluate the result if possible
                        result = function(self, *args)
                        if hasattr(result, "evaluate"):
                            result = result.evaluate()
                        fun_cache[args] = result
                else:
                    result = function(self, *args)
                    if hasattr(result, "evaluate"):
                        result = result.evaluate()
                    fun_cache[args] = result
                return result
        return wrapper
    return decorator


def is_module_available(module, min_version=None):
    """Check using importlib if a module is available."""
    import importlib

    try:
        mod = importlib.import_module(module)
    except ImportError:
        return False

    if not min_version:  # No version check
        return True

    if not hasattr(mod, "__version__"):
        warnings.warn(
            f"Could not check module {module} minimal version, "
            "since __version__ tag not found. Proceeding anyway."
        )
        return True

    if parse(mod.__version__) < parse(min_version):
        warnings.warn(
            f"Found module {module}, but its version {mod.__version__} is below "
            f"the least required (== {min_version}). This module will be ignored."
        )
        return False
    return True


def requires_module(name, min_version=None):
    """
    Decorator to check if the module 'name' is available,
    throw ModuleNotFoundError on call if not.
    """
    def inner(function):
        def wrapper(*args, **kwargs):
            fname = function.__name__
            if not is_module_available(name, min_version):
                raise ModuleNotFoundError(
                    f"Function '{fname}' needs module {name}, but it was "
                    f"not found. Solve by running 'pip install {name}' or "
                    f"'conda install {name}' on your system."
                )
            return function(*args, **kwargs)
        wrapper.__doc__ = function.__doc__
        return wrapper
    return inner


def assert_allclose_signfix(actual, desired, atol=0, **kwargs):
    """
    Call assert_allclose, but beforehand normalise the sign
    of the involved arrays (i.e. the two arrays may differ
    up to a sign factor of -1)
    """
    actual, desired = normalise_sign(actual, desired, atol=atol)
    np.testing.assert_allclose(actual, desired, atol=atol, **kwargs)


def normalise_sign(*items, atol=0):
    """
    Normalise the sign of a list of numpy arrays
    """
    def sign(item):
        flat = np.ravel(item)
        flat = flat[np.abs(flat) > atol]
        if flat.size == 0:
            return 1
        else:
            return np.sign(flat[0])
    desired_sign = sign(items[0])
    return tuple(desired_sign / sign(item) * item for item in items)
