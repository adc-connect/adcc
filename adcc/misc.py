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
import numpy as np


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

    return property(get)


def expand_test_templates(arguments, template_prefix="template_"):
    """
    Expand the test templates of the class cls using the arguments
    provided as a list of tuples to this function
    """
    parsed_args = []
    for args in arguments:
        if isinstance(args, tuple):
            parsed_args.append(args)
        else:
            parsed_args.append((args, ))

    def inner_decorator(cls):
        for fctn in dir(cls):
            if not fctn.startswith(template_prefix):
                continue
            basename = fctn[len(template_prefix):]
            for args in parsed_args:
                newname = "test_" + basename + "_"
                newname += "_".join(str(a) for a in args)

                # Call the actual function by capturing the
                # fctn and args arguments by-value using the
                # trick of supplying them as default arguments
                # (which are evaluated at definition-time)
                def caller(self, fctn=fctn, args=args):
                    return getattr(self, fctn)(*args)
                setattr(cls, newname, caller)
        return cls
    return inner_decorator


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
