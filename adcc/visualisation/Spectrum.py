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
import numpy as np

from . import shapefctns
from ..misc import requires_module


class Spectrum:
    """
    Class for representing arbitrary spectra. This design is strongly
    inspired by the approach used by pymatgen.
    """

    # TODO Ideas:
    #   - normalise()   normalise spectrum by scaling it
    #   - shift()       shift spectrum such that max peak is at a
    #                   particular posn
    #   - interpolate_value(self, x, order=1):
    #                   interpolate value by linear / polynomial interpolation
    #                   between existing points
    #  support element-wise multiplication, division, addition /
    #                   subtraction of spectra ?

    def __init__(self, x, y, *args, **kwargs):
        """Pass spectrum data to initialise the class.

        To allow the copy and other functions to work properly, all arguments
        and keywords arguments from implementing classes should be passed here.

        Parameters
        ----------
        x : ndarray
            Data on the first axis
        y : ndarray
            Data on the second axis
        args
            Arguments passed to subclass upon construction
        kwargs
            Keyword arguments passed to subclass upon construction.
        """
        self.x = np.array(x).flatten()
        self.y = np.array(y).flatten()
        self.xlabel = "x"
        self.ylabel = "y"
        if self.x.size != self.y.size:
            raise ValueError("Sizes of x and y mismatch: {} versus {}."
                             "".format(self.x.size, self.y.size))
        self._args = args
        self._kwargs = kwargs

    def broaden_lines(self, width=None, shape="lorentzian", xmin=None, xmax=None):
        """Apply broadening to the current spectral data and
        return the broadened spectrum.

        Parameters
        ----------
        shape : str or callable, optional
            The shape of the broadening to use (lorentzian or gaussian),
            by default lorentzian broadening is used. This can be a callable
            to directly specify the function with which each line of the
            spectrum is convoluted.
        width : float, optional
            The width to use for the broadening (stddev for the gaussian,
            gamma parameter for the lorentzian).
            Optional if shape is a callable.
        xmin : float, optional
            Explicitly set the minimum value of the x-axis for broadening
        xmax : float, optional
            Explicitly set the maximum value of the x-axis for broadening
        """
        if not callable(shape) and width is None:
            raise ValueError("If shape is not a callable, the width parameter "
                             "is required")

        if not callable(shape):
            if not hasattr(shapefctns, shape):
                raise ValueError("Unknown broadening function: " + shape)
            shapefctn = getattr(shapefctns, shape)

            def shape(x, x0):
                # Empirical scaling factor to make the envelope look nice
                scale = width / 0.272
                return scale * shapefctn(x, x0, width)

        if xmin is None:
            xmin = np.min(self.x)
        if xmax is None:
            xmax = np.max(self.x)
        xextra = (xmax - xmin) / 10
        n_points = min(5000, max(500, int(1000 * (xmax - xmin))))
        x = np.linspace(xmin - xextra, xmax + xextra, n_points)

        y = 0
        for center, value in zip(self.x, self.y):
            y += value * shape(x, center)

        cpy = self.copy()
        cpy.x = x
        cpy.y = y
        return cpy

    def copy(self):
        """Return a consistent copy of the object."""
        cpy = self.__class__(self.x, self.y, *self._args, **self._kwargs)
        cpy.xlabel = self.xlabel
        cpy.ylabel = self.ylabel
        return cpy

    @requires_module("matplotlib")
    def plot(self, *args, style=None, **kwargs):
        """Plot the Spectrum represented by this class.

        Parameters not listed below are passed to the matplotlib plot function.

        Parameters
        ----------
        style : str, optional
            Use some default setup of matplotlib parameters for certain
            types of spectra commonly plotted. Valid are "discrete" and
            "continuous". By default no special style is chosen.
        """
        from matplotlib import pyplot as plt
        if style == "discrete":
            p = plt.plot(self.x, self.y, "x", *args, **kwargs)
            plt.vlines(self.x, 0, self.y, linestyle="dashed",
                       color=p[0].get_color(), linewidth=1)
        elif style == "continuous":
            p = plt.plot(self.x, self.y, "-", *args, **kwargs)
        elif style is None:
            p = plt.plot(self.x, self.y, *args, **kwargs)
        else:
            raise ValueError("Unknown style: " + str(style))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        # if we have negative y-values (e.g., rotatory strengths),
        # draw y = 0 as an extra line for clarity
        if np.any(self.y < 0.0):
            plt.axhline(0.0, color='gray', lw=0.5)
        return p
