import numpy as np

from matplotlib import pyplot as plt


def shape_gaussian(x, mean, stddev):
    fac = 1 / np.sqrt(2 * np.pi * stddev**2)
    return fac * np.exp(-(x - mean)**2 / (2 * stddev**2))


def shape_lorentzian(x, center, gamma):
    return gamma / ((x - center)**2 + gamma**2) / np.pi


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

    def broaden_lines(self, shape="lorentzian", width=None):
        """Apply broadening to the current spectral data and
        return the broadened spectrum.

        Parameters
        ----------
        shape : str, optional
            The shape of the broadening to use (lorentzian or gaussian),
            by default lorentzian broadening is used
        width : float, optional
            The width to use for the broadening, by default a sensible
            value is chosen.
        """
        if shape == "gaussian":
            if width is None:
                width = 0.35

            def gauss(x, mean):
                return shape_gaussian(x, mean, width)
            return self.broaden_lines(shape=gauss)
        elif shape == "lorentzian":
            if width is None:
                width = 0.3

            def lorentzian(x, center):
                return shape_lorentzian(x, center, width)
            return self.broaden_lines(shape=lorentzian)
        elif isinstance(shape, str):
            raise ValueError("Unknown broadening type: " + shape)

        xmin = np.min(self.x)
        xmax = np.max(self.x)
        xextra = (xmax - xmin) / 10
        x = np.linspace(xmin - xextra, xmax + xextra, int(100 * (xmax - xmin)))

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
        if style == "discrete":
            plt.vlines(self.x, 0, self.y, linestyle="dashed", color="grey",
                       linewidth=1)
            p = plt.plot(self.x, self.y, "x", *args, **kwargs)
        elif style == "continuous":
            p = plt.plot(self.x, self.y, "-", *args, **kwargs)
        elif style is None:
            p = plt.plot(self.x, self.y, *args, **kwargs)
        else:
            raise ValueError("Unknown style: " + str(style))
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        return p
