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

from io import StringIO


def load_libtensor(fle):
    """
    Load a libtensor text file and return a numpy array
    representing the contained data.
    """
    if isinstance(fle, str):
        with open(fle, "r") as f:
            return load_libtensor(f)

    # First line is the dimension and the shape
    v = fle.readline().split()

    try:
        dim = int(v[0])
    except ValueError:
        raise ValueError("Could not parse libtensor file: "
                         "Could not read dimension: " + v[0])

    if len(v) != dim + 1:
        raise ValueError("Could not parse libtensor file: "
                         "Length of dimensionality string (== " + str(len(v))
                         + ") and expected number of dimensions (== " + str(dim)
                         + ") does not agree.")

    shape = []
    for i in range(dim):
        try:
            shape.append(int(v[i + 1]))
        except ValueError:
            raise ValueError("Could not parse libtensor file: "
                             "Could not parse the " + str(i) + "th element of "
                             "the dimensionality string.")
    shape = tuple(shape)

    # Construct a string where all fields are on an individual line
    translated = "\n".join(map(lambda x: "\n".join(x.split()), fle.readlines()))

    # Read that into numpy and return
    return np.loadtxt(StringIO(translated)).reshape(shape)


if __name__ == "__main__":
    # Both the string and the reference are extracted from
    # an actual unit test inside libtensor
    tensor_stream = StringIO(
        """4 2 3 2 3
           0.666739916801930 0.789241932186695 0.354933631458927
           0.148729512658850 0.315136114580255 0.917841851387831
           0.745541362541637 0.636638423369451 0.131076619411402
           0.223687650737997 0.778043211995630 0.743877360414235
           0.876509816225663 0.437432365273278 0.551743844733686
           0.756052944470046 0.977138144356729 0.590605047515037
           0.986884746662248 0.241478511671335 0.812884764694630
           0.368243749194956 0.986671562269798 0.269242285834029
           0.698482607691375 0.633621372857885 0.369196404658801
           0.973523775057021 0.977914983508967 0.173239489854350
           0.307523274999390 0.350790596757097 0.217292801077974
           0.036913933392764 0.697086228627064 0.695325836804376
        """)

    tensor = load_libtensor(tensor_stream)
    assert tensor[0, 0, 0, 0] == 0.666739916801930
    assert tensor[0, 0, 0, 1] == 0.789241932186695
    assert tensor[0, 0, 0, 2] == 0.354933631458927
    assert tensor[0, 0, 1, 0] == 0.148729512658850
    assert tensor[0, 0, 1, 1] == 0.315136114580255
    assert tensor[0, 0, 1, 2] == 0.917841851387831
    assert tensor[0, 1, 0, 0] == 0.745541362541637
    assert tensor[0, 1, 0, 1] == 0.636638423369451
    assert tensor[0, 1, 0, 2] == 0.131076619411402
    assert tensor[0, 1, 1, 0] == 0.223687650737997
    assert tensor[0, 1, 1, 1] == 0.778043211995630
    assert tensor[0, 1, 1, 2] == 0.743877360414235
    assert tensor[0, 2, 0, 0] == 0.876509816225663
    assert tensor[0, 2, 0, 1] == 0.437432365273278
    assert tensor[0, 2, 0, 2] == 0.551743844733686
    assert tensor[0, 2, 1, 0] == 0.756052944470046
    assert tensor[0, 2, 1, 1] == 0.977138144356729
    assert tensor[0, 2, 1, 2] == 0.590605047515037
    assert tensor[1, 0, 0, 0] == 0.986884746662248
    assert tensor[1, 0, 0, 1] == 0.241478511671335
    assert tensor[1, 0, 0, 2] == 0.812884764694630
    assert tensor[1, 0, 1, 0] == 0.368243749194956
    assert tensor[1, 0, 1, 1] == 0.986671562269798
    assert tensor[1, 0, 1, 2] == 0.269242285834029
    assert tensor[1, 1, 0, 0] == 0.698482607691375
    assert tensor[1, 1, 0, 1] == 0.633621372857885
    assert tensor[1, 1, 0, 2] == 0.369196404658801
    assert tensor[1, 1, 1, 0] == 0.973523775057021
    assert tensor[1, 1, 1, 1] == 0.977914983508967
    assert tensor[1, 1, 1, 2] == 0.173239489854350
    assert tensor[1, 2, 0, 0] == 0.307523274999390
    assert tensor[1, 2, 0, 1] == 0.350790596757097
    assert tensor[1, 2, 0, 2] == 0.217292801077974
    assert tensor[1, 2, 1, 0] == 0.036913933392764
    assert tensor[1, 2, 1, 1] == 0.697086228627064
    assert tensor[1, 2, 1, 2] == 0.695325836804376
