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
import os
from dataclasses import dataclass


@dataclass
class Molecule:
    name: str
    charge: int = 0
    multiplicity: int = 1
    core_orbitals: int = 1

    @property
    def xyz(self):
        return xyz[self.name]


molecules = [
    Molecule("h2o", 0, 1),
    Molecule("h2s", 0, 1),
    # Molecule("cn", 0, 2),
    # "ch2nh2",
    # Molecule("hf", 0, 1),
    # Molecule("formaldehyde", 0, 1),
]


# all coordinates in Bohr
xyz = {
    "h2o": """
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    """,
    #
    "h2s": """
    S  -0.38539679062   0 -0.27282082253
    H  -0.0074283962687 0  2.2149138578
    H   2.0860198029    0 -0.74589639249
    """,
    #
    "cn": """
    C 0 0 0
    N 0 0 2.2143810738114829
    """,
    #
    "hf": """
    H 0 0 0
    F 0 0 2.5
    """,
    #
    "ch2nh2": """
    C -1.043771327642266  0.9031379094521343 -0.0433881118200138
    N  1.356218645077853 -0.0415928720016770  0.9214682528604154
    H -1.624635343811075  2.6013402912925274  1.0436579440747924
    H -2.522633198204392 -0.5697335292951204  0.1723619198215792
    H  2.681464678974086  1.3903093043650074  0.6074335654801934
    H  1.838098806841944 -1.5878801706882844 -0.2108367437177239
    """,
    #
    "r2methyloxirane": """
    O        0.0000000000      0.0000000000      0.0000000000
    C        2.7197505315      0.0000000000      0.0000000000
    H        3.5183865867      0.0000000000      1.8891781049
    C        1.3172146985     -2.2531204594     -0.7737861566
    H        1.1322105108     -3.8348590209      0.5099803895
    H        1.1680586154     -2.6812379131     -2.7700739383
    C        3.9569164314      1.7018306568     -1.8913890844
    H        3.0053826354      1.5478557721     -3.7088898774
    H        3.8622600497      3.6616656215     -1.2676471822
    H        5.9395792877      1.1934754318     -2.1292489119
    """,  # (R)-2-Methyloxirane
    #
    "formaldehyde": """
    C 2.0092420208996 3.8300915804899 0.8199294419789
    O 2.1078857690998 2.0406638776593 2.1812021228452
    H 2.0682421748693 5.7438044586615 1.5798996515014
    H 1.8588483602149 3.6361694243085 -1.2192956060942
    """
}

_thisdir = os.path.dirname(__file__)
pe_potentials = {
    # Formaldehyde + 6 Water molecules
    "fa_6w": os.path.join(_thisdir, "potentials/fa_6w.pot"),
}
