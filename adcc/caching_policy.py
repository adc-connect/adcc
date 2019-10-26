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
from libadcc import CachingPolicy_i


class CacheAllPolicy(CachingPolicy_i):
    """
    Policy which caches everything. Useful for testing to
    speed things up.
    """
    def __init__(self):
        CachingPolicy_i.__init__(self)

    def should_cache(self, tensor_label, tensor_space,
                     leading_order_contraction):
        return True


class DefaultCachingPolicy(CachingPolicy_i):
    def __init__(self):
        CachingPolicy_i.__init__(self)

    def should_cache(self, tensor_label, tensor_space,
                     leading_order_contraction):
        # For now be stupid and store everything by default
        return True


class GatherStatisticsPolicy(CachingPolicy_i):
    """
    This caching policy advises against caching any data,
    it does, however, keep track of the number of times the
    caching for a particular object has been requested and thus
    allows to gain some insight on the helpfulness of particular
    cachings.
    """
    def __init__(self):
        CachingPolicy_i.__init__(self)
        self.call_count = {}

    def should_cache(self, label, space, contraction):
        key = (label, space, contraction)
        value = self.call_count.get(key, 0)
        self.call_count[key] = value + 1
        return False

    def _repr_pretty_(self, pp, cycle):
        if not self.call_count or cycle:
            pp.text("GatherStatisticsPolicy()")
            return

        maxlal = max(len(k[0]) for k in self.call_count)
        maxsp = max(len(k[1]) for k in self.call_count)
        maxcon = max(len(k[2]) for k in self.call_count)
        maxcon = max(maxcon, 12)

        fmt = (
            "| {:" + str(maxlal) + "} {:" + str(maxsp) + "}"
            + "    {:" + str(maxcon) + "}  {:6d} |\n"
        )
        maxbody = 0
        body = ""
        for k, v in self.call_count.items():
            txt = fmt.format(k[0], k[1], k[2], v)
            body += txt
            maxbody = max(maxbody, len(txt))

        cutline = "+" + (maxbody - 3) * "-" + "+"
        title = ("|{:^" + str(maxbody - 3) + "s}|"
                 "\n").format("Tensor caching statistics")
        header = ("| {:^" + str(maxlal + maxsp + 1) + "}    {:^" + str(maxcon)
                  + "}  {:>6s} |\n").format("Tensor", "contraction", "count")
        pp.text(cutline + "\n" + title + cutline + "\n"
                + header + body + cutline)


# TODO Ideas:
#        - Cache on the n-th use of an object
#        - Do not cache the tensors needed to compute pia / pib,
#          because they will always be needed only once right now
#          (even in ADC(3))
