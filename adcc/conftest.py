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
import os

import pytest


def update_testdata():
    import subprocess

    testdata_dir = os.path.join(os.path.dirname(__file__), "testdata")
    cmd = [testdata_dir + "/0_download_testdata.sh"]
    if pytest.config.option.mode == "full":
        cmd.append("--full")
    subprocess.check_call(cmd)


def pytest_addoption(parser):
    parser.addoption(
        "--mode", default="fast", choices=["fast", "full"],
        help="Mode for testing (fast or full)"
    )
    parser.addoption(
        "--skip-update", default=False, action="store_true",
        help="Skip updating testdata"
    )


def pytest_collection_modifyitems(config, items):
    slowcases = ["h2o_def2tzvp", "h2o_ccpvdz", "cn_ccpvdz", "h2s_6311g"]

    if config.getoption("mode") == "fast":
        skip_slow = pytest.mark.skip(reason="need '--mode full' option to run.")
        for item in items:
            if any(name in kw for kw in item.keywords for name in slowcases):
                item.add_marker(skip_slow)


def pytest_runtestloop(session):
    if not pytest.config.option.skip_update:
        update_testdata()
