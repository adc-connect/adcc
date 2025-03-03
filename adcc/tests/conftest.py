from . import testcases

from pathlib import Path
import os
import pytest
import subprocess


_testdata_dirname = "data"


def update_testdata(session):
    testdata_dir = Path(__file__).resolve().parent / _testdata_dirname
    cmd = [f"{testdata_dir}/update_testdata.sh"]
    if session.config.option.mode == "full":
        cmd.append("--full")
    subprocess.check_call(cmd)


#
# Pytest Hooks
#

def pytest_addoption(parser: pytest.Parser) -> None:
    # could also use pytest.mark.fast/slow
    # -> pytest -m fast/slow to only run the marked tests
    parser.addoption(
        "--mode", default="fast", choices=["fast", "full"],
        help="Mode for testing (fast or full)"
    )
    parser.addoption(
        "--skip-update", default=False, action="store_true",
        help="Skip updating testdata"
    )
    parser.addoption(
        "--allocator", default="standard", choices=["standard", "libxm"],
        help="Allocator to use for the tests"
    )


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    # Called after collection has been performed.
    # May filter or re-order the items in-place.
    # -> Skip all "slow" tests if not running in full mode
    if config.getoption("mode") == "fast":
        slow_cases = [case.file_name for case in testcases.available
                      if case.only_full_mode]
        skip_slow = pytest.mark.skip(reason="need '--mode full' option to run.")
        for item in items:
            if any(name in kw for kw in item.keywords for name in slow_cases):
                item.add_marker(skip_slow)


def pytest_collection(session):
    # Perform the collection phase for the given session.
    if not session.config.option.skip_update:
        update_testdata(session)


def pytest_runtestloop(session):
    # Perform the main runtest loop (after collection finished).
    if os.environ.get("CI", "false") == "true":
        import adcc

        # use more moderate thread setup in continuous integration environment
        print("Detected continuous integration session")
        adcc.set_n_threads(2)
    if session.config.option.allocator != "standard":
        import adcc

        allocator = session.config.option.allocator
        adcc.memory_pool.initialise(allocator=allocator)
        print(f"Using allocator: {allocator}")
