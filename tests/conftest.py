from . import testcases

import pytest


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
    if config.getoption("mode") == "fast":
        slow_cases = [case.file_name for case in testcases.available
                      if case.only_full_mode]
        skip_slow = pytest.mark.skip(reason="need '--mode full' option to run.")
        for item in items:
            if any(name in kw for kw in item.keywords for name in slow_cases):
                item.add_marker(skip_slow)
