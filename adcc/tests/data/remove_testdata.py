from pathlib import Path
import argparse
import h5py
import re


def remove(file: Path, cases: list[str] = None, density_orders: list[str] = None):
    """
    Removes test data for the given reference cases and density orders in the
    given test data (hdf5) file and deletes the file if all data is removed.
    """
    # hfdata files are independent of case and density_order
    # -> can only delete all content and remove the file
    if "hfdata" in file.name:
        file.unlink()
        return
    # deal with hfimport/mpdata and adc data files
    is_empty = False
    with h5py.File(file, "r+") as hdf5_file:  # r+: read/write, file must exist
        cases_to_remove = []
        for case, case_data in hdf5_file.items():
            if cases is not None and case not in cases:  # keep the case
                continue
            # mpdata and hfimport are independent of the density order
            # if density_orders is not given we remove the data for all
            # density_orders
            if "mpdata" in file.name or "hfimport" in file.name or \
                    density_orders is None:
                cases_to_remove.append(case)
                continue
            # at this point we have to have a file with adc reference data
            assert re.search(r"adc[0-9]", file.name)
            density_orders_to_remove = [
                dens_oder for dens_oder in case_data.keys()
                if density_orders is None or dens_oder in density_orders
            ]
            for density_order in density_orders_to_remove:
                del case_data[density_order]

            if not case_data.keys():
                cases_to_remove.append(case)
        # delete the data and check if the file is empty afterwards
        for case in cases_to_remove:
            del hdf5_file[case]
        if not hdf5_file.keys():
            is_empty = True
    if is_empty:
        file.unlink()


def parse_cmdline() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="remove test data",
        description=("Helper script to remove specific test data to allow their "
                     "regeneration."),
    )
    parser.add_argument(
        "datafiles", type=str, nargs="+",
        help="The data (hdf5) files in which to remove test data."
    )
    parser.add_argument(
        "-c", "--cases", type=str, nargs="*", default=None,
        help=("The reference cases for which to remove test data. If not given "
              "data for all reference cases will be removed (you can also "
              "remove the entire files in that case).")
    )
    parser.add_argument(
        "--density-orders", dest="density_orders", type=str, nargs="*",
        default=None,
        help=("The gs_density_orders for which to remove test data. If not given "
              "the data will be removed for all density orders (if applicable).")
    )
    return parser.parse_args()


def main():
    opts = parse_cmdline()

    for file in opts.datafiles:
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"Could not find file {file}.")
        if not file.is_file():
            raise ValueError(f"Can only remove data from files. Got {file}.")
        if file.suffix != ".hdf5":
            raise ValueError(f"Can only remove data from hdf5 files. Got {file}.")

        remove(file, opts.cases, opts.density_orders)


if __name__ == "__main__":
    main()
