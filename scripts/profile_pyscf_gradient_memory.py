#!/usr/bin/env python3
"""
Playground script for profiling PySCF/adcc gradient memory with memray.

Example commands:

    python scripts/profile_pyscf_gradient_memory.py --molecule benzene \
        --basis sto-3g --tei-contraction direct --memray-output direct.bin
    python -m memray flamegraph direct.bin

    python scripts/profile_pyscf_gradient_memory.py --molecule benzene \
        --basis sto-3g --tei-contraction full_ao --memray-output full_ao.bin
    python -m memray summary full_ao.bin

The script uses memray's in-process tracker when ``--memray-output`` is given,
so setup/import/SCF allocations are excluded and only the selected gradient
calculation is tracked.  Use ``--compare-to full_ao`` only for tiny cases.  It
runs two gradients in the same process, which is convenient for algebra checks
but not for clean memory measurements.
"""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from pyscf import gto, scf

import adcc


WATER_GEOMETRY = """
O  0.0000000000  0.0000000000  0.0000000000
H  0.0000000000 -0.7570000000  0.5870000000
H  0.0000000000  0.7570000000  0.5870000000
"""


BENZENE_GEOMETRY = """
C  1.3970000000  0.0000000000  0.0000000000
C  0.6985000000  1.2098378780  0.0000000000
C -0.6985000000  1.2098378780  0.0000000000
C -1.3970000000  0.0000000000  0.0000000000
C -0.6985000000 -1.2098378780  0.0000000000
C  0.6985000000 -1.2098378780  0.0000000000
H  2.4810000000  0.0000000000  0.0000000000
H  1.2405000000  2.1484096046  0.0000000000
H -1.2405000000  2.1484096046  0.0000000000
H -2.4810000000  0.0000000000  0.0000000000
H -1.2405000000 -2.1484096046  0.0000000000
H  1.2405000000 -2.1484096046  0.0000000000
"""


GEOMETRIES = {
    "benzene": BENZENE_GEOMETRY,
    "water": WATER_GEOMETRY,
}


def build_hf(molecule, basis, conv_tol):
    mol = gto.M(atom=GEOMETRIES[molecule], basis=basis, unit="Angstrom", verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 10 * conv_tol
    return mf.run()


def gradient_target(scfres, method, conv_tol, n_singlets, state_index):
    if method == "mp2":
        return adcc.LazyMp(adcc.ReferenceState(scfres))
    states = adcc.run_adc(
        scfres, method=method, n_singlets=n_singlets, conv_tol=conv_tol
    )
    return states.excitations[state_index]


def compute_gradient(target, args, tei_contraction):
    start = time.perf_counter()
    grad = adcc.nuclear_gradient(
        target,
        conv_tol=args.conv_tol,
        tei_contraction=tei_contraction,
        tei_shell_chunk_size=args.shell_chunk_size,
        tei_pair_chunk_size=args.pair_chunk_size,
        tei_pair_density_storage=args.pair_density_storage,
    )
    elapsed = time.perf_counter() - start
    return grad, elapsed


def print_result(label, grad, elapsed):
    print(f"\n=== {label} ===")
    print(f"Elapsed wall time: {elapsed:.3f} s")
    print("Total gradient / Hartree Bohr^-1:")
    print(np.array2string(grad.total, precision=12, suppress_small=False))
    print("Timer:")
    print(grad.timer.describe(colour=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule", default="benzene", choices=sorted(GEOMETRIES),
        help="Benchmark molecule. Benzene is the default to make memory visible.",
    )
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument(
        "--method", default="mp2",
        help="mp2 or an ADC method accepted by adcc.run_adc, e.g. adc2",
    )
    parser.add_argument("--conv-tol", type=float, default=1e-9)
    parser.add_argument("--n-singlets", type=int, default=1)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument(
        "--tei-contraction", default="direct",
        choices=["auto", "direct", "shell_batched", "full_ao"],
    )
    parser.add_argument("--shell-chunk-size", type=int, default=1)
    parser.add_argument("--pair-chunk-size", type=int, default=None)
    parser.add_argument(
        "--pair-density-storage", default="memory",
        choices=["memory", "hdf5", "outcore"],
    )
    parser.add_argument(
        "--compare-to", choices=["direct", "shell_batched", "full_ao"],
        help="Optional correctness comparison. Not recommended for memray runs.",
    )
    parser.add_argument(
        "--memray-output",
        help="Write a memray profile for the selected gradient computation only.",
    )
    parser.add_argument(
        "--memray-native", action="store_true",
        help="Collect native traces for the in-process memray tracker.",
    )
    args = parser.parse_args()

    print("Building PySCF RHF reference")
    print(
        f"molecule={args.molecule} basis={args.basis} "
        f"conv_tol={args.conv_tol}"
    )
    scfres = build_hf(args.molecule, args.basis, args.conv_tol)
    target = gradient_target(
        scfres, args.method, args.conv_tol, args.n_singlets, args.state_index
    )

    if args.memray_output:
        import memray
        print(f"Tracking gradient allocations with memray: {args.memray_output}")
        with memray.Tracker(
                args.memray_output, native_traces=args.memray_native):
            grad, elapsed = compute_gradient(target, args, args.tei_contraction)
    else:
        grad, elapsed = compute_gradient(target, args, args.tei_contraction)
    print_result(args.tei_contraction, grad, elapsed)

    if args.compare_to:
        ref, ref_elapsed = compute_gradient(target, args, args.compare_to)
        print_result(args.compare_to, ref, ref_elapsed)
        print("\nMax |difference|:", np.max(np.abs(grad.total - ref.total)))


if __name__ == "__main__":
    main()
