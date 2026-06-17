#!/usr/bin/env python3
"""Geometry optimisation with adcc nuclear gradients and geomeTRIC.

This example uses the geomeTRIC custom-engine protocol.  geomeTRIC is optional;
install it separately before running this script.
"""

import adcc
from pyscf import gto


mol = gto.M(
    atom="O 0 0 0; H 0 0 1.795239827225189; H 1.693194615993441 0 -0.599043184453037",
    basis="sto-3g",
    unit="Bohr",
    verbose=0,
)

scanner = adcc.nuclear_gradient_scanner(
    mol_template=mol,
    method="mp2",
    scf_conv_tol=1e-11,
    scf_conv_tol_grad=1e-9,
)

try:
    from geometric.engine import Engine
except ImportError as exc:
    raise SystemExit("Install geomeTRIC to run this example: pip install geometric") from exc


class AdccEngine(Engine):
    def calc_new(self, coords, dirname):  # noqa: D102 - geomeTRIC protocol
        return scanner.calc_new(coords)


# For a complete script one also needs to create geomeTRIC's Molecule,
# InternalCoordinates and OptParams objects (or use geomeTRIC's Python helpers)
# and pass this engine to run_optimizer.  The important adcc-specific part is
# the scanner above: it runs PySCF at each geometry, hands the converged SCF
# result to adcc, evaluates the MP2 gradient, and returns geomeTRIC's
# {"energy", "gradient"} dictionary.
print("Initial energy / gradient:")
print(scanner.calc_new(mol.atom_coords().ravel()))
