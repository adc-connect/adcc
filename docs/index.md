# adcc:  Seamlessly connect your program to ADC

```note::  This documentation page is still under construction.

```

**ADC connect** -- or `adcc` in short -- is a `python`-based framework to
connect to arbitrary programs and perform calculations based on the
algebraic-diagrammatic construction
approach (ADC) on top of their existing self-consistent field (SCF) procedures.
Four SCF codes can be used with `adcc` out of the box, namely
[molsturm](https://molsturm.org),
[psi4](https://github.com/psi4/psi4),
[pyscf](https://github.com/pyscf/pyscf)
and veloxchem.

The range of supported algebraic-diagrammatic construction (ADC)
methods includes the ADC(n) family **up to level 3**,
including spin-flip and core-valence separation variants.
For all methods transition and excited state **properties are available**.
See the [Performing ADC calculations with `adcc`](calculations.md)
for more details.

The next code snippet should give you an idea,
how `adcc` works in practice.
It shows how an ADC(3) calculation for 3 singlet excited states
of water can be performed on top of a restricted Hartree-Fock reference
computed using `pyscf`.
```python
from pyscf import gto, scf
import adcc

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

# Run an ADC(3) calculation, solving for 3 singlets
state = adcc.adc3(scfres, n_singlets=3)

# Print the resulting states
print(state.describe())
```
Sounds interesting? See [Getting started](getting_started.md)
and [Performing calculations with `adcc`](calculations.md)
for installation instructions and some more information to get going.

## Contents
```eval_rst
.. toctree::
   :maxdepth: 2

   getting_started
   calculations
   reference
   developers
   adccore
   publications
   license
   acknowledgements

* :ref:`genindex`
```
