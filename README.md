# adcc
`adcc` is a python-based framework to connect to arbitrary host
programs for running ADC calculations.

## Quick overview
Currently `adcc` supports all ADC(n) variants up to level 3,
that is:
- ADC(0)
- ADC(1)
- ADC(2)
- ADC(2)-x
- ADC(3)
- Core-valence variants:
	- For all ADC(n) methods excluding ADC(3), the core-valence approximation
	  can be applied
- Spin-flip variants:
	- For a black-box computation of low-multiplicity multi-reference problems
	  spin-flip ADC variants can be employed.
- Properties:
	- One-particle transition density matrices and one-particle excited
	  states density matrices can be computed for all implemented methods.
	- These matrices can be used to compute properties such
	  as the oscillator strength in user code.
- Both restricted as well as unrestricted references are supported.


## Installation
Once we have figured out how to do it, it should be as simple as
```
# Install pybind first to suppress error messages during
# installation of adcc.
pip install pybind11
pip install adcc
```
Right now, the package is not
uploaded to [pypi](https://pypi.org), however.


## Developer setup
### Users with access to adccore source code
Having cloned this `adcc` repository,
proceed to clone `adccore` **into the `adcc` directory**,
that is, run
```
git clone https://path/to/adccore adccore
```
inside the directory, where this **README.md** file is found.
You should now have the following folder structure:
```
adcc/README.md
adcc/adcc/__init__.py
...
adcc/adccore/README.md
adcc/adccore/CMakeLists.txt
...
```
Afterwards build, test and (optionally) install `adcc`.
```
./setup.py test   # Builds and tests adcc
pip install .     # Installs adcc (optional)
```

### Users without access to adccore source code
Download the tarball of the compiled `adccore` library
into the `external/adccore` directory:
```
cd ./external/adccore
wget " https://path/to/binary/adccore-VERSION.tar.gz"
tar xzf adccore.tar.gz
cd ../..
```

Afterwards proceed as above, that is:
```
./setup.py test   # Builds and tests adcc
pip install .     # Installs adcc (optional)
```

## Commands of the `setup.py` script worth knowing
The following commands of the `setup.py` script are very useful
for development and worth knowing:

- `setup.py build_ext`: Build the `C++` part of `adcc` in the current directory.
  This includes `adccore` in case you have the source code repository set up as
  described above.
- `setup.py test`: Run the `adcc` unit tests via `pytest`. Implies `build_ext`.


## `adcc` and `pyscf` example
The following example runs an SCF in `pyscf` and an ADC(3) calculation on top:
```python
from pyscf import gto, scf
import adcc

# Run SCF in pyscf using a cc-pvtz basis
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

# Run an adc3 calculation, solving for 3 singlets
state = adcc.adc3(scfres, n_singlets=3, conv_tol=1e-6)

# Print the resulting states
print(state.describe())
```
More examples can be found in the `examples` folder.

## Source code and binary license
The `adcc` source code contained in this repository is released
under the [GNU Lesser General Public License v3 (LGPLv3)](LICENSE).
This license does, however, not apply to the binary
`adccore.so` file distributed inside the folder `/adcc/lib/` of
the release tarball. For its licensing terms,
see [LICENSE_adccore](LICENSE_adccore).
