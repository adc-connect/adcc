<img src="https://raw.githubusercontent.com/adc-connect/adcc/master/docs/logo/logo.png" alt="adcc logo" height="100px" />

# adcc: Seamlessly connect your program to ADC
[![pypi](https://img.shields.io/pypi/v/adcc)](https://pypi.org/project/adcc)
[![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/adc-connect/adcc/blob/master/LICENSE)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://adc-connect.org)
[![Travis](https://travis-ci.org/adc-connect/adcc.svg?branch=master)](https://travis-ci.org/adc-connect/adcc)

adcc (**ADC-connect**) is a python-based framework for performing
the calculation of molecular spectra and electronically excited states
based upon the algebraic-diagrammatic construction (ADC) approach.

Arbitrary host programs may be used to supply a
self-consistent field (SCF) reference to start off the ADC calculation.
Currently adcc comes with ready-to-use interfaces to four programs,
namely pyscf, psi4, VeloxChem or molsturm. Adding other SCF codes or even
statically computed data can be easily achieved as well.
For more details and installation instructions
[see the `adcc` documentation](https://adc-connect.org).

## Citation
A preprint of our paper describing `adcc` can be found
[on HAL](https://hal.archives-ouvertes.fr/hal-02319517)
or [on arXiv](http://arxiv.org/pdf/1910.07757).


## Licence note
The `adcc` source code contained in this repository is released
under the [GNU General Public License v3 (GPLv3)](LICENSE).
This license does, however, not apply to the binary
`adccore.so` file (on Linux) or `adccore.dylib` file (on macOS)
distributed inside the folder `/adcc/lib/` of the `adcc` release tarball.
For its licensing terms, see [LICENSE_adccore](LICENSE_adccore).
