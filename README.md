<img src="https://raw.githubusercontent.com/adc-connect/adcc/master/docs/logo/logo.png" alt="adcc logo" height="100px" />

# adcc: Seamlessly connect your program to ADC
[![license](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://github.com/adc-connect/adcc/blob/master/LICENSE)
[![pypi](https://img.shields.io/pypi/v/adcc)](https://pypi.org/project/adcc)
[![anaconda](https://anaconda.org/adcc/adcc/badges/version.svg)](https://anaconda.org/adcc/adcc)
[![Documentation](https://img.shields.io/badge/doc-latest-blue.svg)](https://adc-connect.org)
[![Binder](https://mybinder.org/badge_logo.svg)](https://try.adc-connect.org)
[![Travis](https://travis-ci.org/adc-connect/adcc.svg?branch=master)](https://travis-ci.org/adc-connect/adcc)
[![DOI](https://zenodo.org/badge/215731857.svg)](https://zenodo.org/badge/latestdoi/215731857)

adcc (**ADC-connect**) is a python-based framework for performing
the calculation of molecular spectra and electronically excited states
based upon the algebraic-diagrammatic construction (ADC) approach.

Arbitrary host programs may be used to supply a
self-consistent field (SCF) reference to start off the ADC calculation.
Currently adcc comes with ready-to-use interfaces to four programs,
namely pyscf, psi4, VeloxChem or molsturm. Adding other SCF codes or
starting a calculation from
statically computed data can be easily achieved.

Try adcc in your browser at https://try.adc-connect.org
or take a look at the [adcc documentation](https://adc-connect.org)
for more details and installation instructions.

## Citation

**Paper:** | [![](https://img.shields.io/badge/DOI-10.1002/wcms.1462-blue)](https://doi.org/10.1002/wcms.1462)
-----------| --------------------------------------------------------------------------------------------------------
**Code:**  | [![DOI](https://zenodo.org/badge/215731857.svg)](https://zenodo.org/badge/latestdoi/215731857)

If you use adcc, please cite
[our paper in WIREs Computational Molecular Science](https://doi.org/10.1002/wcms.1462).
A preprint can be found
[on HAL](https://hal.archives-ouvertes.fr/hal-02319517)
or [on arXiv](http://arxiv.org/pdf/1910.07757).

## Licence note
The adcc source code contained in this repository is released
under the [GNU General Public License v3 (GPLv3)](https://github.com/adc-connect/adcc/blob/master/LICENSE).
This license does, however, not apply to the binary
`adccore.so` file (on Linux) or `adccore.dylib` file (on macOS)
distributed inside the folder `/adcc/lib/` of the `adcc` release tarball.
For its licensing terms, see [LICENSE_adccore](https://github.com/adc-connect/adcc/blob/master/LICENSE_adccore).
