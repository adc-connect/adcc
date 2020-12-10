<img src="https://raw.githubusercontent.com/adc-connect/adcc/master/docs/logo/logo.png" alt="adcc logo" height="100px" />

# adcc: Seamlessly connect your program to ADC                                                                
| **Documentation** | [![][docs-img]][docs-url] [![][binder-img]][binder-url] |
| :------ | :------- |
| **Build Status**  | [![][ci-img]][ci-url] [![][cov-img]][cov-url] [![][lgtm-img]][lgtm-url] |
|  **Installation** | [![][pypi-img]][pypi-url] [![][conda-img]][conda-url] [![][license-img]][license-url]  |

[docs-img]: https://img.shields.io/badge/doc-latest-blue.svg
[docs-url]: https://adc-connect.org
[binder-img]: https://mybinder.org/badge_logo.svg
[binder-url]: https://try.adc-connect.org
[ci-img]: https://github.com/adc-connect/adcc/workflows/CI/badge.svg?branch=master&event=push
[ci-url]: https://github.com/adc-connect/adcc/actions
[cov-img]: https://coveralls.io/repos/adc-connect/adcc/badge.svg?branch=master&service=github
[cov-url]: https://coveralls.io/github/adc-connect/adcc?branch=master
[license-img]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[license-url]: https://github.com/adc-connect/adcc/blob/master/LICENSE
[pypi-img]: https://img.shields.io/pypi/v/adcc
[pypi-url]: https://pypi.org/project/adcc
[conda-img]: https://anaconda.org/adcc/adcc/badges/version.svg
[conda-url]: https://anaconda.org/adcc/adcc
[lgtm-img]: https://img.shields.io/lgtm/grade/python/github/adc-connect/adcc?label=code%20quality
[lgtm-url]: https://lgtm.com/projects/g/adc-connect/adcc/context:python

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
