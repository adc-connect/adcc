<img src="https://raw.githubusercontent.com/adc-connect/adcc/master/docs/logo/logo.png" alt="adcc logo" height="100px" />

# adcc: Seamlessly connect your program to ADC                                                                
| **Documentation:** | [![][docs-img]][docs-url] |
| :-------------- | :--------------------------------------------------------- |
| **Build Status:**  | [![][ci-img]][ci-url] [![][cov-img]][cov-url] |
|  **Installation:** | [![][pypi-img]][pypi-url] [![][conda-img]][conda-url] [![][license-img]][license-url]  |

[docs-img]: https://img.shields.io/badge/doc-latest-blue.svg
[docs-url]: https://adc-connect.github.io/adcc/
[ci-img]: https://github.com/adc-connect/adcc/actions/workflows/ci.yaml/badge.svg
[ci-url]: https://github.com/adc-connect/adcc/actions/workflows/ci.yaml
[cov-img]: https://coveralls.io/repos/adc-connect/adcc/badge.svg?branch=master&service=github
[cov-url]: https://coveralls.io/github/adc-connect/adcc?branch=master
[license-img]: https://img.shields.io/badge/License-GPL%20v3-blue.svg
[license-url]: https://github.com/adc-connect/adcc/blob/master/LICENSE
[pypi-img]: https://img.shields.io/pypi/v/adcc
[pypi-url]: https://pypi.org/project/adcc
[conda-img]: https://anaconda.org/conda-forge/adcc/badges/version.svg
[conda-url]: https://anaconda.org/conda-forge/adcc

adcc (**ADC-connect**) is a Python-based framework for calculating molecular spectra and electronically excited states
with the algebraic-diagrammatic construction (ADC) approach.

Arbitrary host programs may be used to supply a
self-consistent field (SCF) reference to start off the ADC calculation.
Currently, adcc comes with ready-to-use interfaces to different SCF programs like PySCF and Psi4.
Adding other SCF codes or starting a calculation from statically computed data
can be easily achieved.

## Installation

From PyPI:
```bash
pip install adcc
```

From conda-forge:
```bash
conda install adcc -c conda-forge
```

Local development version:
```bash
git clone git@github.com:adc-connect/adcc.git
cd adcc
pip install -e .[dev]
```

For documentation or more detailed installation instructions you might want to have
a look at the [adcc documentation][docs-url].

## Citation

**Paper:** | [![](https://img.shields.io/badge/DOI-10.1002/wcms.1462-blue)](https://doi.org/10.1002/wcms.1462)
:--------- | :----------------------------------------------------------
**Code:**  | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3519764.svg)](https://doi.org/10.5281/zenodo.3519764)

If you use adcc, please cite
[our paper in WIREs Computational Molecular Science](https://doi.org/10.1002/wcms.1462).
A preprint can be found
[on HAL](https://hal.archives-ouvertes.fr/hal-02319517)
or [on arXiv](http://arxiv.org/pdf/1910.07757).
