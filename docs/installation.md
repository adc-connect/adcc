```eval_rst
.. _installation:

```

# Installation

## Obtaining adcc

``` important:: This is the anticipated installation process,
                which is not yet functional.

```

The first step is to **get `adcc`** from [`pypi`](https://pypi.org) using `pip`:
```sh
pip install pybind11     # Install pybind11 first to suppress some error messages
pip install adcc
```

```eval_rst
.. note::
   TODO This needs more details!

```

If the installation of adcc fails due to an issue with compiling the
python extension,
make sure that your `pip` uses `gcc` and not `clang` or any other compiler.
To enforce using `gcc` (e.g. one from Homebrew or a custom installation),
set the environment variables `CC` and `CXX`.

## Obtaining an SCF code

Since adcc does not contain a self-consistent field (SCF) code
at least one of the supported SCF programs needs to be installed as well.
Without expressing any particular preference,
this documentation will mostly focus on Psi4 and PySCF,
since these are very easy to obtain, install and use.

To **install `psi4`** follow the
[conda binary distribution](http://psicode.org/psi4manual/master/conda.html)
instructions. The code is also available in Linux
distributions (e.g. [Debian](https://packages.debian.org/stable/psi4)).

A **PySCF installation** can be achieved following the
[PySCF quickstart guide](https://pyscf.github.io/quickstart.html).
E.g. if you are using `pip` this boils down to
```sh
pip install pyscf
```

## That's it

Congratulations! With these packages installed you are all set
to run ADC calculations.
Feel free to take a look at the
[Performing ADC calculations with adcc](calculations.md) section
for learning how to use adcc in practice.

Finally, if you are interested in developing or contributing
to adcc, even the better! In this case we hope
the [Developer's notes](developers.md) will provide
you with some useful pointers to get started.
