```eval_rst
.. _installation:

```

# Installation

``` important:: This is the anticipated installation process,
                which is not yet functional.

```

The first step is to **get `adcc`** from [`pypi`](https://pypi.org) using `pip`:
```sh
pip install pybind11     # Install pybind11 first to suppress some error messages
pip install adcc
```

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

Congratulations! With these packages installed you are all set
to run ADC calculations.
Feel free to take a look at the
[Performing ADC calculations with adcc](calculations.md) section
for learning how to use adcc in practice.

Finally, if you are interested in developing or contributing
to adcc, even the better! In this case we hope
the [Developer's notes](developers.md) will provide
you with some useful pointers to get started.
