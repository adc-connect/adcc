# Developer's notes

## Components of `adcc`
The `adcc` project consists of three main components,
namely the `adcc` python library,
the `adccore` C++ layer as well as `libadcc`.

The distribution of workload is such that `adccore` is responsible for:

- Interaction with the underlying linear algebra backend, i.e. the tensor library
- Implementation of the ADC working equations
- A unified interface to import Hartree-Fock results into the tensor library
- A unified interface to compute matrix-vector products
  of the ADC matrix in a contraction-based numerical scheme (e.g. in `python`).

In contrast, `adcc`:

- Implements iterative numerical solver schemes (e.g. the Davidson diagonalisation)
- Interacts with `python`-based SCF codes
- Provides high-level functionality and user interaction
- Orchestrates the workflow of an ADC calculation
- Implements analysis and visualisation of results.

While these first two components thus contain real functionality,
`libadcc` is just a tiny wrapper around `adccore`.
It makes use of [Pybind11](https://pybind11.readthedocs.io/)
in order make the interfaces of `adccore` more `python`-friendly.
In this way it is convenient to use `adccore` from `python` (namely via `libadcc`)
and directly from C++ as well.

The functionality of `adcc` has already been described
in [Performing calculations with `adcc`](calculations.md)
and [Functionality reference](reference.md).
In fact many of the functions and classes described there do not actually
live in `adcc`, but are inherited from components defined in `libadcc`.
A separate documentation of `libadcc` is thus not done here.
`adccore`, however, is discussed in more details in
[The `adccore` C++ layer](adccore.md).

## Development setup for `adcc`
Given the above structure of the `adcc` project,
it is sometimes necessary to modify both `adcc` and `adccore`
to implement a new feature.
For this reason it is easiest to have both the source
code for `adcc` and `adccore` checked out to be able to
work on them simultaneously.

For this first clone the `adcc` repository
```
git clone https://path/to/adcc adcc
```
and then clone the `adccore` repository **into the `adcc` directory**,
that is, run
```
cd adcc
git clone https://path/to/adccore adccore
```
directly thereafter.
You should now have the following folder structure:
```
adcc/README.md
adcc/adcc/__init__.py
...
adcc/adccore/README.md
adcc/adccore/CMakeLists.txt
...
```
In this way the build system of `adccore` can be largely
controlled directly from the `setup.py` script of the `adcc`
repository, such that you do not need to worry about
keeping the two repositories in sync.
If you modify a file inside `adccore` the `setup.py` script from `adcc`
will automatically trigger a compilation (and appropriate installation)
of this component for you.

This means that building and testing `adccore` **and** `adcc`
now boils down to a simple
```
./setup.py test
```

## `setup.py` command reference
The `setup.py` script of `adcc` is a largely a typical setuptools script,
but has a few additional commands and features worth knowing:

- `setup.py build_ext`: Build the C++ part of `adcc` in the current directory.
  This includes `adccore` in case you have the source code repository set up as
  described above.
- `setup.py test`: Run the `adcc` unit tests via [`pytest`](https://docs.pytest.org).
  Implies `build_ext`.
  This command has a few useful options:
    - `-m full`: Run the full test suite not only the fast tests
    - `-s`: Skip updating the testdata
    - `-a`: Pass additional arguments to `pytest` ([See pytest documentation](https://docs.pytest.org/en/latest/usage.html)).
      This is extremely valuable in combination with the `-k` and `-s` flags of `pytest`.
      For example
      ```
       ./setup.py test -a "-k 'functionality and adc2'"
      ```
      will run only the tests, which have the keywords "functionality" and
      "adc2" in their description. Of course in such a case still all changes in `adccore`
      will trigger a rebuild of the `C++` components of `adcc` before running these tests ...

- `setup.py build_docs`: Build the documentation locally using
  Doxygen and Sphinx. See the section below for details.

## Building the documentation
This very document is created with [Sphinx](http://sphinx-doc.org) and
[Doxygen](http://doxygen.nl/) extracting parts of the content
directly from the source code documentation.
Building the documentation locally thus requires [Doxygen](http://doxygen.nl/)
to be properly installed on you computer and additionally
Sphinx and a few of its plugins
(e.g. [recommonmark](https://github.com/rtfd/recommonmark)
for [markdown support](https://www.sphinx-doc.org/en/master/usage/markdown.html)
and [breathe](https://github.com/michaeljones/breathe)).
This can be achieved using
```
pip install adcc[build_docs]
```

``` important:: This does not work yet.

```
On the `python`-side we follow the [numpy docstring standard][npdoc].

## Coding conventions
On the `python` end, the repository contains a `setup.cfg` file,
which largely defines the code conventions. Use your favourite `flake8`-plugin
to ensure compliance. On the `C++`-end we provide `.clang-format` files,
such that automatic formatting can be done with
your favourite tool based on `clang-format`.

## What other developers use
- **VIM**: For setting up `vim` with this repository, use the following plugins:
	* [YouCompleteMe](https://github.com/Valloric/YouCompleteMe)
	* [impsort.vim](https://github.com/tweekmonster/impsort.vim)
	* [vim-templates](https://github.com/tibabit/vim-templates)

[npdoc]: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
