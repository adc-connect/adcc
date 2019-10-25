.. _installation:

Installation
============

Roughly speaking installing adcc boils down to
installing openblas, adcc itself and a :ref:`host program<install-hostprogram>`
(i.e. SCF code) of your choice.

We have some more detailed guides for :ref:`install-debian`
and :ref:`install-macos`, where we know things should
be working. Please let us know
if you managed to get adcc to work in the contexts
of other OSes or distributions.

Installing adcc
---------------

.. _install-debian:

Debian and Ubuntu
.................

1. **openblas:**
   Make sure you have the `openblas <http://www.openblas.net/>`_
   BLAS library installed. This library is reasonably wide-spread,
   so there is a good chance it may already be installed on your system.
   If not, you can easily install it using

   .. code-block:: shell

      sudo apt-get install libopenblas-base

2. **Compilation requirements:**
   For compiling the Python extensions of adcc,
   you nee dto have the the ``python`` development headers
   and some essential build packages installed.
   This can be achieved via

   .. code-block:: shell

      sudo apt-get install python3-dev build-essential

3. **adcc:**
   Best install it from `PyPi <https://pypi.org>`_, using ``pip``:

   .. code-block:: shell

      pip install pybind11     # Install pybind11 first to suppress some error messages
      pip install adcc


.. _install-macos:

macOS 10.13 (High Sierra) and 10.14 (Mojave)
............................................

.. attention::
   macOS support is still experimental and so far
   only covers High Sierra and Mojave.
   We would love to hear your feedback in case things fail.

.. note::
   Supported from adcc 0.13.2.

0. **Homebrew:**
   Support for macOS currently requires the `Homebrew <https://brew.sh>`_ package manager
   and a recent version of ``gcc`` (e.g. ``gcc@9``). Hopefully, we will support ``clang`` in the future.
   
1. **adcc:**
   Install from `PyPi <https://pypi.org>`_, using ``pip``:

   .. code-block:: shell

      pip install pybind11     # Install pybind11 first to suppress some error messages
      CXX=g++-9 CC=gcc-9 pip install adcc   # Install adcc using the correct compiler for Python bindings

.. _install-hostprogram:

Installing a host program
-------------------------

Since adcc does not contain a self-consistent field (SCF) code
you should install one of the supported SCF programs needs as well.
Without expressing any particular preference,
this documentation will mostly focus on Psi4 and PySCF,
since these are very easy to obtain, install and use.
If you prefer, feel free to install
`molsturm <https://molsturm.org>`_
or `veloxchem <https://veloxchem.org>`_ instead.
Also note, that connecting to further host programs is not too hard
and can be achieved via a dictionary or an HDF5 file,
see :ref:`hostprograms` for details.

Installing Psi4
...............

- Either use the
  `conda binary distribution <http://psicode.org/psi4manual/master/conda.html>`_
- **or** use the version packaged in `Debian <https://packages.debian.org/stable/psi4>`_
  or Ubuntu via

  .. code-block:: shell

     sudo apt-get install psi4

Installing PySCF
................

A **PySCF installation** can be achieved following the
`PySCF quickstart guide <https://pyscf.github.io/quickstart.html>`_.
E.g. if you are using ``pip`` this boils down to

.. code-block:: shell

   pip install pyscf


Finishing the setup
-------------------

Congratulations! With these packages installed you are all set
to run ADC calculations.
Feel free to take a look at the
:ref:`performing-calculations` section
for learning how to use adcc in practice.

Finally, if you are interested in developing or contributing
to adcc, even the better! In this case we hope
the :ref:`devnotes` will provide
you with some useful pointers to get started.


Troubleshooting
---------------

If the installation of adcc fails due to an issue with compiling the
python extension, check the following:

- Make sure your ``pip`` uses ``gcc`` and not ``clang`` or any other compiler
  for compiling the ``adcc`` extension.
  To enforce using ``gcc`` (e.g. one from Homebrew or a custom openblas installation),
  set the environment variables ``CC`` and ``CXX`` to the full path of your C and C++
  compilers, respectively.
