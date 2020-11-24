:github_url: https://github.com/adc-connect/adcc/blob/master/docs/installation.rst

.. _installation:

Installation
============

For running ADC calculations with adcc you will need two things:
adcc itself and a :ref:`host program<install-hostprogram>`
(i.e. SCF code) of your choice.

We have some more detailed guides for installing adcc
:ref:`install-conda`, :ref:`install-pip-debian`
and :ref:`install-pip-macos`, where we know things should
be working.
If you encounter problems see :ref:`troubleshooting`.
Please get in touch
by `opening an issue <https://github.com/adc-connect/adcc/issues>`_
if you cannot get adcc to work.

Installing adcc
---------------

.. _install-conda:

Using conda (on Debian/Ubuntu and macOS)
........................................

The `conda <https://conda.io>`_ binary packages can be installed
using the `adcc <https://anaconda.org/adcc/>`_ channel:

.. code-block:: shell

   conda install -c adcc adcc

This should work on a recent Debian, Ubuntu or macOS
and with python 3.6, 3.7 and 3.8.


.. _install-pip-debian:

Using pip (on Debian / Ubuntu)
..............................

For installing adcc from `PyPi <https://pypi.org>`_, using ``pip``,
the procedure for Debian / Ubuntu and :ref:`macOS <install-pip-macos>` differs.
For Debian and Ubuntu:

1. **Install openblas:**
   Make sure you have the `openblas <http://www.openblas.net/>`_
   BLAS library installed. This library is reasonably wide-spread,
   so there is a good chance it may already be installed on your system.
   If not, you can easily install it using

   .. code-block:: shell

      sudo apt-get install libopenblas-base

2. **Install compilation requirements:**
   For compiling the Python extensions of adcc,
   you need to have the the ``python`` development headers
   and some essential build packages installed.
   This can be achieved via

   .. code-block:: shell

      sudo apt-get install python3-dev build-essential

3. **Install adcc:**

   .. code-block:: shell

      pip install pybind11     # Install pybind11 first to suppress some error messages
      pip install adcc


.. _install-pip-macos:

Using pip (on macOS)
....................

.. attention::
   macOS support only covers Mojave (10.14) and Catalina (10.15) for the moment.
   For other macOS versions, please :ref:`install adcc using conda <install-conda>`.

The installation on macOS requires a ``clang`` compiler.
Make sure to have XCode and the command line tools installed.
Then install **adcc** using ``pip``:

.. code-block:: shell

   pip install pybind11     # Install pybind11 first to suppress some error messages
   pip install adcc

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

  .. code-block:: shell

     conda install -c psi4 psi4

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
for learning how to use adcc in practice or take
a look at our `examples folder on github <https://code.adc-connect.org/tree/master/examples>`_.

Finally, if you are interested in developing or contributing
to adcc, even the better! In this case we hope
the :ref:`devnotes` will provide
you with some useful pointers to get started.



.. _troubleshooting:

Troubleshooting
---------------

If the installation of adcc fails due to an issue with compiling the
python extension, check the following:

- Make sure your ``pip`` uses the correct compiler. On Linux we only support
  ``gcc`` and not ``clang``. On macOS we only support Apple ``clang`` and
  not ``gcc``. To enforce a compiler, set the environment variables ``CC`` and ``CXX``
  to the full path of your C and C++ compilers, respectively.
