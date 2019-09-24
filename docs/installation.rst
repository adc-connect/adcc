.. _installation:

Installation
============

Installing adcc
---------------

.. important:: This is the anticipated installation process,
               which is not yet functional.

The first step is to **install adcc** from `PyPi <https://pypi.org>`_, using ``pip``:

.. code-block:: shell

   pip install pybind11     # Install pybind11 first to suppress some error messages
   pip install adcc


.. note::
   TODO This needs more details!

If the installation of adcc fails due to an issue with compiling the
python extension,
make sure that your ``pip`` uses ``gcc`` and not ``clang`` or any other compiler.
To enforce using ``gcc`` (e.g. one from Homebrew or a custom installation),
set the environment variables ``CC`` and ``CXX``.

Installing an SCF code
----------------------

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

To **install Psi4** follow the
`conda binary distribution <http://psicode.org/psi4manual/master/conda.html>`_
instructions. The code is also available in Linux
distributions (e.g. `Debian <https://packages.debian.org/stable/psi4>`_).

A **PySCF installation** can be achieved following the
`PySCF quickstart guide <https://pyscf.github.io/quickstart.html>`_.
E.g. if you are using ``pip`` this boils down to

.. code-block:: shell

   pip install pyscf

That's it
---------

Congratulations! With these packages installed you are all set
to run ADC calculations.
Feel free to take a look at the
:ref:`performing-calculations` section
for learning how to use adcc in practice.

Finally, if you are interested in developing or contributing
to adcc, even the better! In this case we hope
the :ref:`devnotes` will provide
you with some useful pointers to get started.
