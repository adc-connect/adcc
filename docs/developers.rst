:github_url: https://github.com/adc-connect/adcc/blob/master/docs/developers.rst

.. _devnotes:

Developer's notes
=================

Components of adcc
------------------

The adcc project consists of two main components,
namely the adcc python library and the hybrid
C++/Python ``libadcc`` library.

The distribution of workload is such that ``libadcc`` is responsible for:

- Interaction with the underlying linear algebra backend, i.e. the tensor library
- A unified interface to import Hartree-Fock results into the tensor library

In contrast, the adcc Python module

- Implements iterative numerical solver schemes (e.g. the Davidson diagonalisation)
- Implements all working equations (Møller-Plesset perturbation theory and ADC matrix expressions)
- Interacts with Python-based SCF codes
- Provides high-level functionality and user interaction
- Orchestrates the workflow of an ADC calculation
- Implements analysis and visualisation of results.

The ``libadcc`` library makes use of `Pybind11 <https://pybind11.readthedocs.io>`_
to expose the necessary C++ functionality to Python.

The functionality of adcc has already been described
in :ref:`performing-calculations` and :ref:`topics`.
In fact many of the functions and classes described
in these chapters are only partly implemented in adcc
and inherit from components defined in ``libadcc``,
which is discussed in more detail in :ref:`libadcc-layer`.

Obtaining the adcc sources
--------------------------

The entire source code of adcc can be obtained
`from github <https://github.com/adc-connect/adcc>`_,
simply by cloning

.. code-block:: shell

   git clone https://code.adc-connect.org

Building and testing libadcc and adcc can be achieved by

.. code-block:: shell

   ./setup.py test

Afterwards modifications on the adcc python level can be done
at wish without re-running any build commands. If you modify source
files in ``libadcc``, make sure to re-run the ``./setup.py test``
such that your changes are being compiled into
the ``libadcc`` shared library.


``setup.py`` reference
----------------------
The ``setup.py`` script of adcc is a largely a typical setuptools script,
but has a few additional commands and features worth knowing:

- ``setup.py build_ext``: Build the C++ part of adcc in the current directory.
- ``setup.py cpptests``: Build and run the C++ tests for ``libadcc``

Documentation, documentation, documentation
-------------------------------------------

This very document is created with `Sphinx <http://sphinx-doc.org>`_ and
`Doxygen <http://doxygen.nl>`_ extracting parts of the content
directly from the source code documentation.
Building the documentation locally thus requires both these tools and additionally
and a few Sphinx plugins
(e.g. `breathe <https://github.com/michaeljones/breathe>`_).
This can be achieved using

.. code-block:: shell

   pip install adcc[build_docs]

On the Python-side we follow the `numpy docstring standard <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_.

Coding conventions
------------------

On the Python end, the repository contains a ``setup.cfg`` file,
which largely defines the code conventions. Use your favourite ``flake8``-plugin
to ensure compliance. On the C++-end we provide ``.clang-format`` files,
such that automatic formatting can be done with
your favourite tool based on ``clang-format``.

What other developers use
-------------------------

- **VIM**: For setting up ``vim`` with this repository,
  you can use the following plugins:

	* `YouCompleteMe <https://github.com/Valloric/YouCompleteMe>`_
	* `impsort.vim <https://github.com/tweekmonster/impsort.vim>`_
	* `vim-templates <https://github.com/tibabit/vim-templates>`_
