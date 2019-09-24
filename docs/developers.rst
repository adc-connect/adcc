.. _devnotes:

Developer's notes
=================

Components of adcc
------------------

The adcc project consists of three main components,
namely the adcc python library,
the ``adccore`` C++ layer as well as ``libadcc``.

The distribution of workload is such that ``adccore`` is responsible for:

- Interaction with the underlying linear algebra backend, i.e. the tensor library
- Implementation of the ADC working equations or interface to ``libadc``
  for more complex expressions.
- A unified interface to import Hartree-Fock results into the tensor library
- A unified interface to compute matrix-vector products
  of the ADC matrix in a contraction-based numerical scheme (e.g. in Python).

In contrast the adcc Python module

- Implements iterative numerical solver schemes (e.g. the Davidson diagonalisation)
- Interacts with Python-based SCF codes
- Provides high-level functionality and user interaction
- Orchestrates the workflow of an ADC calculation
- Implements analysis and visualisation of results.

While these first two components thus contain real functionality,
``libadcc`` is just a wrapper around ``adccore``.
It makes use of `Pybind11 <https://pybind11.readthedocs.io>`_
in order to expose the core code to Python.
This allows to use ``adccore`` from Python (via ``libadcc``)
and directly from C++.

The functionality of adcc has already been described
in :ref:`performing-calculations` and :ref:`full-reference`.
In fact many of the functions and classes described
in these chapters are only partly implemented in adcc
and inherit from components defined in ``adccore``,
which is discussed in more detail in :ref:`adccore-layer`.

Obtaining the adcc sources
--------------------------

.. note::
   Links not yet live!
   Check them once they are

The source code of adcc can be obtained
`from github <https://github.com/adc-connect/adcc>`_,
simply by cloning

.. code-block:: shell

   git clone https://code.adc-connect.org

Unlike adcc, the the ``adccore`` sources are not yet publicly available
at the moment. They should not be
neccessary for most development work on adcc,
since the ``setup.py`` script of adcc
will take care of downloading and installing the appropriate
binary version of ``adccore`` automatically.
This is triggered simply by building and testing adcc,
which can be achieved by

.. code-block:: shell

   ./setup.py test

Afterwards modifications on the adcc python level can be done
at wish building on the rich interface of functionality
exposed from ``adccore`` to the python level.
See the `Pybind11 extension <https://code.adc-connect.org/extension>`_
for details.

In case you need a full source code setup
feel free to contact us and see :ref:`adccore-sources`
for setup details.

.. _adccore-sources:

Development setup with access to adccore source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to modify both adcc and ``adccore``,
to implement a new feature,
you first need access to the
`adccore source code <https://code.adc-connect.org/adccore>`_.
If you do, it is easiest to have both the source
code for adcc and ``adccore`` checked out in one directory,
to be able to work on them simultaneously.

For this first clone the adcc repository

.. code-block:: shell

   git clone https://code.adc-connect.org

and then clone the ``adccore`` repository **into the adcc directory**,
that is, run

.. code-block:: shell

   cd adcc
   git clone https://code.adc-connect.org/adccore adccore

directly thereafter.
You should now have the following folder structure:

.. code-block:: shell

   adcc/README.md
   adcc/adcc/__init__.py
   ...
   adcc/adccore/README.md
   adcc/adccore/CMakeLists.txt
   ...

In this way the build system of ``adccore`` can be
controlled directly from the ``setup.py`` script of the adcc
repository, such that you generally do not need to worry about
keeping the two repositories in sync or building them in the correct order:
If you modify a file inside ``adccore`` the ``setup.py`` script from adcc
will automatically trigger a compilation (and appropriate installation)
of this component for you.

This means that building and testing ``adccore`` **and** adcc
now boils down to a simple

.. code-block:: shell

   ./setup.py test

``setup.py`` reference
----------------------
The ``setup.py`` script of adcc is a largely a typical setuptools script,
but has a few additional commands and features worth knowing:

- ``setup.py build_ext``: Build the C++ part of adcc in the current directory.
  This includes ``adccore`` in case you have the source code repository set up
  as described in :ref:`adccore-sources`.
- ``setup.py test``: Run the adcc unit tests via
  `pytest <https://docs.pytest.org>`_. Implies ``build_ext``.
  This command has a few useful options:

    - ``-m full``: Run the full test suite not only the fast tests
    - ``-s``: Skip updating the testdata
    - ``-a``: Pass additional arguments to ``pytest``
      (`See pytest documentation <https://docs.pytest.org/en/latest/usage.html>`_).
      This is extremely valuable in combination with the ``-k`` and ``-s`` flags
      of ``pytest``.
      For example

      .. code-block:: shell

         ./setup.py test -a "-k 'functionality and adc2'"

      will run only the tests, which have the keywords "functionality" and
      "adc2" in their description. Of course in such a case still all changes in ``adccore``
      will trigger a rebuild of the C++ components of adcc before running these tests ...
- ``setup.py build_docs``: Build the documentation locally using
  Doxygen and Sphinx. See the section below for details.

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

.. important:: The above does not work yet.

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
