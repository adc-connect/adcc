:github_url: https://github.com/adc-connect/adcc/blob/master/docs/developers.rst

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
feel free to :ref:`contact-us` and see :ref:`adccore-sources`
for setup details.

.. _adccore-sources:

Development setup with access to adccore source code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you need to modify both adcc and ``adccore``
to implement a new feature,
you first need to get access
`adccore source code <https://code.adc-connect.org/adccore>`_,
which is not yet publicly available.
Feel free to :ref:`contact-us` to discuss this.

Once you do, configure the url of the ``adccore`` remote
on your system. For this drop a file ``~/.adccore.json``
in your Home directory with the contents

.. code-block:: json

   {"upstream": "ssh://location_of_the_adccore_repository.git"}

where ``location_of_the_adccore_repository.git`` is appropriately
replaced by the url to the ``adccore`` remote. Afterwards you can
proceed as above, i.e. just clone the adcc sources via

.. code-block:: shell

   git clone https://code.adc-connect.org

and initalise the build via

.. code-block:: shell

   ./setup.py test

This will automatically clone ``adccore`` into the subfolder ``adccore``
of the adcc source repository and trigger both building and testing
of ``adccore`` **and** adcc.

Notice, that in this setup, the build system of ``adccore``
is integrated with the ``setup.py`` from adcc,
such that building ``adccore`` is automatically
triggered from the ``setup.py`` script of the adcc repository.
You generally do not need to worry about keeping the two repositories
in sync or building them in the correct order:
If you modify a file inside ``adccore`` the ``setup.py`` script from adcc
will automatically trigger a compilation of this component for you.

One case, which does require manual work, however, is if adcc requires
an newer version of ``adccore``. In this case you will be presented with
an error and you have to manually checkout the appropriate ``adccore``
version by running ``git checkout`` inside the ``adccore`` subdirectory.
For example to obtain version ``0.0.0`` of ``adccore``,
you need to run

.. code-block:: shell

   git checkout v0.0.0.

This is done to avoid automatically overwriting some development changes
you might have made inside ``adccore``.


Building adccore with MKL support
---------------------------------

If you have full source code access
and you are able to follow the :ref:`adccore-sources`,
the `Intel Math Kernel Library (R) <https://software.intel.com/en-us/mkl>`_
can also be integrated into adccore and thus adcc.
In fact this integration happens automatically during the build
process of adccore, given that a numpy linked to the MKL was
detected. For this reason proceed as follows:

1. Load the MKL modules or activate the MKL in your shell as you usally do.
2. Build and install numpy with linkage to this MKL,
   e.g. `Build numpy from source <https://docs.scipy.org/doc/numpy/user/building.html>`_.
3. Build adcc and adccore as described in :ref:`adccore-sources`.


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
