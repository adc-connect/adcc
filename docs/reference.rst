:github_url: https://github.com/adc-connect/adcc/blob/master/docs/reference.rst

.. _pyapi:

API reference
=============

.. note::
   Work in progress. Many functions do not yet follow
   the numpy standard in their documentation!

This page contains a structured overview of the python API of adcc.
See also the :ref:`genindex`.

.. _adccmodule:

adcc module
-----------

.. automodapi:: adcc
   :no-inheritance-diagram:
   :no-heading:


.. _libadcc:

libadcc: Python bindings
------------------------

The libadcc Python module contains python bindings of :ref:`libadcc-layer`.
They are generated directly from the C++ source code
using `pybind11 <https://pybind11.readthedocs.io>`_
and the sources contained in the
`extension <https://github.com/adc-connect/adcc/tree/master/libadcc_src/pyiface>`_
subfolder of the adcc GitHub repository.

It is not recommended calling these functions directly,
but instead resort to the higher-level functionality
from the :ref:`adccmodule`.

.. automodapi:: libadcc
   :no-inheritance-diagram:
   :no-heading:


.. _libadcc-layer:

libadcc: C++ library
--------------------

A reference of the C++ part of *libadcc*
and its classes and functions can be found in the following.
The functions and classes discussed here are exposed to Python
as the :ref:`libadcc` python module.

Reference state
^^^^^^^^^^^^^^^
This category lists the *libadcc* functionality,
which imports the data from the :cpp:class:`libadcc::HartreeFockSolution_i`
interface into the :cpp:class:`libadcc::ReferenceState`
for internal use by the library.
See :ref:`hostprograms` for details how to connect
host programs to adcc.
Important classes in the process are :cpp:class:`libadcc::MoSpaces`,
which collects information about the occupied and virtual
orbital spaces, and :cpp:class:`libadcc::MoIndexTranslation`,
which maps orbitals indices between the ordering used by ``libadcc``
and the one used by the SCF program.

.. doxygengroup:: ReferenceObjects
   :members:
   :content-only:


ADC guess setup
^^^^^^^^^^^^^^^
.. doxygengroup:: AdcGuess
   :members:
   :content-only:


One-particle operators
^^^^^^^^^^^^^^^^^^^^^^
.. doxygengroup:: Properties
   :members:
   :content-only:


Tensor interface
^^^^^^^^^^^^^^^^
The generalised :cpp:class:`libadcc::Tensor` interface
used by adcc and libadcc to perform tensor operations.

.. doxygengroup:: Tensor
   :members:
   :content-only:


Utilities
^^^^^^^^^
Some random things to set up shop.

.. doxygengroup:: Utilities
   :members:
   :content-only:


Tensor implementation using libtensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section describes the implementation of the
Tensor functionality of :cpp:class:`libadcc::Tensor`
using the libtensor tensor library.

.. doxygengroup:: TensorLibtensor
   :members:
   :content-only:
