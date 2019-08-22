The adccore C++ layer
=====================

This page contains a reference of the *adccore* C++ library
and its classes and functions generated automatically
from the *adccore* source code.

Hartree-Fock interface
----------------------
.. doxygenclass:: adcc::HartreeFockSolution_i
   :members:


Reference state
---------------
This category lists the *adccore* functionality,
which imports the data from the :cpp:class:`adcc::HartreeFockSolution_i`
interface into the :cpp:class:`adcc::ReferenceState`
for internal use by the library.
Important classes in the process are :cpp:class:`adcc::MoSpaces`,
which collects information about the occupied and virtual
orbital spaces, and :cpp:class:`adcc::MoIndexTranslation`,
which maps orbitals indices between the ordering used by adccore
and the one used by the SCF program.

.. doxygengroup:: ReferenceObjects
   :members:
   :content-only:


Perturbation theory
-------------------
:cpp:class:`adcc::LazyMp` lazily computes second-order and third-order
Møller-Plesset perturbation theory on top of the reference
held by a :cpp:class:`adcc::ReferenceState`.

.. doxygengroup:: PerturbationTheory
   :members:
   :content-only:


AdcMatrix and matrix cores
--------------------------
:cpp:class:`adcc::AdcMatrix` sets up a representation of the ADC
matrix for a particular method. My the means of matrix cores,
which actually do the work, this allows to perform matrix-vector products
or access the diagonal of such a matrix under a common interface.

.. doxygengroup:: AdcMatrix
   :members:
   :content-only:


ADC guess setup
---------------
.. doxygengroup:: AdcGuess
   :members:
   :content-only:


ISR and one-particle densities
------------------------------
.. doxygengroup:: Properties
   :members:
   :content-only:


Tensor interface
----------------
The generalised :cpp:class:`adcc::Tensor` interface
used by adcc and adccore to perform tensor operations.

.. doxygengroup:: Tensor
   :members:
   :content-only:


Utilities
---------
.. doxygengroup:: Utilities
   :members:
   :content-only:


Metadata access
---------------
These classes and functions provide access to metadate about *adccore*.

.. doxygengroup:: Metadata
   :members:
   :content-only:


Tensor implementation using libtensor
-------------------------------------
This section describes the implementation of the
Tensor functionality of :cpp:class:`adcc::Tensor`
using the libtensor tensor library.

.. doxygengroup:: TensorLibtensor
   :members:
   :content-only:


adcman Davidson interface
-------------------------
.. doxygengroup:: AdcmanInterface
   :members:
   :content-only:
