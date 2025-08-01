:github_url: https://github.com/adc-connect/adcc/blob/master/docs/hostprograms.rst

.. _hostprograms:

Connecting host programs to adcc
================================

Documentation for host programs and how to talk to adcc

Python dictionary or HDF5 file
------------------------------

.. autoclass:: adcc.DataHfProvider
    :members: __init__


Host-program specific interface
-------------------------------

For implementing a host-program specific interface
to adcc, taking advantage of all features of the host program,
a derived class of the :class:`adcc.HartreeFockProvider` has to be implemented.
The interface for this is:

.. autoclass:: adcc.HartreeFockProvider
    :members:

Examples in the adcc source code for these interfaces are
located in the
`adcc/backend folder <https://github.com/adc-connect/adcc/tree/master/adcc/backends>`_.
For example `pyscf.py <https://github.com/adc-connect/adcc/blob/master/adcc/backends/pyscf.py>`_
or `psi4.py <https://github.com/adc-connect/adcc/tree/master/adcc/backends/psi4.py>`_.

.. note::
   TODO Explain the OperatorIntegralProvider and its mechanism.

C++ interface
-------------
For directly passing data to *libadcc* on the C++ level,
the following interface needs to be implemented:

.. doxygenclass:: libadcc::HartreeFockSolution_i
   :members:
