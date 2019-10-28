:github_url: https://github.com/adc-connect/adcc/blob/master/docs/hostprograms.rst

.. _hostprograms:

Connecting host programs to adcc
================================

Documentation for host programs and how to talk to adcc

Python dictionary or HDF5 file
------------------------------

.. autoclass:: adcc.DataHfProvider
    :members: __init__


.. note::
   TODO Have some examples


Host-program specific interface
-------------------------------

For implementing a host-program specific interface
to adcc, taking advantage of all features of the host program,
a derived class of the :class:`adcc.HartreeFockProvider` has to be implemented.
The interface for this is:

.. autoclass:: adcc.HartreeFockProvider
    :members:

.. note::
   TODO Explain the OperatorIntegralProvider and its mechanism.

.. note::
   TODO Point at examples in adcc source code

C++ interface
-------------
For directly passing data to *adccore* on the C++ level,
the following interface needs to be implemented:

.. doxygenclass:: adcc::HartreeFockSolution_i
   :members:
