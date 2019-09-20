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

The interface to be implemented in general is:

.. autoclass:: adcc.HartreeFockProvider
    :members:

.. note::
   Point at examples in adcc source code

C++ interface
-------------
For directly passing data to *adccore* on the C++ level,
the following interface needs to be implemented:

.. doxygenclass:: adcc::HartreeFockSolution_i
   :members:
