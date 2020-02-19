:github_url: https://github.com/adc-connect/adcc/blob/master/docs/reference.rst

.. _pyapi:

API reference
=============

.. note::
   Work in progress. Many function do not yet follow
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

libadcc: Python bindings for adccore
------------------------------------

The libadcc Python module contains the python bindings
of :ref:`adccore-layer`.
They are generated directly from the C++ source code
using `pybind11 <https://pybind11.readthedocs.io>`_
and the sources contained in the
`extension <https://github.com/adc-connect/adcc/tree/master/extension>`_
subfolder of the adcc github repository.

It is not recommended to call these functions directly,
but instead resort to the higher-level functionality
from the :ref:`adccmodule`.

.. automodapi:: libadcc
   :no-inheritance-diagram:
   :no-heading:
