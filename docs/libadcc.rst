:github_url: https://github.com/adc-connect/adcc/blob/master/docs/libadcc.rst

.. _libadcc:

libadcc: Python bindings for adccore
====================================

The libadcc Python module contains the python bindings
of :ref:`adccore-layer`.
They are generated directly from the C++ source code
using `pybind11 <https://pybind11.readthedocs.io>`_
and the sources contained in the
`extension <https://github.com/adc-connect/adcc/tree/master/extension>`_
subfolder of the adcc github repository.

It is not recommended to call these functions directly,
but instead resort to the higher-level functionality
from adcc. See the :ref:`full-reference` for details.

.. automodapi:: libadcc
