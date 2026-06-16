:github_url: https://github.com/adc-connect/adcc/blob/master/docs/topics.rst

.. _topics:

Overview of adcc
================

.. note::
   Work in progress. Should be expanded.

.. note::
   For each logical section of adcc, some nice overview should be given
   in a separate page.

For documentation how to connect host programs to adcc,
see :ref:`hostprograms`.

.. _adcn-methods:

The adcc.adcN family of methods
-------------------------------

.. autofunction:: adcc.run_adc

.. py:function:: adcc.adc0(data_or_matrix, **kwargs)

    Run an ADC(0) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.adc1(data_or_matrix, **kwargs)

    Run an ADC(1) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.adc2(data_or_matrix, **kwargs)

    Run an ADC(2) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.adc2x(data_or_matrix, **kwargs)

    Run an ADC(2)-x calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.adc3(data_or_matrix, **kwargs)

    Run an ADC(3) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.cvs_adc0(data_or_matrix, **kwargs)

    Run an CVS-ADC(0) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.cvs_adc1(data_or_matrix, **kwargs)

    Run an CVS-ADC(1) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.cvs_adc2(data_or_matrix, **kwargs)

    Run an CVS-ADC(2) calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.cvs_adc2x(data_or_matrix, **kwargs)

    Run an CVS-ADC(2)-x calculation. For more details see :py:func:`adcc.run_adc`.

.. py:function:: adcc.cvs_adc3(data_or_matrix, **kwargs)

    Run an CVS-ADC(3) calculation. For more details see :py:func:`adcc.run_adc`.

.. autoclass:: adcc.ExcitedStates
    :members:

Visualisation
-------------

.. automodule:: adcc.visualisation
    :members:
    :inherited-members:


Adc Middle layer
----------------

.. autoclass:: adcc.AdcMatrix
    :members:
    :inherited-members:
    :undoc-members:

.. autoclass:: adcc.AdcMethod
    :members:
    :undoc-members:

.. autoclass:: adcc.ReferenceState
    :members:
    :inherited-members:
    :undoc-members:

.. autoclass:: adcc.LazyMp
    :members:
    :inherited-members:
    :undoc-members:

Tensor and symmetry interface
-----------------------------

.. autoclass:: adcc.Tensor
    :members:
    :inherited-members:
    :undoc-members:

.. autoclass:: adcc.Symmetry
    :members:
    :inherited-members:

.. autoclass:: adcc.AmplitudeVector
    :members:
    :inherited-members:

.. autofunction:: adcc.copy
.. autofunction:: adcc.dot
.. autofunction:: adcc.einsum
.. autofunction:: adcc.lincomb
.. autofunction:: adcc.empty_like
.. autofunction:: adcc.nosym_like
.. autofunction:: adcc.ones_like
.. autofunction:: adcc.zeros_like
.. autofunction:: adcc.evaluate
.. autofunction:: adcc.direct_sum

Solvers
-------

.. automodule:: adcc.solver.davidson
    :members:

.. automodule:: adcc.solver.lanczos
    :members:

.. automodule:: adcc.solver.power_method
    :members:

Adc Equations
-------------

.. automodule:: adcc.adc_pp
    :members:

.. automodule:: adcc.adc_pp.matrix
    :members:


Utilities
---------

.. autofunction:: adcc.banner
