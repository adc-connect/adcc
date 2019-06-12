# Functionality reference

This page contains a structured overview of the
`python` API of `adcc`.
See also the [full index](genindex).

## `run_adc` functions
```eval_rst
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

```


## Adc Middle layer
```eval_rst
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

```

## Tensor and symmetry interface
```eval_rst
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

.. autofunction:: adcc.add
.. autofunction:: adcc.contract
.. autofunction:: adcc.copy
.. autofunction:: adcc.divide
.. autofunction:: adcc.dot
.. autofunction:: adcc.empty_like
.. autofunction:: adcc.linear_combination
.. autofunction:: adcc.multiply
.. autofunction:: adcc.nosym_like
.. autofunction:: adcc.ones_like
.. autofunction:: adcc.subtract
.. autofunction:: adcc.transpose
.. autofunction:: adcc.zeros_like

```

## Properties
TODO

## Analysis
TODO

## Other stuff and utilities
```eval_rst
.. autofunction:: adcc.banner

```
