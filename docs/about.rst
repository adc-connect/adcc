:github_url: https://github.com/adc-connect/adcc/blob/master/docs/about.rst

About this project
==================

The adcc project was initiated at Heidelberg University
from the frustration of the difficulties encountered when
starting ADC calculations from unusual
SCF references (in this case from the `molsturm <https://molsturm.org>`_ code).
In the initial phase of development adcc was heavily
inspired by the `adcman <http://doi.org/10.1080/00268976.2013.859313>`_ package
by Michael Wormit *et. al.*.
By the time of the first release in late 2019 the structure of adcc
has been completely rewritten and polished
and finally in late 2020 the code became fully open-source.

In its current form adcc features a strong focus on flexibility
for both developing novel ADC methods
and performing sophisticated analysis on top of obtained computational results.
Standard workflows like plotting spectra or visualising obtained results
are simple and often only take a single command,
but still provide access to underlying low-level objects to make
customisation possible.
As a result the project is accessible to novices to the field,
but still offers a low barrier to start developing more complex
simulation procedures, where this might be needed.
As numerous :ref:`adcc-related publications<publications>` prove
this philosophy has already enabled key advances
and successfully pushed the state of the art in computational spectroscopy methods.
A more detailed overview of adcc and its features can be found in :cite:`adcc`.


Citation
--------
We kindly ask all users of adcc, who find the package useful for their
research to cite the adcc paper :cite:`adcc` in their publications.


.. _contact-us:

Contact us
----------

You found a bug? Something is not working?
You have a great idea about a possible feature for adcc?
The primary place to discuss such issues about adcc is
`our issue page on github <https://github.com/adc-connect/adcc/issues>`_.
Alternatively you can also reach us by mail (``developers@adc-connect.org``).
