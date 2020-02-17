:github_url: https://github.com/adc-connect/adcc/blob/master/docs/benchmarks.rst

.. _benchmarks:

Benchmarks and timings
======================

This page provides an overview of the time and memory requirements
of running ADC calculations based on a few benchmark cases.

The results have been generated with our automated
benchmarking suite `adcc-bench <https://github.com/adc-connect/adcc-bench>`_,
which is based on `airspeed-velocity <https://github.com/airspeed-velocity/asv>`_.
The benchmarks of adcc-bench are run periodically on the master branch
of adcc in order to track performance of adcc across releases.

.. include:: benchmarks/commit.rst

Cluster benchmarks
------------------

These benchmarks have been run on a machine with two
Intel(R) Xeon(R) E5-2690 CPUs @ 2.90GHz using various
numbers of threads.

*p*-Nitroaniline
~~~~~~~~~~~~~~~~

.. include:: benchmarks/ParaNitroAniline.rst

Phosphine core excited states
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: benchmarks/PhosphineCvs.rst

Methylammonium radical
~~~~~~~~~~~~~~~~~~~~~~

.. include:: benchmarks/MethylammoniumRadical.rst

Water
~~~~~

.. include:: benchmarks/WaterExpensive.rst
