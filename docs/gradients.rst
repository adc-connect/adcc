:github_url: https://github.com/adc-connect/adcc/blob/master/docs/gradients.rst

.. _nuclear-gradients:

Nuclear gradients
=================

.. note::
   Analytic nuclear gradients in adcc are currently experimental and
   available for the PySCF backend.

adcc can evaluate analytic nuclear gradients of the ground-state (MP2) and
excited-state (ADC) energies. This page first shows how to request a gradient,
then documents the supported methods and finally describes the underlying
theory and the memory-bounded contraction algorithm.

Usage
-----

A nuclear gradient is requested via :py:func:`adcc.nuclear_gradient`. For an
excited state, pass the corresponding :py:class:`adcc.Excitation` object (e.g.
one element of ``state.excitations``):

.. code-block:: python

    from pyscf import gto, scf
    import adcc

    mol = gto.M(atom="O 0 0 0; H 0 0 1.795; H 1.693 0 -0.599",
                basis="cc-pvdz", unit="Bohr")
    scfres = scf.RHF(mol)
    scfres.kernel()

    state = adcc.adc2(scfres, n_singlets=3)
    grad = adcc.nuclear_gradient(state.excitations[0])
    print(grad.total)

For a ground-state (MP2) gradient, pass the :py:class:`adcc.LazyMp` object
instead:

.. code-block:: python

    mp = adcc.LazyMp(adcc.ReferenceState(scfres))
    grad = adcc.nuclear_gradient(mp)

The returned object exposes the total gradient via ``grad.total`` as well as
the individual one- and two-electron contributions.

Geometry optimisation scanners
------------------------------

For geometry optimisers it is useful to expose the whole PySCF/adcc/gradient
loop as a single callable.  :py:func:`adcc.nuclear_gradient_scanner` takes a
configured PySCF SCF object as its template and accepts Cartesian coordinates in
Bohr.  It returns ``(energy, gradient)`` in Hartree and Hartree/Bohr:

.. code-block:: python

    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-11
    scfres.conv_tol_grad = 1e-9

    scanner = adcc.nuclear_gradient_scanner(
        scfres,
        method="adc2",
        state_index=0,
        n_singlets=3,
        conv_tol=1e-7,  # forwarded unchanged to adcc.run_adc
    )

    energy, gradient = scanner(mol.atom_coords())

The scanner uses PySCF's scanner machinery to rerun the SCF calculation at each
new geometry with the original SCF object settings.  This keeps the SCF
interface on the PySCF object, where users already configure basis, charge,
spin/reference type, symmetry behaviour and convergence options.  For geomeTRIC
optimisations it is usually safest to construct the PySCF molecule with
``symmetry=False``.  PySCF's scanner also provides the SCF guess continuity
between geometry steps.  For excited states the scanner stores the previous
:class:`adcc.ExcitedStates` and selected :class:`adcc.Excitation`; by default it
follows the state by comparing AO-basis transition and state-difference densities
across geometries using PySCF AO cross-overlap integrals.  A fixed-index mode is
available via ``follow="index"`` for debugging or well-separated states.

The same object also implements geomeTRIC's custom-engine ``calc_new`` protocol:

.. code-block:: python

    result = scanner.calc_new(mol.atom_coords().ravel())
    # result == {"energy": energy, "gradient": gradient.ravel()}

geomeTRIC remains an optional dependency; the plain scanner only requires the
PySCF backend.  Minimum-energy crossing point workflows can be built from two
scanner targets because geomeTRIC's penalty-constrained formulation only needs
the two state energies and gradients, not derivative couplings.

The two-electron term can be evaluated with two different strategies, selected
through the ``eri_contraction`` keyword (see
:ref:`gradients-eri-contraction` below). For the PySCF backend the
memory-bounded ``"direct"`` strategy is chosen automatically; it can be
requested or overridden explicitly together with its tuning knobs:

.. code-block:: python

    grad = adcc.nuclear_gradient(
        state.excitations[0],
        eri_contraction="direct",          # or "full_ao"
        eri_pair_density_storage="hdf5",   # out-of-core packed density
        eri_pair_chunk_size=256,           # bound the working buffer
    )

.. _gradients-supported-methods:

Supported methods
-----------------

Analytic gradients are available for the following methods (for restricted and
unrestricted references, unless noted otherwise):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Method
     - Gradient
     - Notes
   * - MP2 (ground state)
     - ✓
     - via :py:class:`adcc.LazyMp`
   * - ADC(0), ADC(1)
     - ✓
     -
   * - ADC(2), ADC(2)-x
     - ✓
     -
   * - ADC(3)
     - ✓
     -
   * - CVS-ADC(0), CVS-ADC(1)
     - ✓
     - core-valence separation :cite:`Brumboiu2021`
   * - CVS-ADC(2), CVS-ADC(2)-x
     - ✓
     - core-valence separation :cite:`Brumboiu2021`
   * - CVS-ADC(3)
     - ✗
     - not available

Requesting a gradient for an unsupported method raises a
:py:exc:`NotImplementedError`.

.. _gradients-theory:

Theory and algorithm
--------------------

Gradient Lagrangian
~~~~~~~~~~~~~~~~~~~~

Analytic energy gradients for non-variational methods such as MP and ADC are
conveniently formulated within a Lagrangian formalism, a well-established
approach in quantum chemistry. The working equations implemented in adcc follow
the derivation of Rehn and Dreuw :cite:`Rehn2019`; the extension to
core-excited states within the core-valence separation is due to
Brumboiu *et al.* :cite:`Brumboiu2021`.

The gradient of the total energy with respect to a nuclear coordinate :math:`x`
separates into a one-electron and a two-electron contribution,

.. math::
   :label: eqn:gradient_total

   \frac{\mathrm{d} E}{\mathrm{d} x}
     = \sum_{\mu\nu} \gamma_{\mu\nu}\,
       \frac{\partial h_{\mu\nu}}{\partial x}
     - \sum_{\mu\nu} W_{\mu\nu}\,
       \frac{\partial S_{\mu\nu}}{\partial x}
     + \sum_{\mu\nu\kappa\lambda} \Gamma_{\mu\nu\kappa\lambda}\,
       \frac{\partial (\mu\nu|\kappa\lambda)}{\partial x},

where :math:`\mu,\nu,\kappa,\lambda` are atomic-orbital (AO) indices,
:math:`\gamma` is the relaxed one-particle density matrix (OPDM),
:math:`W` the energy-weighted density matrix,
:math:`S` the overlap matrix,
:math:`h` the one-electron Hamiltonian, and
:math:`\Gamma` the relaxed two-particle density matrix (TPDM).
The one-electron terms are cheap; the bottleneck is the contraction of the
TPDM with the derivative electron-repulsion integrals (ERIs)
:math:`\partial (\mu\nu|\kappa\lambda) / \partial x`.

.. _gradients-eri-contraction:

Two-electron contraction strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

adcc offers two strategies for the two-electron term, selected through the
``eri_contraction`` keyword of :py:func:`adcc.nuclear_gradient`.

``full_ao``
   The reference implementation. The MO-basis TPDM is transformed to two full
   AO-basis rank-4 tensors :math:`g^{(1)}` and :math:`g^{(2)}`, which are then
   contracted with the full derivative ERI tensor. Both the AO TPDMs and the
   derivative integrals scale as :math:`\mathcal{O}(N_\text{AO}^4)` in memory
   (the derivative ERIs even as :math:`3\,N_\text{AO}^4`), so this path becomes
   impractical already for moderate basis sets. It is kept as a
   validation/debug fallback.

``direct``
   The memory-bounded production path (default for PySCF). It never forms the
   full AO TPDMs nor the full derivative ERIs. Instead, the TPDM is transformed
   block-by-block directly into a *packed* AO-pair effective density, which is
   streamed against shell-batched derivative integrals. Peak memory is bounded
   by a single AO-pair chunk rather than by :math:`N_\text{AO}^4`.

The remaining subsections describe the ``direct`` algorithm.

The packed AO-pair effective density
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``full_ao`` path produces two AO TPDMs that are contracted with the ERIs in
chemists' notation as :math:`+(pr|qs)` and :math:`-(ps|qr)`, i.e.

.. math::
   :label: eqn:ao_tpdm

   g^{(1)}_{pqrs} &= \sum_{ijkl} C_{ip} C_{jq} \Gamma_{ijkl} C_{kr} C_{ls}, \\
   g^{(2)}_{pqrs} &= \sum_{ijkl} C_{ip} C_{jq} \Gamma_{ijkl} C_{ks} C_{lr},

summed over the spin cases that survive for a restricted reference
(:math:`g^{(1)}`: ``aaaa``, ``bbbb``, ``abab``, ``baba``;
:math:`g^{(2)}`: ``aaaa``, ``bbbb``, ``abba``, ``baab``).
Here :math:`i,j,k,l` are spin-orbital MO indices and :math:`C` are the
spin-blocked MO coefficients.

The ``direct`` path instead builds the combined effective density only in the
layout actually required for the integral contraction. Defining the effective
density as the difference of the two orderings,

.. math::
   :label: eqn:effective_density

   D_{pr,qs} = g^{(1)}_{pqrs} - g^{(2)}_{pqrs},

the two leading AO indices :math:`p,r` are kept dense while the trailing pair
:math:`q,s` is stored in lower-triangular packed form. With the symmetric pair
identified by :math:`(\max(q,s), \min(q,s))` and mapped to the packed index

.. math::
   :label: eqn:pair_index

   m(q,s) = \frac{\max(q,s)\,[\max(q,s)+1]}{2} + \min(q,s),

the packed density has shape
:math:`(N_\text{AO},\, N_\text{AO},\, N_\text{pair})` with
:math:`N_\text{pair} = N_\text{AO}(N_\text{AO}+1)/2`.
This is the conventional symmetric-pair packing used by derivative-ERI
backends that expose only one integral per symmetric ket pair (e.g. PySCF
``aosym="s2kl"``). Because only one entry is stored for an off-diagonal pair
:math:`q \neq s`, the packed entry accumulates both contributing orderings.

Block-wise streaming transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The key observation is that each MO axis of the spin-orbital TPDM splits into a
leading alpha range :math:`[0, n_\alpha)` and a trailing beta range
:math:`[n_\alpha, n_\text{orb})`, and the spin-blocked coefficients are
non-zero only on the matching range. For every required spin case adcc
therefore extracts just the relevant rank-4 spin sub-block of a TPDM block
(via the ``export_block`` method of :py:class:`adcc.Tensor`) and contracts it
with the compacted (non-zero-row) coefficients, never densifying the full
zero-padded block.

For a chunk of AO pairs the central contraction reads, for the direct ordering,

.. math::
   :label: eqn:direct_transform

   D_{pr,m} \mathrel{+}= \sum_{ijkl}
       C_{ip}\, C_{kr}\, \Gamma_{ijkl}\,
       \left( C_{j\,q(m)}\, C_{l\,s(m)} \right),

and analogously for the exchange ordering, where :math:`q(m)` and :math:`s(m)`
are the AO indices of packed pair :math:`m`. The intermediate
:math:`C_{j\,q(m)} C_{l\,s(m)}` is built once per chunk and contracted with the
sub-block, so the working memory is bounded by the chunk size
(the number of AO pairs processed at once) rather than by the full
:math:`N_\text{pair}`.

Contraction with derivative integrals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The packed density is then contracted with the shell-batched derivative ERIs.
PySCF only packs the ket pair (``aosym="s2kl"``), so for a derivative integral
block :math:`E_{ab,qs} = \partial (ab|qs)/\partial x` restricted to the AO
shells of one atom, the symmetrised contraction effectively evaluates

.. math::
   :label: eqn:eri_contraction

   \frac{\partial E_\text{2e}}{\partial x_A} = \sum_{ab}\sum_{q\le s}
       E^{A}_{ab,qs}\,
       \left( D_{ab,qs} + D_{ba,sq} + D_{qs,ab} + D_{sq,ba} \right),

with the four terms accounting for the bra/ket permutational symmetry of the
ERIs. Keeping the shell slices atom-local ensures that atom AO ranges and shell
ranges are never mixed accidentally.

Memory and storage options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``direct`` path exposes two knobs on :py:func:`adcc.nuclear_gradient`:

* ``eri_pair_chunk_size`` -- the number of AO pairs transformed at once. This
  bounds the in-memory working buffer of the transform. When left unset it
  defaults to the full :math:`N_\text{pair}` for in-memory storage, and to a
  smaller bounded value for out-of-core storage.
* ``eri_pair_density_storage`` -- where the packed AO-pair density is kept:

  - ``"memory"`` (default): a single in-memory
    :math:`(N_\text{AO}, N_\text{AO}, N_\text{pair})` array.
  - ``"hdf5"``: a temporary HDF5 file under the scratch directory (or
    ``PYSCF_TMPDIR`` / the system temporary directory), removed after the
    contraction. This bounds the packed *output* density; the transform's
    working buffer is bounded separately by ``eri_pair_chunk_size``.
