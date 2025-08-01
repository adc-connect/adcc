:github_url: https://github.com/adc-connect/adcc/blob/master/docs/theory.rst

Theoretical review of ADC methods
=================================

The polarisation propagator :math:`\Pi(\omega)` is a quantity from many-body
perturbation theory :cite:`Fetter1971`.
Its relationship to electronically excited states spectra can be understood
from the fact
that its poles are located exactly at the vertical excitation energies
:math:`\omega_n = E_n - E_0` :cite:`Fetter1971,Schirmer1982`.
Here, :math:`E_0` is the energy of the ground state of the
exact :math:`N`-electron Hamiltonian :math:`\Op{H}`,
and :math:`E_n` is the energy corresponding to excited state
:math:`\ket{\Psi_n}`.
The structure of :math:`\Pi(\omega)` close to these poles
depends both on the ground state :math:`\ket{\Psi_0}` and excited state
:math:`\ket{\Psi_n}`,
such that, e.g., transition properties :cite:`Schirmer1982,Schirmer2018`
may be extracted from :math:`\Pi(\omega)` as well.

Taking this as a starting point,
the algebraic-diagrammatic construction scheme
for the polarisation propagator (ADC)
examines an alternative representation of the
polarisation propagator :cite:`Schirmer1982`,
the so-called intermediate-state representation (ISR).
In this formalism, a set of creation and annihilation operators is
applied to the exact ground state
and the resulting precursor states are orthogonalised block-wise according
to excitation class :cite:`Schirmer1991`.
This procedure yields the so-called intermediate states
:math:`\left\{ \ket{\tilde{\Psi}_I}  \right\}_I`,
which are then employed to represent the polarisation propagator.
A careful inspection of the resulting expression for :math:`\Pi(\omega)`
and its poles allows to relate the
intermediate states (IS) to the excited states :math:`\ket{\Psi_n}`
of the Hamiltonian :cite:`Schirmer1991`
by a unitary transformation

.. math:: \ket{\Psi_n} = \sum_{I} X_{I,n} \ket{\tilde{\Psi}_I}.

The expansion coefficients :math:`\mat{X}` satisfy a Hermitian eigenvalue problem :cite:`Schirmer1991`

.. math::
   :label: eqn:adc_diagonalisation

   \mat{M} \mat{X} = \mat{\Omega} \mat{X}, \qquad \mat{X}^\dagger \mat{X} = \mat{I},

where :math:`\Omega_{nm} = \delta_{nm} \omega_n` is the diagonal matrix of excitation energies and
:math:`\mat{M}` is the so-called ADC matrix.
Its elements are directly accessible by representing the shifted Hamiltonian using IS, namely

.. math::

   M_{IJ} = \braket{\tilde{\Psi}_I}{\left(\Op{H} - E_0\right) \tilde{\Psi}_J}.

From the ADC eigenvectors :math:`\mat{X}` in the IS basis,
one may compute the density matrix :math:`\mat{\rho}^{n}`
for an excited state :math:`n` or the transition density matrices
:math:`\mat{\rho}^{n\leftarrow m}`
between state :math:`m` and :math:`n`,
in the molecular orbital (MO) basis :cite:`Schirmer2004,Wormit2014`.
Contracting these densities with the MO representation :math:`\mat{O}`
of a one-particle operator :math:`\Op{O}` allow computing arbitrary
state properties :math:`T^{n}`
or transition properties :math:`T^{n\leftarrow m}` through

.. math::

   \begin{aligned}
           T^{n} &= \braket{\Psi_n}{\Op{O} \Psi_n}
                   = \tr (\mat{O} \mat{\rho}^{n}) \\
           T^{n\leftarrow m} &= \braket{\Psi_n}{\Op{O} \Psi_m}
                   = \tr (\mat{O} \mat{\rho}^{n\leftarrow m}). \\
   \end{aligned}

In this way, e.g., the MO representation of the dipole operator
may be contracted with :math:`\mat{\rho}^{n\leftarrow m}` to
obtain the transition dipole moment between
states :math:`m` and :math:`n` and from this the oscillator strength.
Linear and non-linear molecular response properties,
e.g., static polarisabilities or two-photon absorption cross-sections,
are also accessible via this framework
:cite:`Trofimov2006,Knippenberg2012,Fransson2017`.

As described so far, the above formalism builds the IS basis on top of
the exact :math:`N`-electron ground state and is thus exact as well.
For practical calculations, however,
the ADC scheme is not applied to the exact ground state,
but to a Møller-Plesset ground state at order :math:`n`
of perturbation theory.
The resulting ADC method is named ADC(:math:`n`)
and is by construction consistent
with an MP(:math:`n`) ground state.
Detailed derivations and the resulting expressions for the ADC matrix :math:`\mat{M}`
as well as the aforementioned
densities :math:`\mat{\rho}^{n}` and :math:`\mat{\rho}^{n\leftarrow m}`
for various orders can be found in the
literature :cite:`Schirmer1982,Schirmer1991,Wormit2014,Dreuw2014,Schirmer2018`
and will not be discussed here.

.. list-table::

   * - .. figure:: images/matrix/adc_matrix_schematic.png
          :width: 200px

          Fig 1a. Schematic ADC matrix

     - .. figure:: images/matrix/matrix_water_adc2_sto3g.png
          :width: 200px

          Fig 1b. ADC(2) matrix of STO-3G water

     - .. figure:: images/matrix/matrix_water_adc3_sto3g.png
          :width: 200px

          Fig 1c. ADC(3) matrix of STO-3G water


As a result of the construction of ADC(:math:`n`) as excitations on top of
an MP(:math:`n`) ground state, the matrix :math:`\mat{M}`
exhibits a block structure, shown in Figure 1a.
In this the singles block is denoted :math:`M_{11}`,
the doubles block :math:`M_{22}` and the
coupling block :math:`M_{21}`.
One may construct perturbation expansions for the individual blocks as well.
For example in ADC(2) the lower-right :math:`M_{22}` block
is only present in zeroth order.
In ADC(3) on the other hand this block is present at first order,
which makes it consistent with an MP(3) ground state.
In contrast, ADC(2)-x is an \emph{ad hoc} modification of ADC(2),
where only the doubles-doubles block is treated first order like in ADC(3),
but the remaining blocks remain at the same order as in ADC(2) :cite:`Dreuw2014`.

On top of this block structure the individual blocks are sparse
as well, see Figure 1b and c.
This sparsity is a direct consequence of the selection rules obtained from
spin and permutational symmetry in the tensor contractions required
for computing :math:`\mat{M}`.
To exploit this sparsity when diagonalising
the matrix :eq:`eqn:adc_diagonalisation`,
\adcc follows the conventional approach :cite:`Dreuw2014,Wormit2014`
to use contraction-based, iterative
eigensolvers, such as the Jacobi-Davidson :cite:`Davidson1975`.
Furthermore, all tensor operations in the required ADC matrix-vector products
are performed on block-sparse tensors.
For an optimal performance the spin and permutational symmetry of the ADC equations
need to be taken into account when setting up the block tiling
along the tensor axes.
In this setting the computational scaling of ADC(2) is given as :math:`O(N^5)`
where :math:`N` is the number of orbitals,
whereas ADC(2)-x and ADC(3) scale as :math:`O(N^6)`.
This procedure additionally ensures the numerical stability of the eigensolver
with respect to the excitation manifold.
That is to say, that (for restricted references) spin-pure guess vectors
always lead to eigenvectors :math:`\mat{X}` from the same manifold,
such that the excitation manifold to probe can be reliably selected
via the guesses without employing a spin-adapted basis. :cite:`Dreuw2014`

One important modifications of the ADC scheme as discussed above
is the core-valence separation (CVS)
:cite:`Cederbaum1980,Trofimov2000,Wenzel2014b,Wenzel2014a,Wenzel2015`.
In this approximate ADC treatment targeting core-excited states,
the strong localisation of the core electrons
and the weak coupling between core-excited and valence-excited states
is exploited to decouple and discard the valence excitations from the ADC matrix.
This lowers the number of the actively treated orbitals and thus the
computational demand for solving the ADC eigenproblem :eq:`eqn:adc_diagonalisation`.
The validity of this approximation has been analysed in the literature
and is backed up by computational studies comparing with experiment
:cite:`Norman2018,Fransson2019`.
With this, ADC can be used for considering core-excited states,
and subsequent studies have also
established the ability of calculating non-resonant
X-ray emission spectra :cite:`Fransson2019`
and resonant inelastic X-ray scattering :cite:`Rehn2017a`.
Other variants of ADC include spin-flip :cite:`Lefrancois2015`,
where a modified Davidson guess allows treating processes of
simultaneous excitation and spin-flip, tackling few-reference problems
in an elegant and consistent way :cite:`Lefrancois2016,Lefrancois2017`.
Similar to other CI-like methods the range of orbitals which are considered
for building the intermediate states may also be artificially truncated.
For example, when considering valence-excitations,
excitations from the core orbitals may be
dropped leading to a frozen-core (FC) approximation.
Similarly, high-energy virtual orbitals may be left unpopulated,
leading to a frozen-virtual (FV)
or restricted-virtual approximation :cite:`Yang2017`.
