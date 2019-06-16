# Performing calculations with adcc

This section gives a practical guide for performing ADC calculations with `adcc`.
It deliberately does not show all tricks and all tweaks,
but instead provides a working man's subset of most important features.
To checkout the full API with all details of the mentioned functions and
classes, see the [Full adcc reference](reference.md)'

## Overview of supported features
Currently `adcc` supports all ADC(n) variants up to level 3,
that is **ADC(0)**, **ADC(1)**, **ADC(2)**, **ADC(2)-x** and **ADC(3)**.
For each of these methods, basic state properties and transition properties
such as the state dipole moments or the oscillator strengths are available.
More complicated analysis can be performed in user code by requesting
the full state and transition density matrices as `numpy` arrays.
The code supports the **spin-flip** variant of all aforementioned methods
and furthermore allows the **core-valence separation** to be applied
for all methods (except ADC(3)).

## General ADC(n) calculations
General ADC(n) calculations,
that is calculations without any additional approximations,
are invoked on a SCF reference by passing it
to a function from `adcc`, which resembles the name of the method.
In this sense `adcc.adc0` performs an ADC(0) calculation,
whereas `adcc.adc3` performs an ADC(3) calculation.
The distinction between ADC(2) and ADC(2)-x is made
by using either `adcc.adc2` or `adcc.adc2x`.

Let us return to our [introductory example](index.md),
where we performed a cc-pVTZ ADC(3) calculation of water:
```python
from pyscf import gto, scf
import adcc

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvtz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

# Run an ADC(3) calculation, solving for 3 singlets
state = adcc.adc3(scfres, n_singlets=3)
```
If we run this code, we first get some output from `pyscf`
like `converged SCF energy = -76.0571904154804`
and afterwards see a convergence table such as
```
Starting adc3  singlet Jacobi-Davidson ...
Niter n_ss  max_residual  time  Ritz values
  1     6       0.28723   2.5s  [0.42804006 0.50764932 0.50832029]
  2    12      0.036665   5.2s  [0.31292983 0.38861435 0.39597989]
  3    18     0.0022763   4.4s  [0.30538169 0.37872815 0.38776093]
  4    24    0.00076294   5.2s  [0.30444427 0.3770635  0.38676531]
  5    30    4.1766e-05   5.2s  [0.30432968 0.37681918 0.38663898]
=== Restart ===
  6    12    6.6038e-06   5.0s  [0.30432361 0.37680258 0.38663161]
  7    18    6.7608e-07   5.4s  [0.30432287 0.37679979 0.38663062]
=== Converged ===
    Number of matrix applies:    84
    Total solver time:            32s 818ms
```
There are a few things to note here:

- There was no need to explicitly pass any information
  about the molecular geometry or the basis set directly to `adcc`.
  The only thing `adcc` needs to get going is the *converged* SCF result
  contained in the `scfres` object in the above code example.
- Apart from the SCF result in `scfref`, the `adcc.adc3` method takes
  extra keyword arguments such as `n_singlets` in this case. These arguments
  allow to specify, which and how many states to compute, how accurate
  this should be done and which algorithms to use.
  These arguments will be discussed in detail in this section.
- The Jacobi-Davidson convergence table allows to monitor the convergence
  as the calculation proceeds. `n_ss` refers to the number of vectors
  in the subspace. The more vectors in the subspace, the more costly
  a single iteration is, but the faster the calculation typically
  converges. The implementation in `adcc` makes a compromise,
  by shrinking the subspace (called a `== Restart ==`) after a few
  iterations. The `max_residual` provides a measure for the
  remaining numerical error. `time` gives a rough idea for the
  time needed for the displayed iteration.
  Lastly `Ritz values` provides the current estimates to the excitation
  energies (in Hartree). Only the first few requested excitations
  are displayed here.
- The final lines inform about the number of times the ADC(3)
  matrix had to be applied to some vectors (i.e. the number of
  matrix-vector products with the ADC matrix, which had to be
  computed). It also shows the total time for the Jacobi-Davidson solver in order
  to converge the requested states. Typically the runtime is directly
  related to the number of such applies and this number should therefore
  be used when trying to identify a suitable set of `adcc` parameters for converging
  a calculation.

There is of course no need to use `pyscf` for the Hartree-Fock reference. We could have done
exactly the same thing using `psi4` as the SCF driver, e.g.
```python
import psi4

# Run SCF in Psi4
mol = psi4.geometry("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
""")
psi4.core.be_quiet()
psi4.set_options({'basis': "cc-pvtz", 'e_convergence': 1e-13, 'd_convergence': 1e-7})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run an ADC(3) calculation in adcc, solving for 3 singlets
state = adcc.adc3(wfn, n_singlets=3)
```
which will give rise to a similar Davidson convergence than before.
In either case the `state` object, which was returned
from the `adcc.adc3` function now contains the resulting states
and can be used to compute excited states properties
or analyse the states further, see the section on properties below.
A good summary about the states is available using the `describe()`
method, like so:
```
print(state.describe())
```
which returns a table such as
```
+-----------------------------------------------------+
| adc3                           singlet ,  converged |
+-----------------------------------------------------+
|  #        excitation energy       |v1|^2    |v2|^2  |
|          (au)           (eV)                        |
|  0     0.3043229      8.281047    0.9428   0.05721  |
|  1     0.3767998      10.25324    0.9449   0.05514  |
|  2     0.3866306      10.52076    0.9418   0.05823  |
+-----------------------------------------------------+
```
which describes the contained states by their
`excitation energy` in Hartree and electron volts as well as
the square norm of the singles (`|v1|^2`) and doubles (`|v2|^2`)
parts of the corresponding excitation vectors.

Without a doubt, ADC(3) is a rather expensive method,
taking already noticable time for a simple system such as
a triple zeta water calculation. For comparison an equivalent ADC(1) calculation,
started with
```python
state = adcc.adc1(scfres, n_singlets=3)
```
on top of the same `pyscf` reference state, gives rise to
```
Starting adc1  singlet Jacobi-Davidson ...
Niter n_ss  max_residual  time  Ritz values
  1     6       0.01356  116ms  [0.355402   0.43416334 0.43531311]
  2    12     0.0019488   41ms  [0.33653051 0.40287876 0.41843608]
  3    18    1.9961e-05   65ms  [0.33603959 0.40167202 0.41791942]
  4    24    2.7046e-07   69ms  [0.33603543 0.40166584 0.41791101]
=== Converged ===
    Number of matrix applies:    48
    Total solver time:           295.018ms
```
on the same machine, i.e. is both faster per iteration
and needs less iterations in total.
Other means to influence the calculation runtime
and determine the number and kind of states to compute
is discussed in the next section.

### Calculation parameters
The `adcc.adcN` family of methods
(such as `adcc.adc1` and `adcc.adc3` above)
each take a number of arguments:

- **n_singlets**, **n_triplets** and **n_states**
  control the number and kind of states to compute.
  `n_singlets` and `n_triplets` are only available for restricted
  references and ensure to only obtain singlets or triplets in the ADC
  calculations. `n_states` is available for all references and does
  not impose such a restriction. E.g.
  ```python
  state = adcc.adc2(scfres, n_singlets=6)
  ```
  would compute six excited states, which could have any spin.
  In the case of unrestricted references they will most likely
  not be spin-pure.
- **conv_tol** (convergence tolerance)
  specifies the tolerance on the `max_residual`
  in the Jacobi-Davidson scheme. It thus influences the numerical
  accuracy of the calculations. More accurate calculations take
  longer, especially to reach tolerances below `1e-8` can become very slow.
  The default value is `1e-6`, which is usually
  a good compromise between accuracy and runtime.
- **max_subspace** (maximal subspace size)
  specifies the maximal number of subspace vectors in the Jacobi-Davidson
  scheme before a restart occurs. The defaults are usually good,
  but do not be shy to increase this value if you encounter convergence problems.
- **n_guesses** (Number of guess vectors):
  By default `adcc` uses twice as many guess vectors as states to be computed.
  Sometimes increasing this value by a few vectors can be helpful.
  If you encounter a convergence to zero eigenvalues, than decreasing this
  parameter might solve the problems.
- **max_iter** (Maximal number of iterations)
  The default value (70) should be good in most cases. If convergence
  does not happen after this number of iterations, then usually something
  is wrong anyway and the other parameters should be adjusted.
- **output**: Providing a parameter `output=None` silences the ADC run
  (apart from warnings and errors) and only returns the converged result.
  For example:
  ```python
  state = adcc.adc2(scfres, n_singlets=3, output=None)
  ```

### Reusing intermediate data
Since solving the ADC equations can be very costly
various intermediates are only computed once and stored in memory.
For performing a second ADC calculation for the identical system,
it is thus wise to re-use this data as much as possible.

A very common use case is to compute singlets *and* triplets
on top of a restricted reference.
In order to achieve this with maximal data reuse,
one can use the following pattern:
```python
singlets = adcc.adc2(scfres, n_singlets=3)
triplets = adcc.adc2(singlets.matrix, n_triplets=5)
```
This will perform both an ADC(2) calculation for 3 singlets
as well as 5 triplets on top of the HF reference in `scfres`
by using the ADC(2) matrix stored in the `singlets.matrix` attribute
along with its intermediates.

If the ADC method is to be varied between
the first and the second run, one may at least reuse the
Møller-Plesset ground state, like so
```python
adc2_state = adcc.adc2(scfres, n_singlets=3)
adc2x_state = adcc.adc2x(adc2_state.ground_state, n_singlets=3)
```
which computes 3 singlets both at ADC(2) and ADC(2)-x level.
A slightly improved convergence of the second ADC(2)-x calculation
can be achieved, if we exploit the similarity of ADC(2) and ADC(2)-x
and use the eigenvectors from ADC(2) as the guess vectors for ADC(2)-x.
This can be achieved using the `guesses` parameter:
```python
adc2_state = adcc.adc2(scfres, n_singlets=3)
adc2x_state = adcc.adc2x(adc2_state.ground_state, n_singlets=3,
                         guesses=adc2_state.eigenvectors)
```

This trick of course can also be used to tighten a
previous ADC result in case a smaller convergence tolerance is needed,
e.g.
```python
# Only do a crude solve first
state = adcc.adc2(scfres, n_singlets=3, conv_tol=1e-3)

# Inspect state and get some idea what's going on
# ...

# Now converge tighter, using the previous result
state = adcc.adc2(state.matrix, n_singlets=3, conv_tol=1e-7,
                  guesses=state.eigenvectors)
```

## Spin-flip calculations
```note:: Describe: What is spin-flip? Why?

```
Two things need to be changed in order to run a spin-flip calculation with `adcc`.
Firstly, a triplet Hartree-Fock reference should be employed
and secondly, instead of using the `n_states` or `n_singlets` parameter,
one uses the special parameter `n_spin_flip` instead to specify the number
of states to be computed. An example for using `pyscf` to
compute the spin-flip ADC(2)-x states of hydrogen fluoride near the
dissociation limit.
```python
import adcc
from pyscf import gto, scf

# Run SCF in pyscf aiming for a triplet
mol = gto.M(
    atom='H 0 0 0;'
         'F 0 0 3.0',
    basis='6-31G',
    unit="Bohr",
    spin=2  # =2S, ergo triplet
)
scfres = scf.UHF(mol)
scfres.conv_tol = 1e-13
scfres.kernel()

# Run ADC(2)-x with spin-flip
states = adcc.adc2x(scfres, n_spin_flip=5)
print(states.describe())
```

## Core-valence-separated calculations
```note:: Describe: What is CVS? Why?

```
For performing core-valence separated calculations,
`adcc` adds the prefix `cvs_` to the method functions discussed already above.
In other words, running a CVS-ADC(2)-x calculation can be achieved
using `cvs_adc2x`, a CVS-ADC(1) calculation using `cvs_adc1`.
Such a calculation requires one additional parameter,
namely `n_core_orbitals`, which determines the number of **spatial** orbitals
to put into the core space. This is to say, that `n_core_orbitals=1` will
not just place one orbital into the core space,
much rather one alpha and one beta orbital. Similarly `n_core_orbitals=2`
places two alphas and two betas into the core space and so on.

For example, in order to perform a CVS-ADC(2) calculation of water,
which places the oxygen 1s core electrons into the core space,
we need to run the code
```python
import psi4

# Run SCF in Psi4
mol = psi4.geometry("""
    O 0 0 0
    H 0 0 1.795239827225189
    H 1.693194615993441 0 -0.599043184453037
    symmetry c1
    units au
""")
psi4.core.be_quiet()
psi4.set_options({'basis': "cc-pvtz", 'e_convergence': 1e-13, 'd_convergence': 1e-7})
scf_e, wfn = psi4.energy('SCF', return_wfn=True)

# Run CVS-ADC(2) solving for 4 singlet excitations of the oxygen 1s
states = adcc.cvs_adc2(wfn, n_singlets=4, n_core_orbitals=1)
```


## Property calculations
```note:: This section should be written.

```


## Further examples and details
Some further examples can be found in the `examples` folder
of the [`adcc` code repository](https://code.adc-connect.org/examples).
```eval_rst
For more details about the calculation parameters,
see the reference for :py:func:`adcc.run_adc`.

```
