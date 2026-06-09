from __future__ import annotations
import numpy
import pybind11_stubgen.typing_ext
import typing

__all__: list[str] = [
    "AdcMemory",
    "HartreeFockProvider",
    "HartreeFockSolution_i",
    "MoIndexTranslation",
    "MoSpaces",
    "ReferenceState",
    "Symmetry",
    "Tensor",
    "amplitude_vector_enforce_spin_kind",
    "direct_sum",
    "evaluate",
    "fill_pp_doubles_guesses",
    "get_n_threads",
    "get_n_threads_total",
    "linear_combination_strict",
    "make_symmetry_eri",
    "make_symmetry_operator",
    "make_symmetry_operator_basis",
    "make_symmetry_orbital_coefficients",
    "make_symmetry_orbital_energies",
    "make_symmetry_triples",
    "set_n_threads",
    "set_n_threads_total",
    "tensordot",
    "trace",
]

class AdcMemory:
    """
    Class controlling the memory allocations for adcc ADC calculations. Python binding to :cpp:class:`libadcc::AdcMemory`.
    """
    def __init__(self) -> None: ...
    def __repr__(self) -> str: ...
    def initialise(
        self, pagefile_directory: str, max_block_size: int, allocator: str
    ) -> None: ...
    @property
    def allocator(self) -> str:
        """
        Return the allocator to which the class is initialised.
        """
    @property
    def contraction_batch_size(self) -> int:
        """
        Get or set the batch size for contraction, i.e. the number of elements handled simultaneously in a tensor contraction.
        """
    @contraction_batch_size.setter
    def contraction_batch_size(self, arg1: int) -> None: ...
    @property
    def max_block_size(self) -> int:
        """
        Return the maximal block size a tenor may have along each axis.
        """
    @property
    def pagefile_directory(self) -> str:
        """
        Return the pagefile_directory value:
        Note: This value is only meaningful if allocator != "standard"
        """

class HartreeFockProvider(HartreeFockSolution_i):
    """
    Abstract class defining the interface for passing data from the host program to adcc. All functions of this class need to be overwritten explicitly from python.
    In the remaining documentation we denote with `nf` the value returned by `get_n_orbs_alpha()` and with `nb` the value returned by `get_nbas()`.
    """
    def __init__(self) -> None: ...
    def fill_eri_ffff(self, arg0: tuple, arg1: numpy.ndarray) -> None:
        """
        Fill the passed numpy array `arg1` with a part of the electron-repulsion integral tensor in the molecular orbital basis. The indexing convention is the chemist's notation, i.e. the index tuple `(i,j,k,l)` refers to the integral :math:`(ij|kl)`. The block to store is specified by the provided tuple of ranges `arg0`, which gives the range of indices to place into the buffer along each of the axis. The index counting is done in spin orbitals, so the full range in each axis is `range(0, 2 * nf)`.
        """
    def fill_eri_phys_asym_ffff(self, arg0: tuple, arg1: numpy.ndarray) -> None:
        """
        Fill the passed numpy array `arg1` with a part of the **antisymmetrised** electron-repulsion integral tensor in the molecular orbital basis. The indexing convention is the physicist's notation, i.e. the index tuple `(i,j,k,l)` refers to the integral :math:`\\langle ij||kl \\rangle`. The block to store is specified by the provided tuple of ranges `arg0`, which gives the range of indices to place into the buffer along each of the axis. The index counting is done in spin orbitals, so the full range in each axis is `range(0, 2 * nf)`.
        """
    def fill_fock_ff(self, arg0: tuple, arg1: numpy.ndarray) -> None:
        """
        Fill the passed numpy array `arg1` with a part of the Fock matrix in the molecular orbital basis. The block to store is specified by the provided tuple of ranges `arg0`, which gives the range of indices to place into the buffer along each of the axis. The index counting is done in spin orbitals, so the full range in each axis is `range(0, 2 * nf)`. The implementation should not assume that the alpha-beta and beta-alpha blocks are not accessed even though they are zero by spin symmetry.
        """
    def fill_occupation_f(self, arg0: numpy.ndarray) -> None:
        """
        Fill the passed numpy array of size `(2 * nf, )` with the occupation number for each SCF orbital.
        """
    def fill_orbcoeff_fb(self, arg0: numpy.ndarray) -> None:
        """
        Fill the passed numpy array of size `(2 * nf, nb)` with the SCF orbital coefficients, i.e. the uniform transform from the one-particle basis to the molecular orbitals.
        """
    def fill_orben_f(self, arg0: numpy.ndarray) -> None:
        """
        Fill the passed numpy array of size `(2 * nf, )` with the SCF orbital energies.
        """
    def flush_cache(self) -> None:
        """
        This function is called to signal that potential cached data could now be flushed to save memory or other resources.
        This can be used to purge e.g. intermediates for the computation of electron-repulsion integral tensor data.
        """
    def get_conv_tol(self) -> float:
        """
        Returns the tolerance value used for SCF convergence. Should be roughly equivalent to the l2 norm of the Pulay error.
        """
    def get_energy_scf(self) -> float:
        """
        Returns the final total SCF energy (sum of electronic and nuclear terms.
        """
    def get_n_bas(self) -> int:
        """
        Returns the number of *spatial* one-electron basis functions. This value is abbreviated by `nb` in the documentation.
        """
    def get_n_orbs_alpha(self) -> int:
        """
        Returns the number of HF *spin* orbitals of alpha spin. It is assumed the same number of beta spin orbitals are used. This value is abbreviated by `nf` in the documentation.
        """
    def get_nuclear_multipole(
        self, arg0: int, arg1: tuple
    ) -> numpy.ndarray[numpy.float64]:
        """
        Returns the nuclear multipole of the requested order. For `0` returns the total nuclear charge as an array of size 1, for `1` returns the nuclear dipole moment as an array of size 3.
        """
    def get_restricted(self) -> bool:
        """
        Return *True* for a restricted SCF calculation, *False* otherwise.
        """
    def get_spin_multiplicity(self) -> int:
        """
        Returns the spin multiplicity of the HF ground state. A value of 0* (for unknown) should be supplied for unrestricted calculations.
        """
    def has_eri_phys_asym_ffff(self) -> bool:
        """
        Returns whether `fill_eri_phys_asym_ffff` function is implemented and should be used(*True*) or whether antisymmetrisation should be done inside adcc starting from the `fill_eri_ffff` function (*False*)
        """
    def transform_gauge_origin_to_xyz(self, arg0: str) -> tuple:
        """
        Transforms a string specifying the gauge origin to a tuple containing the x, y, z Cartesian components.
        """

class HartreeFockSolution_i:
    """
    Interface class representing the data expected in adcc from an interfacing HF / SCF program. Python binding to :cpp:class:`HartreeFockSolution_i`
    """
    @property
    def backend(self) -> str: ...
    @property
    def conv_tol(self) -> float: ...
    @property
    def energy_scf(self) -> float: ...
    @property
    def fock_ff(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def n_alpha(self) -> int: ...
    @property
    def n_bas(self) -> int: ...
    @property
    def n_beta(self) -> int: ...
    @property
    def n_orbs(self) -> int: ...
    @property
    def n_orbs_alpha(self) -> int: ...
    @property
    def n_orbs_beta(self) -> int: ...
    @property
    def occupation_f(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def orbcoeff_fb(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def orben_f(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def restricted(self) -> bool: ...
    @property
    def spin_multiplicity(self) -> int: ...

class MoIndexTranslation:
    """
    Helper object to extract information from indices into orbitals subspaces and to map them between different indexing conventions (full MO space, MO subspaces, indexing convention in the HF Provider / SCF host program, ... Python binding to :cpp:class:`libadcc::MoIndexTranslation`.
    """
    @typing.overload
    def __init__(self, arg0: MoSpaces, arg1: str) -> None:
        """
        Construct a MoIndexTranslation class from an MoSpaces object and the identifier for the space (e.g. o1o1, v1o1, o3v2o1v1, ...)
        """
    @typing.overload
    def __init__(self, arg0: MoSpaces, arg1: list[str]) -> None:
        """
        Construct a MoIndexTranslation class from an MoSpaces object and the list of identifiers for the space (e.g. ["o1", "o1"] ...)
        """
    def block_index_of(self, arg0: tuple) -> tuple:
        """
        Get the block index of an index, i.e. get the index which points to the block of the tensor in which the element with the passed index is contained in.
        """
    def block_index_spatial_of(self, arg0: tuple) -> tuple:
        """
        Get the spatial block index of an index

        The spatial block index is the result of block_index_of modulo the spin blocks,
        i.e. it maps an index onto the index of the *spatial* blocks only, such that
        the resulting value is identical for two index where the MOs only differ
        by spin. For example the 1st core alpha and the 1st core beta orbital will
        map to the same value upon a call of this function.
        """
    @typing.overload
    def combine(self, arg0: tuple, arg1: tuple) -> tuple:
        """
        Combine a block index and an in-block index into the appropriate index. Effectively undoes the effect of 'split'.
        """
    @typing.overload
    def combine(self, arg0: str, arg1: tuple, arg2: tuple) -> tuple:
        """
        Combine a spin block (given as a string of 'a's or 'b's), a spatial-only block index and an in-block index into the appropriate index. Essentially undoes the effect of 'spin_of', 'block_index_spatial_of' and 'inblock_index_of'.
        """
    def full_index_of(self, arg0: tuple) -> tuple:
        """
        Map an index given in the space, which was passed upon construction, to the corresponding index in the full MO index range (the ffff space).
        """
    def hf_provider_index_of(self, arg0: tuple) -> tuple:
        """
        Map an index (given in the space passed upon construction) to the indexing convention of the host program provided to adcc as the HF provider.
        """
    def inblock_index_of(self, arg0: tuple) -> tuple:
        """
        Get the in-block index, i.e. the index within the tensor block.
        """
    def map_range_to_hf_provider(self, arg0: tuple) -> list:
        """
        Map a range of indices to host program indices, i.e. the indexing convention
        used in the HfProvider, which provides the SCF data to adcc.

        Since the mapping between subspace and host program indices might not be contiguous,
        a list of pairs of ranges is returned. In each pair, the first entry represents a
        range of indices (indexed in the MO subspace) and the second entry represents
        the equivalent range of indices in the Hartree-Fock provider these are mapped to.

          ranges    Tuple of pairs of indices: One index pair for each dimension. Each
                    pair describes the range of indices along one axis, which should be
                    mapped to the indexing convention of the HfProvider. The range should
                    be thought of as a half-open interval [start, end), where start and
                    end are the indexed passed as a pair to the function.
        """
    def spin_of(self, arg0: tuple) -> str:
        """
        Get the spin block of each of the index components as a string.
        """
    def split(self, arg0: tuple) -> tuple:
        """
        Split an index into block index and in-block index
        """
    def split_spin(self, arg0: tuple) -> tuple:
        """
        Split an index into a spin block descriptor, a spatial block index and an in-block index.
        """
    @property
    def mospaces(self) -> MoSpaces:
        """
        Return the MoSpaces object supplied on initialisation
        """
    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions.
        """
    @property
    def shape(self) -> tuple:
        """
        Return the length along each dimension.
        """
    @property
    def space(self) -> str:
        """
        Return the space supplied on initialisation.
        """
    @property
    def subspaces(self) -> list[str]: ...

class MoSpaces:
    """
    Class setting up the molecular orbital index spaces and subspaces and exposing information about them. Python binding to :cpp:class:`libadcc::MoSpaces`.
    """
    def __init__(
        self,
        arg0: HartreeFockSolution_i,
        arg1: AdcMemory,
        arg2: list[int],
        arg3: list[int],
        arg4: list[int],
    ) -> None:
        """
        Construct an MoSpaces object from a HartreeFockSolution_i, a pointer to
        an AdcMemory object.

        adcmem_ptr        ADC memory keep-alive object to be used in all Tensors
                          constructed using this MoSpaces object.
        core_orbitals     List of orbitals indices (in the full fock space, original
                          ordering of the hf object), which defines the orbitals to
                          be put into the core space, if any. The same number
                          of alpha and beta orbitals should be selected. These will
                          be forcibly occupied.
        frozen_core_orbitals
                          List of orbital indices, which define the frozen core,
                          i.e. those occupied orbitals, which do not take part in
                          the ADC calculation. The same number of alpha and beta
                          orbitals has to be selected.
        frozen_virtuals    List of orbital indices, which the frozen virtuals,
                          i.e. those virtual orbitals, which do not take part
                          in the ADC calculation. The same number of alpha and beta
                          orbitals has to be selected.
        """
    def n_orbs(self, arg0: str) -> int:
        """
        The number of orbitals in a particular orbital subspace
        """
    def n_orbs_alpha(self, arg0: str) -> int:
        """
        The number of alpha orbitals in a particular orbital subspace
        """
    def n_orbs_beta(self, arg0: str) -> int:
        """
        The number of beta orbitals in a particular orbital subspace
        """
    @property
    def has_core_occupied_space(self) -> bool:
        """
        Does this object have a core-occupied space (i.e. is it ready for core-valence separation)?
        """
    @property
    def irrep_totsym(self) -> str:
        """
        Return the totally symmetric irreducible representation.
        """
    @property
    def irreps(self) -> list[str]:
        """
        The irreducible representations in the point group to which this class has been set up.
        """
    @property
    def map_block_irrep(self) -> dict[str, list[str]]:
        """
        Contains for each orbital space the mapping from each *block* used inside the space to the irreducible representation it correspond to.
        """
    @property
    def map_block_spin(self) -> dict[str, list[str]]:
        """
        Contains for each orbital space the mapping from each *block* used inside the space to the spin it correspond to ('a' is alpha and 'b' is beta)
        """
    @property
    def map_block_start(self) -> dict[str, list[int]]:
        """
        Contains for each orbital space the indices at which a new tensor block starts. Thus this list contains at least on index, namely 0.
        """
    @property
    def map_index_hf_provider(self) -> dict[str, list[int]]:
        """
        Contains for each orbital space (e.g. f, o1) a mapping from the indices used inside
        the respective space to the molecular orbital index convention used in the host
        program, i.e. to the ordering in which the molecular orbitals have been exposed
        in the HartreeFockSolution_i object passed on class construction.
        """
    @property
    def point_group(self) -> str:
        """
        The name of the point group for which the data in this class has been set up.
        """
    @property
    def restricted(self) -> bool:
        """
        Are the orbitals resulting from a restricted SCF calculation, such that alpha and beta electron share the same spatial part.
        """
    @property
    def subspaces(self) -> list[str]:
        """
        The list of all orbital subspaces known to this object.
        """
    @property
    def subspaces_occupied(self) -> list[str]:
        """
        The list of occupied orbital subspaces known to this object.
        """
    @property
    def subspaces_virtual(self) -> list[str]:
        """
        The list of virtual orbital subspaces known to this object.
        """

class ReferenceState:
    """
    Class representing information about the reference state for adcc. Python binding to:cpp:class:`libadcc::ReferenceState`.
    """
    def __init__(self, arg0: HartreeFockSolution_i, arg1: MoSpaces, arg2: bool) -> None:
        """
        Setup a ReferenceStateject using an MoSpaces object.

        hfsoln_ptr        Pointer to the Interface to the host program,
                          providing the HartreeFockSolution data, which
                          will be provided by this object.
        mo_ptr            MoSpaces object containing info about the MoSpace setup
                          and the point group symmetry.
        symmetry_check_on_import
                          Should symmetry of the imported objects be checked
                          explicitly during the import process. This massively slows
                          down the import process and has a dramatic impact on memory
                          usage and should thus only be used for debugging import routines
                          from the host programs. Do not enable this unless you know
                          that you really want to.
        """
    def eri(self, arg0: str) -> Tensor:
        """
        Return the ERI (electron-repulsion integrals) tensor block corresponding to the provided space.
        """
    def flush_hf_cache(self) -> None:
        """
        Tell the contained HartreeFockSolution_i object (which was passed upon construction), that a larger amount of import operations is done and that the next request for further imports will most likely take some time, such that intermediate caches can now be flushed to save some memory or other resources.
        """
    def fock(self, arg0: str) -> Tensor:
        """
        Return the Fock matrix block corresponding to the provided space.
        """
    def gauge_origin_to_xyz(self, arg0: str) -> tuple: ...
    def import_all(self) -> None:
        """
        Normally the class only imports the Fock matrix blocks and electron-repulsion integrals of a particular space combination when this is requested by a call to above fock() or eri() functions. This function call, however, instructs the class to immediately import *all* such blocks. Typically you do not want to do this.
        """
    def nuclear_quadrupole(
        self,
        arg0: typing.Annotated[list[float], pybind11_stubgen.typing_ext.FixedSize(3)],
    ) -> numpy.ndarray[numpy.float64]: ...
    def orbital_coefficients(self, arg0: str) -> Tensor:
        """
        Return the molecular orbital coefficients corresponding to the provided space (alpha and beta coefficients are returned)
        """
    def orbital_coefficients_alpha(self, arg0: str) -> Tensor:
        """
        Return the alpha molecular orbital coefficients corresponding to the provided space
        """
    def orbital_coefficients_beta(self, arg0: str) -> Tensor:
        """
        Return the beta molecular orbital coefficients corresponding to the provided space
        """
    def orbital_energies(self, arg0: str) -> Tensor:
        """
        Return the orbital energies corresponding to the provided space
        """
    @property
    def backend(self) -> str:
        """
        The identifier of the back end used for the SCF calculation.
        """
    @property
    def cached_eri_blocks(self) -> list[str]:
        """
        Get or set the list of momentarily cached ERI tensor blocks

        Setting this property allows to drop ERI tensor blocks if they are no longer needed to save memory.
        """
    @cached_eri_blocks.setter
    def cached_eri_blocks(self, arg1: list[str]) -> None: ...
    @property
    def cached_fock_blocks(self) -> list[str]:
        """
        Get or set the list of momentarily cached Fock matrix blocks

        Setting this property allows to drop fock matrix blocks if they are no longer needed to save memory.
        """
    @cached_fock_blocks.setter
    def cached_fock_blocks(self, arg1: list[str]) -> None: ...
    @property
    def conv_tol(self) -> float:
        """
        SCF convergence tolererance
        """
    @property
    def energy_scf(self) -> float:
        """
        Final total SCF energy
        """
    @property
    def has_core_occupied_space(self) -> bool:
        """
        Is a core occupied space setup, such that a core-valence separation can be applied.
        """
    @property
    def irreducible_representation(self) -> str:
        """
        Reference state irreducible representation
        """
    @property
    def mospaces(self) -> MoSpaces:
        """
        The MoSpaces object supplied on initialisation
        """
    @property
    def n_alpha(self) -> int:
        """
        Number of alpha electrons
        """
    @property
    def n_beta(self) -> int:
        """
        Number of beta electrons
        """
    @property
    def n_orbs(self) -> int:
        """
        Number of molecular orbitals
        """
    @property
    def n_orbs_alpha(self) -> int:
        """
        Number of alpha orbitals
        """
    @property
    def n_orbs_beta(self) -> int:
        """
        Number of beta orbitals
        """
    @property
    def nuclear_dipole(self) -> numpy.ndarray[numpy.float64]: ...
    @property
    def nuclear_total_charge(self) -> float: ...
    @property
    def restricted(self) -> bool:
        """
        Return whether the reference is restricted or not.
        """
    @property
    def spin_multiplicity(self) -> int:
        """
        Return the spin multiplicity of the reference state. 0 indicates that the spin cannot be determined or is not integer (e.g. UHF)
        """
    @property
    def timer(self) -> typing.Any:
        """
        Obtain the timer object of this class.
        """

class Symmetry:
    """
    Container for Tensor symmetry information
    """
    @typing.overload
    def __init__(self, arg0: MoSpaces, arg1: str) -> None:
        """
        Construct a Symmetry class from an MoSpaces object and the identifier for the space (e.g. o1o1, v1o1, o3v2o1v1, ...). Python binding to :cpp:class:`libadcc::Symmetry`.
        """
    @typing.overload
    def __init__(
        self, arg0: MoSpaces, arg1: str, arg2: dict[str, tuple[int, int]]
    ) -> None:
        """
        Construct a Symmetry class from an MoSpaces object, a space string and a map to supply the number of orbitals for some additional axes.
        For the additional axis the pair contains either two numbers (for the number of alpha and beta orbitals in this axis) or only one number and a zero (for an axis, which as only one spin kind, alpha or beta).

        This is an advanced constructor. Use only if you know what you do.
        """
    def clear(self) -> None:
        """
        Clear the symmetry.
        """
    def describe(self) -> str:
        """
        Return a descriptive string.
        """
    @property
    def empty(self) -> bool:
        """
        Is the symmetry empty (i.e. noy symmetry setup)
        """
    @property
    def irreps_allowed(self) -> list[str]:
        """
        The list of irreducible representations, for which the tensor shall be non-zero. If this is *not* set, i.e. an empty list, all irreps will be allowed.
        """
    @irreps_allowed.setter
    def irreps_allowed(self, arg1: list[str]) -> None: ...
    @property
    def mospaces(self) -> MoSpaces:
        """
        Return the MoSpaces object supplied on initialisation
        """
    @property
    def ndim(self) -> int:
        """
        Return the number of dimensions.
        """
    @property
    def permutations(self) -> list[str]:
        """
        The list of index permutations, which do not change the tensor.
        A minus may be used to indicate anti-symmetric
        permutations with respect to the first (reference) permutation.

        For example the list ["ij", "ji"] defines a symmetric matrix
        and ["ijkl", "-jikl", "-ijlk", "klij"] the symmetry of the ERI
        tensor. Not all permutations need to be given to fully describe
        the symmetry. Beware that the check for errors and conflicts
        is only rudimentary at the moment.
        """
    @permutations.setter
    def permutations(self, arg1: list[str]) -> None: ...
    @property
    def shape(self) -> tuple:
        """
        Return the shape of tensors constructed from this symmetry.
        """
    @property
    def space(self) -> str:
        """
        Return the space supplied on initialisation.
        """
    @property
    def spin_block_maps(self) -> list[tuple[str, str, float]]:
        """
        A list of tuples of the form ("aaaa", "bbbb", -1.0), i.e.
        two spin blocks followed by a factor. This maps the second onto the first
        with a factor of -1.0 between them.
        """
    @spin_block_maps.setter
    def spin_block_maps(self, arg1: list[tuple[str, str, float]]) -> None: ...
    @property
    def spin_blocks_forbidden(self) -> list[str]:
        """
        List of spin-blocks, which are marked forbidden (i.e. enforce them to stay zero).
        Blocks are given as a string in the letters 'a' and 'b', e.g. ["aaba", "abba"]
        """
    @spin_blocks_forbidden.setter
    def spin_blocks_forbidden(self, arg1: list[str]) -> None: ...

class Tensor:
    """
    Class representing the Tensor objects used for computations in adcc
    """

    flags: list[str]
    @typing.overload
    def __add__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __add__(self, arg0: Tensor) -> Tensor: ...
    def __getitem__(self, arg0: tuple) -> float:
        """
        Get a tensor element or a slice of tensor elements.
        """
    def __iadd__(self, arg0: Tensor) -> Tensor: ...
    def __imul__(self, arg0: float) -> Tensor: ...
    def __init__(self, arg0: Symmetry) -> None:
        """
        Construct a Tensor object using a Symmetry object describing its symmetry properties.
        The returned object is not guaranteed to contain initialised memory. Python binding to :cpp:class:`libadcc::Tensor`
        """
    def __isub__(self, arg0: Tensor) -> Tensor: ...
    def __itruediv__(self, arg0: float) -> Tensor: ...
    def __len__(self) -> int: ...
    def __matmul__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __mul__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __mul__(self, arg0: Tensor) -> Tensor:
        """
        Multiply two tensors elementwise.
        """
    def __neg__(self) -> Tensor: ...
    def __pos__(self) -> Tensor: ...
    def __radd__(self, arg0: float) -> Tensor: ...
    def __repr__(self) -> typing.Any: ...
    def __rmul__(self, arg0: float) -> Tensor: ...
    def __rsub__(self, arg0: float) -> Tensor: ...
    def __setitem__(self, arg0: tuple, arg1: float) -> float:
        """
        Set a tensor element or a slice of tensor elements. The operation will adhere symmetry, i.e. alter all elements equivalent by symmetry at once.
        """
    def __str__(self) -> typing.Any: ...
    @typing.overload
    def __sub__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __sub__(self, arg0: Tensor) -> Tensor: ...
    @typing.overload
    def __truediv__(self, arg0: float) -> Tensor: ...
    @typing.overload
    def __truediv__(self, arg0: Tensor) -> Tensor:
        """
        Divide two tensors elementwise.
        """
    @typing.overload
    def antisymmetrise(self, arg0: list) -> Tensor: ...
    @typing.overload
    def antisymmetrise(self, *args) -> Tensor: ...
    def copy(self) -> Tensor:
        """
        Returns a deep copy of the tensor.
        """
    @typing.overload
    def describe_expression(self, arg0: str) -> str:
        """
        Return a string providing a hopefully descriptive representation of the tensor expression stored inside the object.
        """
    @typing.overload
    def describe_expression(self) -> str: ...
    def describe_symmetry(self) -> str:
        """
        Return a string providing a hopefully descriptive representation of the symmetry information stored inside the tensor.
        """
    def diagonal(self, *args) -> Tensor: ...
    @typing.overload
    def dot(self, arg0: Tensor) -> float: ...
    @typing.overload
    def dot(self, arg0: list) -> numpy.ndarray[numpy.float64]: ...
    def empty_like(self) -> Tensor: ...
    def evaluate(self) -> Tensor:
        """
        Ensure the tensor to be fully evaluated and resilient in memory. Usually happens automatically when needed. Might be useful for fine-tuning, however.
        """
    def is_allowed(self, arg0: tuple) -> bool:
        """
        Is a particular index allowed by symmetry
        """
    def nosym_like(self) -> Tensor: ...
    def ones_like(self) -> Tensor: ...
    def select_n_absmax(self, arg0: int) -> list:
        """
        Select the n absolute maximal elements.
        """
    def select_n_absmin(self, arg0: int) -> list:
        """
        Select the n absolute minimal elements.
        """
    def select_n_max(self, arg0: int) -> list:
        """
        Select the n maximal elements.
        """
    def select_n_min(self, arg0: int) -> list:
        """
        Select the n minimal elements.
        """
    @typing.overload
    def set_from_ndarray(self, arg0: numpy.ndarray) -> Tensor:
        """
        Set all tensor elements from a standard np::ndarray by making a copy. Provide an optional tolerance argument to increase the tolerance for the check for symmetry consistency.
        """
    @typing.overload
    def set_from_ndarray(
        self, arg0: numpy.ndarray[numpy.float64], arg1: float
    ) -> Tensor:
        """
        Set all tensor elements from a standard np::ndarray by making a copy. Provide an optional tolerance argument to increase the tolerance for the check for symmetry consistency.
        """
    def set_immutable(self) -> None:
        """
        Set the tensor as immutable, allowing some optimisations to be performed.
        """
    def set_mask(self, arg0: str, arg1: float) -> None:
        """
        Set all elements corresponding to an index mask, which is given by a string eg. 'iijkli' sets elements T_{iijkli}
        """
    def set_random(self) -> Tensor:
        """
        Set all tensor elements to random data, adhering to the internal symmetry.
        """
    @typing.overload
    def symmetrise(self, arg0: list) -> Tensor: ...
    @typing.overload
    def symmetrise(self, *args) -> Tensor: ...
    def to_ndarray(self) -> numpy.ndarray[numpy.float64]:
        """
        Export the tensor data to a standard np::ndarray by making a copy.
        """
    @typing.overload
    def transpose(self) -> Tensor: ...
    @typing.overload
    def transpose(self, arg0: tuple) -> Tensor: ...
    def zeros_like(self) -> Tensor: ...
    @property
    def T(self) -> Tensor: ...
    @property
    def mutable(self) -> bool: ...
    @property
    def ndim(self) -> int: ...
    @property
    def needs_evaluation(self) -> bool:
        """
        Does the tensor need evaluation or is it fully evaluated and resilient in memory.
        """
    @property
    def shape(self) -> tuple: ...
    @property
    def size(self) -> int: ...
    @property
    def space(self) -> str: ...
    @property
    def subspaces(self) -> list[str]: ...

def amplitude_vector_enforce_spin_kind(arg0: Tensor, arg1: str, arg2: str) -> None:
    """
    Apply the spin symmetrisation required to make the doubles and higher parts of an amplitude vector consist of components for a particular spin kind only.
    """

def direct_sum(a: Tensor, b: Tensor) -> Tensor: ...
def evaluate(arg0: Tensor) -> Tensor: ...
def fill_pp_doubles_guesses(
    guesses_d: list[Tensor],
    mospaces: MoSpaces,
    df02: Tensor,
    df13: Tensor,
    spin_change_twice: int,
    degeneracy_tolerance: float,
) -> int:
    """
    Fill the passed vector of doubles blocks with doubles guesses using the delta-Fock matrices df02 and df13, which are the two delta-Fock matrices involved in the doubles block.

    guesses_d    Vectors of guesses, all elements are assumed to be initialised to zero and the symmetry is assumed to be properly set up.
    mospaces     Mospaces object
    df02         Delta-Fock between spaces 0 and 2 of the ADC matrix
    df13         Delta-Fock between spaces 1 and 3 of the ADC matrix
    spin_change_twice   Twice the value of the spin change to enforce in an excitation.
    degeneracy_tolerance  Tolerance for two entries of the diagonal to be considered degenerate, i.e. identical.
    Returns     The number of guess vectors which have been properly initialised (the others are invalid and should be discarded).
    """

def get_n_threads() -> int:
    """
    Get the number of running worker threads used by adcc.
    """

def get_n_threads_total() -> int:
    """
    Get the total number of threads (running and sleeping) used by adcc. This will disappear in the future. Do not rely on it.
    """

def linear_combination_strict(
    coefficients: numpy.ndarray[numpy.float64], tensors: list
) -> Tensor: ...
def make_symmetry_eri(arg0: MoSpaces, arg1: str) -> Symmetry:
    """
    Return the Symmetry object like it would be set up for the passed subspace
    of the electron-repulsion tensor.

      mospaces    MoSpaces object
      space       Space string (e.g. o1v1o1v1)
    """

def make_symmetry_operator(arg0: MoSpaces, arg1: str, arg2: str, arg3: str) -> Symmetry:
    """
    Return the Symmetry object for an orbital subspace block of a one-particle operator

      mospaces    MoSpaces object
      space       Space string (e.g. o1v1)
      symmetry    Describes the symmetry of the tensor (only in effect if both               subspaces
                  of the space string are identical).              nosymmetry
                  hermitian
                  antihermitian
      cartesian_transformation
                  The cartesian function according to which the operator transforms.

    Valid cartesian_transformation values include:
         "1"                   Totally symmetric (default)
         "x", "y", "z"         Coordinate axis
         "xx", "xy", "yz" ...  Products of two coordinate axis
         "Rx", "Ry", "Rz"      Rotations about the coordinate axis
    """

def make_symmetry_operator_basis(
    arg0: MoSpaces, arg1: int, arg2: str, arg3: int, arg4: str
) -> Symmetry:
    """
    Return the symmetry object for an operator in the AO basis. The object will
    represent a block-diagonal matrix of the form
        ( M 0 )
        ( 0 M ).
    where M is an n_bas x n_bas block and is indentical in upper-left
    and lower-right.

    mospaces_ptr     MoSpaces pointer
    n_bas            Number of AO basis functions
    symmetry         Is the tensor symmetric (only in effect if both space
                     axes identical). false disables a setup of permutational
                     symmetry.
    n_particle_op    NParticleOperator
    blocks           Spin blocks to include. Valid are "ab", "a" and "b".
    """

def make_symmetry_orbital_coefficients(
    arg0: MoSpaces, arg1: str, arg2: int, arg3: str
) -> Symmetry:
    """
    Return the Symmetry object like it would be set up for the passed subspace
    of the orbital coefficients tensor.

      mospaces    MoSpaces object
      space       Space string (e.g. o1b)
      n_bas       Number of basis functions
      blocks      Spin blocks to include. Valid are "ab", "a" and "b".
    """

def make_symmetry_orbital_energies(arg0: MoSpaces, arg1: str) -> Symmetry:
    """
    Return the Symmetry object like it would be set up for the passed subspace
    of the orbital energies tensor.

      mospaces    MoSpaces object
      space       space string (e.g. o1)
    """

def make_symmetry_triples(arg0: MoSpaces, arg1: str) -> Symmetry:
    """
    Return the Symmetry object like it would be set up for the passed subspace
    of a triples amplitude tensor.

      mospaces    MoSpaces object
      space       Space string (e.g. o1o1o1v1v1v1)
    """

def set_n_threads(arg0: int) -> None:
    """
    Set the number of running worker threads used by adcc
    """

def set_n_threads_total(arg0: int) -> None:
    """
    Set the total number of threads (running and sleeping) used by adcc. This will disappear in the future. Do not rely on it.
    """

@typing.overload
def tensordot(a: Tensor, b: Tensor, axes: typing.Iterable) -> typing.Any: ...
@typing.overload
def tensordot(a: Tensor, b: Tensor, axes: int) -> typing.Any: ...
@typing.overload
def tensordot(a: Tensor, b: Tensor) -> typing.Any: ...
@typing.overload
def trace(subscripts: str, tensor: Tensor) -> float: ...
@typing.overload
def trace(tensor: Tensor) -> float: ...

__backend__: dict = {
    "name": "libtensorlight",
    "version": "3.0.1",
    "authors": "Evgeny Epifanovsky, Michael Wormit, Dmitry Zuev Sam Manzer, Ilya Kaliman, Michael F. Herbst and Maximilian Scheurer",
    "features": ["libxm"],
    "blas": "Apple",
}
