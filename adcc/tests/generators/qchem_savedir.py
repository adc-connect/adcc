from pathlib import Path
import numpy as np
import struct
import h5py


Array4D = np.ndarray[tuple[int, int, int, int]]


class QchemSavedir:
    """
    Generates a Qchem savedir containing the SCF solution
    (MO coefficients, Fock matrix in AO basis, ...)
    from numpy arrays such that another QChem calculation can
    read the data from the savedir as SCF guess.
    The numpy arrays are assumed to have the correct order, i.e.,
    the basis functions and MOs have to match the order used
    in the following QChem calculation.
    MOs in Qchem are sorted as: occ_a, virt_a, occ_b, virt_b,
    where a and b denote alpha and beta spin, respectively.

    Parameters
    ----------
    savedir: str
        name (or path) of the directory in which the files are written.
    """
    _filenames = {
        "mo_coeffs":         ("53",        "qchem_fortran_style"),
        "density_matrix_ao": ("54",        "qchem_fortran_style"),
        "fock_matrix_ao":    ("58",        "qchem_fortran_style"),
        "energies":          ("99",        "qchem_fortran_style"),
        "dimensions":        ("819",       "qchem_fortran_style"),
        "integrals":         ("integrals", "hdf5"),
    }

    def __init__(self, savedir: str | Path) -> None:
        if isinstance(savedir, str):
            savedir = Path(savedir)
        self.savedir = savedir.resolve()
        self.savedir.mkdir(parents=True, exist_ok=True)

    def write(self, scf_energy: float,
              mo_coeffs: np.ndarray[tuple[int, int]],
              fock_ao: np.ndarray[tuple[int, int]],
              orb_energies: np.ndarray[tuple[int]],
              ao_density_aa: np.ndarray[tuple[int, int]],
              ao_density_bb: np.ndarray[tuple[int, int]],
              purecart: int,
              n_basis: int | None = None,
              n_orbitals: int | None = None,
              n_fragments: int = 0,
              eri_blocks: dict[str, Array4D] | None = None,
              ao_integrals: dict[str, np.ndarray] | None = None) -> None:
        """
        Write all content in the savedir.

        Parameters
        ----------
        scf_energy: float
            The SCF energy.
        mo_coeffs: np.ndarray
            (MO x basis) array containing the MO coefficients.
        fock_ao: np.ndarray
            (basis x basis) Fock matrix in the ao basis.
        orb_energies: np.ndarray
            Vector of shape (MO,) containing the orbital energies.
        ao_density_aa: np.ndarray
            alpha, alpha block of the (MO x MO) density matrix in the AO basis.
        ao_density_bb: np.ndarray
            beta, beta block of the (MO x MO) density matrix in the AO basis.
        purecart: int
            Indicates for which angular momentums cartesian angular functions
            are used in the basis, i.e., if the 6 cartesian d-orbitals
            d_xx, d_xy, d_xz, d_yy, d_yz, d_zz
            are used or the 5 specircal harmonics
            d_xy, d_xz, d_xy, d_z^2, d_x^2-y^2.
            2222: use cartesian functions for h, g, f and d orbitals
            1111: use spherical harmonics for h, g, f and d orbitals
        n_basis: int, optional
            The number of basis function. If not given determined from the
            shape of the MO coefficients.
        n_orbitals: int, optional
            The number of orbitals. If not given determined from the
            shape of the MO coefficients.
        n_fragments: int, optional
            Probably related to the number of fragments generated for the SAD
            SCF Guess. Should not be relevant. The default value (0) seems to work.
        eri_blocks: dict[str, np.ndarray], optional
            The anti-symmetric ERI blocks in the MO basis. Blocks of the form
            'ooov' or 'ococ' are expected. Written to a HDF5 located in the savedir.
            Reading the anti-symmetric ERI is only supported within adcman!
            If not given, Qchem will compute the integrals during the calculation.
        ao_integrals: dict[str, np.ndarray], optional
            Integral matrices in the AO basis. Currently the dipole matrices
            (expected names: 'dx', 'dy' and 'dz') are supported.
            Reading the ani-symmetric ERI is only supported within adcman!
            If not given, Qchem will compute the integrals during the calculation.
        """
        # determine n_basis and n_orbitals if not provided
        if n_basis is None or n_orbitals is None:
            n_orbs, n_bas = mo_coeffs.shape
            if n_basis is None:
                n_basis = n_bas
            if n_orbitals is None:
                # we always have the same number of alpha and beta orbitals!
                assert not n_orbs % 2
                n_orbitals = n_orbs // 2
        # write all data into savedir
        self.write_mo_coeffs(mo_coeffs=mo_coeffs, orb_energies=orb_energies)
        self.write_scf_energy(scf_energy)
        self.write_ao_density(density_aa=ao_density_aa, density_bb=ao_density_bb)
        self.write_ao_fock(fock_ao)
        self.write_dims(n_basis=n_basis, n_orbitals=n_orbitals,
                        purecart=purecart, n_fagments=n_fragments)
        if eri_blocks is not None:
            self.write_antisym_eri(**eri_blocks)
        if ao_integrals is not None:
            self.write_ao_integrals(**ao_integrals)

    def write_mo_coeffs(self, mo_coeffs: np.ndarray[tuple[int, int]],
                        orb_energies: np.ndarray[tuple[int]]) -> None:
        """
        Writes the file for MO coefficients. Since the file also contains
        the orbital energies, they have to be provided too.

        Parameters
        ----------
        mo_coeffs: np.ndarray
            Array containing the MO coefficients: MO x basis array
        orb_energies: np.ndarray
            Vector of orbital energies.
        """
        fpath = self._get_filename("mo_coeffs")
        with open(fpath, "wb") as f:
            f.write(mo_coeffs.tobytes())
            f.write(orb_energies.tobytes())

    def write_scf_energy(self, scf_energy: float) -> None:
        """
        Write the file containing the SCF energy.
        """
        fname = self._get_filename("energies")
        with open(fname, "wb") as f:
            f.write(
                struct.pack("<12d", 0., scf_energy, *[0. for _ in range(10)])
            )

    def write_ao_density(self, density_aa: np.ndarray[tuple[int, int]],
                         density_bb: np.ndarray[tuple[int, int]]) -> None:
        """
        Write the file containing the SCF density in the AO basis.

        Parameters
        ----------
        density_aa: np.ndarray
            alpha, alpha block of the AO (MOxMO) density matrix.
        density_bb: np.ndarray
            beta, beta block of the AO (MOxMO) density matrix.
        """
        fname = self._get_filename("density_matrix_ao")
        with open(fname, "wb") as f:
            f.write(density_aa.tobytes())
            f.write(density_bb.tobytes())

    def write_ao_fock(self, fock: np.ndarray[tuple[int, int]]):
        """
        Write the Fock matrix in the AO basis to file. The Fock matrix is
        a (basis x basis) array.
        """
        fname = self._get_filename("fock_matrix_ao")
        with open(fname, "wb") as f:
            f.write(fock.tobytes())

    def write_dims(self, n_basis: int, n_orbitals: int, purecart: int,
                   n_fagments: int = 0) -> None:
        """
        Write the file containing general information about the dimensions
        and the basis.

        Parameters
        ----------
        n_basis: int
            The number of basis functions.
        n_orbitals: int
            The number of alpha orbitals.
        purecart: int
            Indicates for which angular momentums cartesian angular functions
            are used in the basis, i.e., if the 6 cartesian d-orbitals
            d_xx, d_xy, d_xz, d_yy, d_yz, d_zz
            are used or the 5 spherical harmonics
            d_xy, d_xz, d_xy, d_z^2, d_x^2-y^2.
            2222: use cartesian functions for h, g, f and d orbitals
            1111: use spherical harmonics for h, g, f and d orbitals
        n_fagments: int, optional
            Probably related to the number of fragments generated for the SAD
            SCF Guess. Should not be relevant. The default value (0) seems to work.
        """
        fname = self._get_filename("dimensions")
        with open(fname, "wb") as f:
            f.write(
                struct.pack("<4i", n_basis, n_orbitals, purecart, n_fagments)
            )

    def write_antisym_eri(self, **eri_blocks: np.ndarray) -> None:
        """
        Write the anti-symmetric ERI to the savedir using a HDF5 file.
        Only supported within adcman.

        Parameters
        ----------
        eri_blocks: np.ndarray
            The anti-symmetric ERI blocks to store in the hdf5 file in the savedir.
            ERI blocks in the form 'ooov' or 'ococ' are expected.
        """
        spaces = {"o": "o1", "c": "o2", "v": "v1"}
        fname = self._get_filename("integrals")
        hdf5_file = h5py.File(fname, "a")  # Read/write if exists, create otherwise
        for block, tensor in eri_blocks.items():
            if len(block) != 4 or not all(sp in spaces for sp in block):
                raise ValueError(f"Found unexpected ERI block: {block}")
            key = "hf/i_" + "".join(spaces[sp] for sp in block)
            hdf5_file.create_dataset(key, data=tensor.flatten())

    def write_ao_integrals(self, **ao_integrals: np.ndarray) -> None:
        """
        Write integrals like the dipole vector operator matrices to the savedir
        using a HDF5 file. Currently supported are
        - the dipole vector AO matrices (expected as: 'dx', 'dy', 'dz')
        Only supported within adcman.
        """
        dipoles = ("dx", "dy", "dz")
        fname = self._get_filename("integrals")
        hdf5_file = h5py.File(fname, "a")  # Read/write if exists, create otherwise
        for name, tensor in ao_integrals.items():
            if name in dipoles:
                hdf5_file.create_dataset(f"ao/{name}_bb", data=tensor.flatten())
                continue
            raise ValueError(f"Unknown ao integral {name}")

    def _get_filename(self, content: str) -> Path:
        """
        Get the file name in which to write the desired content.
        """
        if content not in self._filenames:
            raise ValueError(f"Unknown file content {content}. Can't determine "
                             "the file name.")
        name, file_type = self._filenames[content]
        if file_type == "hdf5":
            fpath = self.savedir / f"{name}.hdf5"
        elif file_type == "qchem_fortran_style":
            fpath = self.savedir / f"{name}.0"
        else:
            raise ValueError(f"Unknown file type {file_type}.")
        return fpath
