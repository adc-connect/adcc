from pathlib import Path
import numpy as np
import struct
import warnings


class QchemSavedir:
    """
    Generates a Qchem savedir containing the SCF solution
    (MO coefficients, fock matrix in AO basis, ...)
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
        "mo_coeffs":          53,
        "density_matrix_ao":  54,
        "fock_matrix_ao":     58,
        "energies":           99,
        "dimensions":        819,
    }

    def __init__(self, savedir: str) -> None:
        self.savedir = Path(savedir).resolve()
        self.savedir.mkdir(parents=True, exist_ok=True)

    def write(self, scf_energy: float, mo_coeffs: np.ndarray,
              fock_ao: np.ndarray, orb_energies: np.ndarray,
              ao_density_aa: np.ndarray, ao_density_bb: np.ndarray,
              purecart: int, n_basis: int = None,
              n_orbitals: int = None, n_fragments: int = 0):
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
        # write all data in the savedir
        self.write_mo_coeffs(mo_coeffs=mo_coeffs, orb_energies=orb_energies)
        self.write_scf_energy(scf_energy)
        self.write_ao_density(density_aa=ao_density_aa, density_bb=ao_density_bb)
        self.write_ao_fock(fock_ao)
        self.write_dims(n_basis=n_basis, n_orbitals=n_orbitals,
                        purecart=purecart, n_fagments=n_fragments)

    def write_mo_coeffs(self, mo_coeffs: np.ndarray,
                        orb_energies: np.ndarray) -> None:
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

    def write_ao_density(self, density_aa: np.ndarray,
                         density_bb: np.ndarray) -> None:
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

    def write_ao_fock(self, fock: np.ndarray):
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
            are used or the 5 specircal harmonics
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

    def _get_filename(self, content: str) -> Path:
        """
        Get the file name in which to write the desired content.
        """
        id = self._filenames.get(content, None)
        if id is None:
            raise ValueError(f"Unknown file content {content}. Can't determine "
                             "the file name.")
        fpath = self.savedir / f"{id}.0"
        if fpath.exists():
            warnings.warn(f"The file {fpath} already exists.")
        return fpath
