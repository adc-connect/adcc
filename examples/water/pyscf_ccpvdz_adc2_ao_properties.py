#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import numpy as np

from matplotlib import pyplot as plt
from adcc.visualisation import ExcitationSpectrum

from pyscf import gto, scf
from pyscf.tools import cubegen

from scipy import constants

# Hartree to eV
eV = constants.value("Hartree energy in eV")


# Dump cube files
dump_cube = False


def plot_spectrum(energies, strengths, width=0.045):
    energies = np.array(energies).flatten()
    strengths = np.array(strengths).flatten()

    def ngauss(en, osc, x, w):
        """A normalised Gaussian"""
        fac = osc / np.sqrt(2 * np.pi * w**2)
        return fac * np.exp(-(x - en)**2 / (2 * w**2))

    xval = np.arange(np.min(np.min(energies) - 1, 0),
                     np.max(energies) + 1, 0.01)
    yval = np.zeros(xval.size)
    for en, osc in zip(energies, strengths):
        yval += ngauss(en, osc, xval, width)
    plt.plot(xval, yval)


#
# Run SCF in pyscf
#
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='cc-pvdz',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-12
scfres.conv_tol_grad = 1e-9
scfres.kernel()

print(adcc.banner())

# Run an adc2 calculation:
state = adcc.adc2(scfres, n_singlets=7, conv_tol=1e-8)

#
# Get HF density matrix and nuclear dipole
#
ρ_hf_tot = scfres.make_rdm1()

# Compute dipole integrals
dip_ao = mol.intor_symmetric('int1e_r', comp=3)

# compute nuclear dipole
charges = mol.atom_charges()
coords = mol.atom_coords()
dip_nucl = np.einsum('i,ix->x', charges, coords)

#
# MP2 density correction
#
mp2dm_mo = state.ground_state.mp2_diffdm
dm_mp2_ao = mp2dm_mo.transform_to_ao_basis(state.reference_state)
ρ_mp2_tot = (dm_mp2_ao[0] + dm_mp2_ao[1]).to_ndarray() + ρ_hf_tot

# dipole moment operator function from backend
dips = state.reference_state.operators.electric_dipole
#
# Compute properties
#
exc_energies = []    # Excitation energies
osc_strengths = []    # Oscillator strength

print()
print("  st  ex.ene. (au)         f     transition dipole moment (au)"
      "        state dip (au)")
for i, ampl in enumerate(state.excitation_vectors):
    # Compute transition density matrix
    tdm_mo = state.transition_dms[i]
    tdm_ao = tdm_mo.transform_to_ao_basis(state.reference_state)
    ρ_tdm_tot = (tdm_ao[0] + tdm_ao[1]).to_ndarray()

    # Compute transition dipole moment
    exci = state.excitation_energies[i]
    tdip = np.einsum('xij,ij->x', dip_ao, ρ_tdm_tot)
    osc = 2. / 3. * np.linalg.norm(tdip)**2 * np.abs(exci)

    # Compute excited states density matrix and excited state dipole moment
    opdm_mo = state.state_diffdms[i]
    opdm_ao = opdm_mo.transform_to_ao_basis(state.reference_state)
    ρdiff_opdm_ao = (opdm_ao[0] + opdm_ao[1]).to_ndarray()
    sdip_el = np.einsum('xij,ij->x', dip_ao, ρdiff_opdm_ao + ρ_mp2_tot)
    sdip = sdip_el - dip_nucl

    # Print findings
    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    # fmt += "   [{9:9.3g}, {10:9.3g}, {11:9.3g}]"
    print(state.kind[0], fmt.format(i, exci, osc, *tdip, *sdip))

    if dump_cube:
        # Dump LUNTO and HONTO
        u, s, v = np.linalg.svd(ρ_tdm_tot)
        # LUNTOs
        cubegen.orbital(mol=mol, coeff=u.T[0],
                        outfile="nto_{}_LUNTO.cube".format(i))
        # HONTOs
        cubegen.orbital(mol=mol, coeff=v[0],
                        outfile="nto_{}_HONTO.cube".format(i))

    # Save oscillator strength and excitation energies
    osc_strengths.append(osc)
    exc_energies.append(state.excitation_energies[i])
exc_energies = np.array(exc_energies) * eV
osc_strengths = np.array(osc_strengths)

sp = ExcitationSpectrum(exc_energies, osc_strengths)
sp.xlabel = "Energy (eV)"
sp.plot(style="discrete", color="r")
sp_broad = sp.broaden_lines(shape="lorentzian")
sp_broad.plot(color="b", style="continuous")

plt.show()
