#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import numpy as np

from scipy import constants
from matplotlib import pyplot as plt
from adcc.visualisation import ExcitationSpectrum

from pyscf import gto, scf
from pyscf.tools import cubegen

eV = constants.value("Hartree energy in eV")  # Hartree to eV

#
# This script shows a more low-level approach to property calculations
# showcasing how state and transition density matrices can be obtained
# and worked with. An example showing the recommended way to plot spectra
# can be found in pyscf_ccpvdz_adc2_spectrum.py
#


# Run SCF in pyscf
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

# Get dipole integrals from pyscf
dip_ao = mol.intor_symmetric('int1e_r', comp=3)

# compute nuclear dipole
charges = mol.atom_charges()
coords = mol.atom_coords()
dip_nucl = np.einsum('i,ix->x', charges, coords)

#
# Compute properties
#
exc_energies = []    # Excitation energies
osc_strengths = []    # Oscillator strength

print()
print("  st  ex.ene. (au)         f     transition dipole moment (au)"
      "        state dip (au)")
for i, exci in enumerate(state.excitation_energies):
    # Compute transition density matrix
    tdm_ao = state.transition_dms[i].to_ao_basis()
    ρ_tdm_tot = (tdm_ao[0] + tdm_ao[1]).to_ndarray()

    # Compute transition dipole moment
    tdip = np.einsum('xij,ij->x', dip_ao, ρ_tdm_tot)
    osc = 2. / 3. * np.linalg.norm(tdip)**2 * np.abs(exci)

    # Compute excited states density matrix and excited state dipole moment
    opdm_ao = state.state_dms[i].to_ao_basis()
    ρ_opdm_tot = (opdm_ao[0] + opdm_ao[1]).to_ndarray()
    sdip_el = np.einsum('xij,ij->x', dip_ao, ρ_opdm_tot)
    sdip = sdip_el - dip_nucl

    # Print findings
    fmt = "{0:2d}  {1:12.8g} {2:9.3g}   [{3:9.3g}, {4:9.3g}, {5:9.3g}]"
    fmt += "   [{6:9.3g}, {7:9.3g}, {8:9.3g}]"
    # fmt += "   [{9:9.3g}, {10:9.3g}, {11:9.3g}]"
    print(state.kind[0], fmt.format(i, exci, osc, *tdip, *sdip))

    # Build LUNTO and HONTO by SVD
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
sp_broad = sp.broaden_lines(shape="lorentzian", width=0.08)
sp_broad.plot(color="b", style="continuous")

plt.show()
