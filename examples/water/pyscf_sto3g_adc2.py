#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc
import numpy as np

from adcc.adc_pp.modified_transition_moments import modified_transition_moments
from adcc.IsrMatrix import IsrMatrix
from adcc.State2States import State2States
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm
from adcc.NParticleOperator import product_trace
from pyscf import gto, scf


# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='sto-3g',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-14
scfres.conv_tol_grad = 1e-10
scfres.kernel()

hf = adcc.ReferenceState(scfres)
mp = adcc.LazyMp(hf)

singlets = adcc.adc3(scfres, n_singlets=3, isr_order=3)
mp = singlets.ground_state
tdm = singlets.transition_dipole_moment
state = singlets
#method = singlets.method
#fmatrix
n_ref = len(singlets.excitation_vector)
dips = singlets.reference_state.operators.electric_dipole
mtms = modified_transition_moments("adc3", singlets.ground_state, dips)
for i in range(n_ref):
    excivec = singlets.excitation_vector[i]
    res_tdm = np.array([excivec @ mtms[i] for i in range(3)])

dipole_integrals = hf.operators.electric_dipole
from_a = singlets.excitation_vector[1]
to_a = singlets.excitation_vector[2]
s2s_tdm = state2state_transition_dm("adc3", mp, from_a, to_a)

s2s_moments = [product_trace(comp, s2s_tdm) for comp in dipole_integrals]


#bmatrix
matrix = IsrMatrix("adc3", mp, dips)
for ifrom in range(n_ref - 1):
    B_Yn = matrix @ state.excitations[ifrom].excitation_vector
    state2state = State2States(state, initial=ifrom)
    for j, ito in enumerate(range(ifrom + 1, n_ref)):
        s2s_tdm = [state.excitations[ito].excitation_vector @ vec
                   for vec in B_Yn]

print(singlets.describe())

#state diffdm
for i in range(len(state.excitation_vector)):
    # Check that we are talking about the same state when
    # comparing reference and computed
    #assert state.excitation_energy[i] == refdata["eigenvalues"][i]

    dm_ao_a, dm_ao_b = state.state_diffdm[i].to_ao_basis()
#s2s dm


print(" method :", singlets.method)
print("property method :", singlets.property_method)

print("Excited state dipole moment:")
print(singlets.state_dipole_moment)
print("trnsition dipole moment")
print(tdm)
print("state2state momrnts")
print(s2s_moments)
print(mtms)
print("f matrix")
print(res_tdm)
print("b matrix")
print(s2s_tdm)
# print("state diffdm")
# print(dm_ao_a, dm_ao_b)


#dipole_integrals = hf.operators.electric_dipole
# from_a = singlets.excitation_vector[1]
# to_a = singlets.excitation_vector[2]
# s2s_tdm = state2state_transition_dm("adc3", mp, from_a, to_a)

# s2s_moments = [product_trace(comp, s2s_tdm) for comp in dipole_integrals]
# print(s2s_moments)
# ddm = singlets.state_diffdm[0]
# print(ddm)

# print(mp.mp3_dipole_moment)
# print(np.linalg.norm(mp.mp3_dipole_moment) * 2.5412)

#print(mp.mp2_dipole_moment)
# print(np.linalg.norm(mp.mp2_dipole_moment) * 2.5412)



