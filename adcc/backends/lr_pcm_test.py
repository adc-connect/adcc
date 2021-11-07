import psi4
import adcc


mol = psi4.geometry("""
0 1
    C 2.0092420208996 3.8300915804899 0.8199294419789
    O 2.1078857690998 2.0406638776593 2.1812021228452
    H 2.0682421748693 5.7438044586615 1.5798996515014
    H 1.8588483602149 3.6361694243085 -1.2192956060942
units au
symmetry c1
""")
psi4.set_options({
    'basis': "6-31g",
    'scf_type': "pK",
    'e_convergence': 1e-12,
    'd_convergence': 1e-11,
    'reference': "RHF",
    'pcm': True,
    'pcm_scf_type': "total"
    }
)
psi4.pcm_helper(""" 
    Units = AU
    Cavity {
        Type = Gepol
        Area = 0.3
        Radiiset = Bondi
        Scaling = True
    }
    Medium {
        Solvertype = IEFPCM
        Solvent = Water
        Nonequilibrium = True
    }
""")

# psi4.core.be_quiet()
psi4.core.set_output_file('adcc.out', False)
scf_e, scf_wfn = psi4.energy('scf', return_wfn=True)
state = adcc.run_adc(scf_wfn, method="adc1", n_singlets=3,
                     conv_tol=1e-7, environment="ptlr")

print(state.describe())

for i, energy in enumerate(state.excitation_energy):
    print(f"\nstate: {i+1}")
    print("uncorrected {:.7f}, corrected {:.7f}, diff(correction): {:.7f}"
          .format(state.excitation_energy_uncorrected[i], energy,
                  state.excitation_energy_uncorrected[i] - energy
                  )
          )
