import psi4
import adcc
from psi4.driver.procrouting.response.scf_response import tdscf_excitations
from adcc.testdata import static_data
import yaml
import os


basis_remap = {
    "sto3g": "sto-3g",
    "def2tzvp": "def2-tzvp",
    "ccpvdz": "cc-pvdz",
}


def run_psi4_tdscf(xyz, basis, charge=0, multiplicity=1,
                   conv_tol=1e-12, conv_tol_grad=1e-11, max_iter=150,
                   pcm_options=None):
    mol = psi4.geometry(f"""
        {charge} {multiplicity}
        {xyz}
        units au
        symmetry c1
    """)
    psi4.set_options({
        'basis': basis_remap[basis],
        'scf_type': "pK",
        'e_convergence': conv_tol,
        'd_convergence': conv_tol_grad,
        'maxiter': max_iter,
        'reference': "RHF",
        'save_jk': True,
    })
    if pcm_options:
        # think its better to use a dict for the pcm options, because there
        # are already enough function arguments. Of course one could just
        # fix the options... would be sufficient for now too.
        psi4.set_options({'pcm': True, 'pcm_scf_type': "total"})
        psi4.pcm_helper(f"""
            Units = AU
            Cavity {{
                Type = Gepol
                Area = {pcm_options.get("weight", 0.3)}
            }}
            Medium {{
                Solvertype = {pcm_options.get("pcm_method", "IEFPCM")}
                Solvent = {pcm_options.get("solvent", "Water")}
                Nonequilibrium = {pcm_options.get("neq", True)}
            }}
        """)
    psi4.core.be_quiet()
    _, wfn = psi4.energy('scf', return_wfn=True, molecule=mol)

    res = tdscf_excitations(wfn, states=5, triplets="NONE", tda=True,
                            r_convergence=1e-7)

    # remove cavity files from PSI4 PCM calculations
    if pcm_options:
        for cavityfile in os.listdir(os.getcwd()):
            if cavityfile.startswith(("cavity.off_", "PEDRA.OUT_")):
                os.remove(cavityfile)
    return wfn, res


def run_adcc_ptlr(wfn):
    return adcc.run_adc(wfn, method="adc1", n_singlets=5,
                        conv_tol=1e-7, environment="ptlr")


def dump_results(molecule, basis, **kwargs):
    pcm_options = kwargs.get("pcm_options", None)
    if pcm_options:
        name = f"{molecule}_{basis}_pcm_adc1"
    else:
        name = f"{molecule}_{basis}_adc1"

    geom = static_data.xyz[molecule]

    ret = {"basis": basis,
           "method": "adc1",
           "molecule": molecule}

    wfn, res = run_psi4_tdscf(geom, basis, pcm_options=pcm_options)

    ret["energy_scf"] = wfn.energy()
    # yaml safe_dump doesn't like np.floats
    ret["lr_excitation_energy"] = [round(float(r["EXCITATION ENERGY"]), 9)
                                   for r in res]
    ret["lr_osc_strength"] = [round(float(r["OSCILLATOR STRENGTH (LEN)"]), 5)
                              for r in res]

    if pcm_options:
        state = run_adcc_ptlr(wfn)
        ret["ptlr_adcc_excitation_energy"] = [
            round(float(e), 9) for e in state.excitation_energy
        ]
    return name, ret


def main():
    basis_set = ["sto3g", "ccpvdz"]
    pcm_options = {"weight": 0.3, "pcm_method": "IEFPCM", "neq": True,
                   "solvent": "Water"}
    psi4_results = {}
    for basis in basis_set:
        key, ret = dump_results("formaldehyde", basis, pcm_options=pcm_options)
        psi4_results[key] = ret
        print(f"Dumped {key}.")

    with open("psi4_dump.yml", 'w') as yamlfile:
        yaml.safe_dump(psi4_results, yamlfile,
                       sort_keys=False, default_flow_style=None)


if __name__ == "__main__":
    main()
