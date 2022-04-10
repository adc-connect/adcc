import adcc
from pyscf import gto, scf, tdscf
from pyscf.solvent import ddCOSMO
from adcc.testdata import static_data
import yaml


def run_pyscf_tdscf(xyz, basis, charge=0, multiplicity=1, conv_tol=1e-12,
                    conv_tol_grad=1e-11, max_iter=150, pcm_options=None):
    mol = gto.M(
        atom=xyz,
        basis=basis,
        unit="Bohr",
        charge=charge,
        # spin in the pyscf world is 2S
        spin=multiplicity - 1,
        verbose=0,
        # Disable commandline argument parsing in pyscf
        parse_arg=False,
        dump_input=False,
    )

    if pcm_options:
        mf = ddCOSMO(scf.RHF(mol))
        mf.with_solvent.eps = pcm_options.get("eps")
    else:
        mf = scf.RHF(mol)
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = conv_tol_grad
    mf.max_cycle = max_iter

    mf.kernel()

    if pcm_options:
        mf.with_solvent.eps = pcm_options.get("eps_opt")
        # for n_eq solvation only PTE implemented
        mf.with_solvent.equilibrium_solvation = True
        cis = ddCOSMO(tdscf.TDA(mf))
    else:
        cis = tdscf.TDA(mf)
    cis.nstates = 5
    cis.conv_tol = 1e-7
    cis.kernel()
    return mf, cis


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

    hf, cis = run_pyscf_tdscf(geom, basis, pcm_options=pcm_options)

    ret["energy_scf"] = float(hf.e_tot)
    ret["lr_excitation_energy"] = [float(round(s, 9)) for s in cis.e]
    ret["lr_osc_strength"] = [float(round(s, 5)) for s in cis.oscillator_strength()]

    if pcm_options:
        state = run_adcc_ptlr(hf)
        ret["ptlr_adcc_excitation_energy"] = [
            round(float(e), 9) for e in state.excitation_energy
        ]
    return name, ret


def main():
    basis_set = ["sto3g", "ccpvdz"]
    pcm_options = {"eps": 78.3553, "eps_opt": 1.78}
    pyscf_results = {}
    for basis in basis_set:
        key, ret = dump_results("formaldehyde", basis, pcm_options=pcm_options)
        pyscf_results[key] = ret
        print(f"Dumped {key}.")

    with open("pyscf_dump.yml", 'w') as yamlfile:
        yaml.safe_dump(pyscf_results, yamlfile,
                       sort_keys=False, default_flow_style=None)


if __name__ == "__main__":
    main()
