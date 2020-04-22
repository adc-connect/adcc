#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
import adcc

from pyscf import gto, scf
from matplotlib import pyplot as plt

enantiomers = {
    "(S)-2-methyloxirane": """
        O	0.7971066654	0.9044360742	0.0836962049
        C	-0.1867183086	-0.0290724859	0.5536827176
        C	-1.4336843546	-0.1726679227	-0.2822214295
        C	1.1302222000	-0.4892393880	0.0894444115
        H	1.2197487995	-0.9517340291	-0.8946449424
        H	1.8923895176	-0.7869225283	0.8107731933
        H	-0.3474086480	0.0162374592	1.6337796505
        H	-2.0955293870	0.6891134744	-0.1384941617
        H	-1.9883466588	-1.0759327249	-0.0005360999
        H	-1.1805969868	-0.2349473270	-1.3455182514
    """,
    "(R)-2-methyloxirane": """
        O	-0.8328602577	0.7897730814	-0.2375616734
        C	0.1486158153	0.0360794279	0.4890402083
        H	0.1511355430	0.2453732348	1.5616840340
        C	-1.0455791318	-0.6173265887	-0.0661148292
        H	-1.8751757963	-0.8906447658	0.5872264586
        H	-0.9556948808	-1.2169886069	-0.9730489338
        C	1.5085709521	-0.0930940130	-0.1498440685
        H	1.4131373959	-0.3116398357	-1.2181956153
        H	2.0766856596	0.8381684090	-0.0415125805
        H	2.0848517291	-0.8974263240	0.3232009602
    """
}


for molecule in enantiomers:
    molecular_geometry = enantiomers[molecule]
    mol = gto.M(
        atom=molecular_geometry,
        basis='6-31G',
    )
    scfres = scf.RHF(mol)
    scfres.conv_tol = 1e-10
    scfres.conv_tol_grad = 1e-8
    scfres.kernel()

    # Run an adc2 calculation:
    state = adcc.adc2(scfres, n_singlets=10, conv_tol=1e-5)
    print(state.describe(rotatory_strengths=True))

    # Plot rotatory strengths
    plots = state.plot_spectrum(yaxis="rotatory_strength", width=0.005,
                                label=molecule)
plt.legend()
plt.show()
