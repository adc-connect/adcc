from adcc import AdcMethod
from adcc import LazyMp
from adcc import block as b
from adcc.functions import einsum
from adcc.AmplitudeVector import AmplitudeVector


def isrmvp_adc0(ground_state, dip, vec):
    assert isinstance(vec, AmplitudeVector)
    ph = (
            + 1.0 * einsum('ic,ac->ia', vec.ph, dip.vv) 
            - 1.0 * einsum('ka,ki->ia', vec.ph, dip.oo)
    )
    return AmplitudeVector(ph=ph)


def isrmvp_adc2(ground_state, dip, vec):
    assert isinstance(vec, AmplitudeVector)
    if dip.is_symmetric:
        dip_vo = dip.ov.transpose((1, 0))
    else:
        dip_vo = dip.vo.copy()
    p0 = ground_state.mp2_diffdm
    t2 = ground_state.t2(b.oovv)

    ph = (
            # product of the ph diagonal block with the singles block of the vector
            # zeroth order
            + 1.0 * einsum('ic,ac->ia', vec.ph, dip.vv)
            - 1.0 * einsum('ka,ki->ia', vec.ph, dip.oo)
            # second order
            # (2,1)
            - 1.0 * einsum('ic,jc,aj->ia', vec.ph, p0.ov, dip_vo)
            - 1.0 * einsum('ka,kb,bi->ia', vec.ph, p0.ov, dip_vo)
            - 1.0 * einsum('ic,ja,jc->ia', vec.ph, p0.ov, dip.ov) # h.c.
            - 1.0 * einsum('ka,ib,kb->ia', vec.ph, p0.ov, dip.ov) # h.c.
            # (2,2)
            - 0.25 * einsum('ic,mnef,mnaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.25 * einsum('ic,mnef,mncf,ae->ia', vec.ph, t2, t2, dip.vv) # h.c.
            # (2,3)
            - 0.5 * einsum('ic,mnce,mnaf,ef->ia', vec.ph, t2, t2, dip.vv)
            + 1.0 * einsum('ic,mncf,jnaf,jm->ia', vec.ph, t2, t2, dip.oo)
            # (2,4)
            + 0.25 * einsum('ka,mnef,inef,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.25 * einsum('ka,mnef,knef,mi->ia', vec.ph, t2, t2, dip.oo) # h.c.
            # (2,5)
            - 1.0 * einsum('ka,knef,indf,ed->ia', vec.ph, t2, t2, dip.vv)
            + 0.5 * einsum('ka,knef,imef,mn->ia', vec.ph, t2, t2, dip.oo)
            # (2,6)
            + 0.5 * einsum('kc,knef,inaf,ec->ia', vec.ph, t2, t2, dip.vv)
            - 0.5 * einsum('kc,mncf,inaf,km->ia', vec.ph, t2, t2, dip.oo)
            + 0.5 * einsum('kc,inef,kncf,ae->ia', vec.ph, t2, t2, dip.vv) # h.c.
            - 0.5 * einsum('kc,mnaf,kncf,mi->ia', vec.ph, t2, t2, dip.oo) # h.c.
            # (2,7)
            - 1.0 * einsum('kc,kncf,imaf,mn->ia', vec.ph, t2, t2, dip.oo)
            + 1.0 * einsum('kc,knce,inaf,ef->ia', vec.ph, t2, t2, dip.vv)

            # product of the ph-2p2h coupling block with the doubles block of the vector
            + 0.5 * (
                - 2.0 * einsum('ilad,ld->ia', vec.pphh, dip.ov)
                + 2.0 * einsum('ilad,lndf,fn->ia', vec.pphh, t2, dip_vo)
                + 2.0 * einsum('ilca,lc->ia', vec.pphh, dip.ov)
                - 2.0 * einsum('ilca,lncf,fn->ia', vec.pphh, t2, dip_vo)
                - 2.0 * einsum('klad,kled,ei->ia', vec.pphh, t2, dip_vo)
                - 2.0 * einsum('ilcd,nlcd,an->ia', vec.pphh, t2, dip_vo)
            )
    )

    pphh = (
            # product of the 2p2h-ph coupling block with the singles block of the vector
            + 0.5 * (
                (
                    - 1.0 * einsum('ia,bj->ijab', vec.ph, dip_vo)
                    + 1.0 * einsum('ia,jnbf,nf->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('ja,bi->ijab', vec.ph, dip_vo)
                    - 1.0 * einsum('ja,inbf,nf->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('ib,aj->ijab', vec.ph, dip_vo)
                    - 1.0 * einsum('ib,jnaf,nf->ijab', vec.ph, t2, dip.ov)
                    - 1.0 * einsum('jb,ai->ijab', vec.ph, dip_vo)
                    + 1.0 * einsum('jb,inaf,nf->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ka,ijeb,ke->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('kb,ijea,ke->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(2,3)
                +(
                    - 1.0 * einsum('ic,njab,nc->ijab', vec.ph, t2, dip.ov)
                    + 1.0 * einsum('jc,niab,nc->ijab', vec.ph, t2, dip.ov)
                ).antisymmetrise(0,1)
            )

            # product of the 2p2h diagonal block with the doubles block of the vector
            + 0.5 * (
                (
                    + 2.0 * einsum('ijcb,ac->ijab', vec.pphh, dip.vv)
                    - 2.0 * einsum('ijca,bc->ijab', vec.pphh, dip.vv)
                ).antisymmetrise(2,3)
                +(
                    - 2.0 * einsum('kjab,ki->ijab', vec.pphh, dip.oo)
                    + 2.0 * einsum('kiab,kj->ijab', vec.pphh, dip.oo)
                ).antisymmetrise(0,1)
            )
    )
    return AmplitudeVector(ph=ph, pphh=pphh)


DISPATCH = {
    "adc0": isrmvp_adc0,
    "adc1": isrmvp_adc0, # identical to ADC(0)
    "adc2": isrmvp_adc2,
}

    
def isrmatrix_vector_product(method, ground_state, dips, vec):
    """Compute the matrix-vector product of an ISR one-particle operator
    for the provided ADC method.
    The product was derived using the original equations from the work of
    Schirmer and Trofimov (J. Schirmer and A. B. Trofimov, “Intermediate state
    representation approach to physical properties of electronically excited
    molecules,” J. Chem. Phys. 120, 11449–11464 (2004).).

    Parameters
    ----------
    method: str, AdcMethod
        The  method to use for the computation of the matrix-vector product
    ground_state : adcc.LazyMp
        The MP ground state
    dips : OneParticleOperator or list of OneParticleOperator
        One-particle matrix elements associated with the dipole operator        
    vec: AmplitudeVector
        A vector with singles and doubles block

    Returns
    -------
    AmplitudeVector or list of AmplitudeVector
    """
    if not isinstance(method, AdcMethod):
        method = AdcMethod(method)
    if method.name not in DISPATCH:
        raise NotImplementedError(f"isrmatrix_vector_product is not implemented for {method.name}.")
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    unpack = False
    if not isinstance(dips, list):
        unpack = True
        dips = [dips]

    ret = [DISPATCH[method.name](ground_state, dip, vec) for dip in dips]
    if unpack:
        assert len(ret) == 1
        ret = ret[0]
    return ret
