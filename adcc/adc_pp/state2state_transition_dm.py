#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2018 by the adcc authors
##
## This file is part of adcc.
##
## adcc is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published
## by the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## adcc is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with adcc. If not, see <http://www.gnu.org/licenses/>.
##
## ---------------------------------------------------------------------
from adcc import block as b
from adcc.LazyMp import LazyMp
from adcc.AdcMethod import IsrMethod
from adcc.functions import einsum
from adcc.Intermediates import Intermediates
from adcc.AmplitudeVector import AmplitudeVector
from adcc.OneParticleDensity import OneParticleDensity
from adcc.NParticleOperator import OperatorSymmetry

from .util import check_doubles_amplitudes, check_singles_amplitudes


def s2s_tdm_isr0(mp, amplitude_l, amplitude_r, intermediates):
    check_singles_amplitudes([b.o, b.v], amplitude_l, amplitude_r)
    ul1 = amplitude_l.ph
    ur1 = amplitude_r.ph

    dm = OneParticleDensity(mp, symmetry=OperatorSymmetry.NOSYMMETRY)
    dm.oo = -einsum('ja,ia->ij', ul1, ur1)
    dm.vv = einsum('ia,ib->ab', ul1, ur1)
    return dm


def s2s_tdm_isr1(mp, amplitude_l, amplitude_r, intermediates):
    dm = s2s_tdm_isr0(mp, amplitude_l, amplitude_r, intermediates)

    try:
        check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
        ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
        ur1, ur2 = amplitude_r.ph, amplitude_r.pphh
        dm.ov += -2.0 * einsum("jb,ijab->ia", ul1, ur2)
        dm.vo += -2.0 * einsum("ijab,jb->ai", ul2, ur1)
    except ValueError:
        pass

    return dm


def s2s_tdm_isr2(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_isr1(mp, amplitude_l, amplitude_r, intermediates)

    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh

    t2 = mp.t2(b.oovv)
    p0 = mp.mp2_diffdm
    p1_oo = dm.oo.evaluate()  # ADC(1) tdm
    p1_vv = dm.vv.evaluate()  # ADC(1) tdm

    # ADC(2) ISR intermediate (TODO Move to intermediates)
    rul1 = einsum('ijab,jb->ia', t2, ul1).evaluate()
    rur1 = einsum('ijab,jb->ia', t2, ur1).evaluate()

    dm.oo += (
        - 2.0 * einsum('ikab,jkab->ij', ur2, ul2)
        + 0.5 * einsum('ik,kj->ij', p1_oo, p0.oo)
        + 0.5 * einsum('ik,kj->ij', p0.oo, p1_oo)
        - 0.5 * einsum('ikcd,lk,jlcd->ij', t2, p1_oo, t2)
        + 1.0 * einsum('ikcd,jkcb,db->ij', t2, t2, p1_vv)
        - 0.5 * einsum('ia,jkac,kc->ij', ur1, t2, rul1)
        - 0.5 * einsum('ikac,kc,ja->ij', t2, rur1, ul1)
        - 1.0 * einsum('ia,ja->ij', rul1, rur1)
    )
    dm.vv += (
        + 2.0 * einsum('ijac,ijbc->ab', ul2, ur2)
        - 0.5 * einsum("ac,cb->ab", p1_vv, p0.vv)
        - 0.5 * einsum("ac,cb->ab", p0.vv, p1_vv)
        - 0.5 * einsum("klbc,klad,cd->ab", t2, t2, p1_vv)
        + 1.0 * einsum("klbc,jk,jlac->ab", t2, p1_oo, t2)
        + 0.5 * einsum("ikac,kc,ib->ab", t2, rul1, ur1)
        + 0.5 * einsum("ia,ikbc,kc->ab", ul1, t2, rur1)
        + 1.0 * einsum("ia,ib->ab", rur1, rul1)
    )

    # (TODO Move to intermediates)
    p1_ov = -2.0 * einsum("jb,ijab->ia", ul1, ur2).evaluate()
    p1_vo = -2.0 * einsum("ijab,jb->ai", ul2, ur1).evaluate()

    dm.ov += (
        - einsum("ijab,bj->ia", t2, p1_vo)
        - einsum("ib,ba->ia", p0.ov, p1_vv)
        + einsum("ij,ja->ia", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ia", ur1, t2, ul2)
        - einsum("ikcd,jkcd,ja->ia", t2, ul2, ur1)
    )
    dm.vo += (
        - einsum("ijab,jb->ai", t2, p1_ov)
        - einsum("ib,ab->ai", p0.ov, p1_vv)
        + einsum("ji,ja->ai", p1_oo, p0.ov)
        - einsum("ib,klca,klcb->ai", ul1, t2, ur2)
        - einsum("ikcd,jkcd,ja->ai", t2, ur2, ul1)
    )
    return dm


def s2s_tdm_adc3(mp, amplitude_l, amplitude_r, intermediates):
    check_doubles_amplitudes([b.o, b.o, b.v, b.v], amplitude_l, amplitude_r)
    dm = s2s_tdm_adc2(mp, amplitude_l, amplitude_r, intermediates)

    #ADC amplitudes
    ul1, ul2 = amplitude_l.ph, amplitude_l.pphh
    ur1, ur2 = amplitude_r.ph, amplitude_r.pphh
    
    t2_1 = mp.t2(b.oovv)
    t1_2 = mp.diffdm(level=2).ov
    t2_2 = mp.td2(b.oovv)
    t3_2 = mp.tt2(b.ooovvv)

    #mp2 second order correction
    p0_2_oo = mp.mp2_diffdm.oo
    p0_2_vv = mp.mp2_diffdm.vv
    p0_2_ov = mp.mp2_diffdm.ov
    
    
    #third order correction 
    p0_3_oo = mp.mp3_dm_correction.oo
    p0_3_vv = mp.mp3_dm_correction.vv
    p0_3_ov = mp.mp3_dm_correction.ov

    dm.oo = (  
	    - 1 * einsum("ja,ia->ij", ul1, ur1)  
	    - 2 * einsum("jkab,ikab->ij", ul2, ur2)  
	    - 2 * einsum("ib,jb->ij", 
                    einsum("ka,ikab->ib", ul1, ur2), t1_2)  
	    - 2 * einsum("jb,ib->ij", 
                    einsum("jkab,ka->jb", ul2, ur1), t1_2)  
	    - 0.5 * einsum("jk,ik->ij", 
                      einsum("ja,ka->jk", ul1, ur1), p0_2_oo) 
	    - 0.5 * einsum("jk,ik->ij", 
                      einsum("ja,ka->jk", ul1, ur1), p0_3_oo)  
	    - 0.5 * einsum("ik,jk->ij", 
                      einsum("ka,ia->ik", ul1, ur1), p0_2_oo)       
	    - 0.5 * einsum("ik,jk->ij", 
                      einsum("ka,ia->ik", ul1, ur1), p0_3_oo)  
	    + 1 * einsum("jklc,iklc->ij", 
                    einsum("lb,jkbc->jklc", ur1, t2_1), 
                          einsum("la,ikac->iklc", ul1, t2_1))  
	    + 1 * einsum("jklc,iklc->ij", 
                    einsum("lb,jkbc->jklc", ur1, t2_2), 
                          einsum("la,ikac->iklc", ul1, t2_1))  
	    + 1 * einsum("jklc,iklc->ij", 
                    einsum("lb,jkbc->jklc", ur1, t2_1), 
                          einsum("la,ikac->iklc", ul1, t2_2))  
	    + 0.5 * einsum("ilbc,jlbc->ij", 
                      einsum("kl,ikbc->ilbc", 
                            einsum("ka,la->kl", ul1, ur1), t2_1), t2_2)  
	    + 0.5 * einsum("ikbc,jkbc->ij", 
                      einsum("kl,ilbc->ikbc", 
                            einsum("la,ka->kl", ul1, ur1), t2_1), t2_1)
	    + 0.5 * einsum("jlbc,ilbc->ij", 
                      einsum("kl,jkbc->jlbc", 
                            einsum("la,ka->kl", ul1, ur1), t2_1), t2_2) 
	    + 0.5 * einsum("jb,ib->ij", 
                      einsum("kc,jkbc->jb", 
                            einsum("la,klac->kc", ul1, t2_1), t2_1), ur1)  
	    + 0.5 * einsum("jb,ib->ij", 
                      einsum("kc,jkbc->jb", 
                            einsum("la,klac->kc", ul1, t2_1), t2_2), ur1)  
	    + 0.5 * einsum("jb,ib->ij", 
                      einsum("kc,jkbc->jb", 
                            einsum("la,klac->kc", ul1, t2_2), t2_1), ur1) 
	    - 1 * einsum("jc,ic->ij", 
                    einsum("lb,jlbc->jc", ur1, t2_2), 
                        einsum("ka,ikac->ic", ul1, t2_1))  
	    - 1 * einsum("jc,ic->ij", 
                    einsum("kb,jkbc->jc", ur1, t2_1), 
                          einsum("la,ilac->ic", ul1, t2_1)) 
	    - 1 * einsum("jc,ic->ij", 
                    einsum("kb,jkbc->jc", ur1, t2_1), 
                          einsum("la,ilac->ic", ul1, t2_2))  
	    - 0.5 * einsum("ia,ja->ij", 
                      einsum("lc,ilac->ia", 
                            einsum("kb,klbc->lc", ur1, t2_1), t2_1), ul1)  
	    - 0.5 * einsum("ia,ja->ij", 
                      einsum("lc,ilac->ia", 
                            einsum("kb,klbc->lc", ur1, t2_2), t2_1), ul1)  
	    - 0.5 * einsum("ia,ja->ij", 
                      einsum("lc,ilac->ia", 
                            einsum("kb,klbc->lc", ur1, t2_1), t2_2), ul1)  
 )
    dm.vv = (
	    + 1 * einsum("ia,ib->ab", ul1, ur1)  
	    + 2 * einsum("ijac,ijbc->ab", ul2, ur2)  
	    - 2 * einsum("ia,ib->ab", 
                    einsum("ijac,jc->ia", ul2, ur1), t1_2)  
	    + 2 * einsum("jb,ja->ab", 
                    einsum("ic,ijbc->jb", ul1, ur2), t1_2)  
	    - 0.5 * einsum("ib,ia->ab", 
                      einsum("ic,bc->ib", ur1, p0_2_vv), ul1)  
	    - 0.5 * einsum("ib,ia->ab", 
                      einsum("ic,bc->ib", ur1, p0_3_vv), ul1)  
	    - 0.5 * einsum("ia,ib->ab", 
                      einsum("ic,ac->ia", ul1, p0_2_vv), ur1)  
	    - 0.5 * einsum("ia,ib->ab", 
                      einsum("ic,ac->ia", ul1, p0_3_vv), ur1)  
	    + 1 * einsum("ka,kb->ab", 
                    einsum("jd,jkad->ka", ur1, t2_1), 
                          einsum("ic,ikbc->kb", ul1, t2_1))  
	    + 1 * einsum("ka,kb->ab", 
                    einsum("jd,jkad->ka", ur1, t2_1), 
                          einsum("ic,ikbc->kb", ul1, t2_2))  
	    + 1 * einsum("ka,kb->ab", 
                    einsum("jc,jkac->ka", ur1, t2_2), 
                          einsum("id,ikbd->kb", ul1, t2_1))  
	    + 0.5 * einsum("ib,ia->ab", 
                      einsum("kd,ikbd->ib", 
                            einsum("jc,jkcd->kd", ur1, t2_1), t2_1), ul1) 
	    + 0.5 * einsum("ib,ia->ab", 
                      einsum("kd,ikbd->ib", 
                            einsum("jc,jkcd->kd", ur1, t2_2), t2_1), ul1) 
	    + 0.5 * einsum("ib,ia->ab", 
                      einsum("kd,ikbd->ib", 
                            einsum("jc,jkcd->kd", ur1, t2_1), t2_2), ul1)  
	    - 1 * einsum("ikad,ikbd->ab", 
                    einsum("ij,jkad->ikad", 
                          einsum("ic,jc->ij", ul1, ur1), t2_1), t2_1)  
	    - 1 * einsum("ikad,ikbd->ab", 
                    einsum("ij,jkad->ikad", 
                          einsum("ic,jc->ij", ul1, ur1), t2_1), t2_2) 
	    - 1 * einsum("jkbc,jkac->ab", 
                    einsum("ij,ikbc->jkbc", 
                          einsum("id,jd->ij", ul1, ur1), t2_1), t2_2)  
	    - 0.5 * einsum("ijka,ijkb->ab", 
                      einsum("id,jkad->ijka", ur1, t2_1), 
                            einsum("ic,jkbc->ijkb", ul1, t2_1)) 
	    - 0.5 * einsum("ijka,ijkb->ab", 
                      einsum("id,jkad->ijka", ur1, t2_1), 
                            einsum("ic,jkbc->ijkb", ul1, t2_2)) 
	    - 0.5 * einsum("ja,jb->ab",
                      einsum("kc,jkac->ja", 
                            einsum("id,ikcd->kc", ul1, t2_1), t2_1), ur1)  
	    - 0.5 * einsum("ja,jb->ab", 
                      einsum("kc,jkac->ja", 
                            einsum("id,ikcd->kc", ul1, t2_2), t2_1), ur1) 
	    - 0.5 * einsum("ja,jb->ab", 
                      einsum("kc,jkac->ja", 
                            einsum("id,ikcd->kc", ul1, t2_1), t2_2), ur1)  
	    - 0.5 * einsum("ijka,ijkb->ab", 
                      einsum("ic,jkac->ijka", ur1, t2_2), 
                            einsum("id,jkbd->ijkb", ul1, t2_1)) 
)
    dm.ov = (
	    - 2 * einsum("jb,ijab->ia", ul1, ur2)  
	    + 1 * einsum("ib,ab->ia", 
                    einsum("jc,ijbc->ib", ul1, ur2), p0_2_vv) 
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jb,ib->ij", ul1, t1_2), ur1) 
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jb,ib->ij", ul1, p0_3_ov), ur1)  
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jb,ib->ij", ul1, ur1), t1_2) 
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jb,ib->ij", ul1, ur1), p0_3_ov)  
	    - 1 * einsum("ja,ij->ia", 
                    einsum("kb,jkab->ja", ul1, ur2), p0_2_oo)  
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jkbc,ikbc->ij", ul2, t2_1), ur1)  
	    - 1 * einsum("ij,ja->ia", 
                    einsum("jkbc,ikbc->ij", ul2, t2_2), ur1) 
	    - 1 * einsum("ijkc,jkac->ia", 
                    einsum("jkbc,ib->ijkc", ul2, ur1), t2_1)   
	    - 1 * einsum("ijkc,jkac->ia", 
                    einsum("jkbc,ib->ijkc", ul2, ur1), t2_2)   
	    - 2 * einsum("kb,ikab->ia", 
                    einsum("jkbc,jc->kb", ul2, ur1), t2_1)   
	    - 2 * einsum("kb,ikab->ia", 
                    einsum("jkbc,jc->kb", ul2, ur1), t2_2)   
	    + 1 * einsum("kc,ikac->ia", 
                    einsum("jk,jc->kc", 
                          einsum("kb,jb->jk", ul1, ur1), t1_2), t2_1)   
	    + 1 * einsum("kc,ikac->ia", 
                    einsum("jk,jc->kc", 
                          einsum("jb,kb->jk", ur1, t1_2), ul1), t2_1)   
	    + 0.5 * einsum("ij,ja->ia", 
                      einsum("ikcd,jkcd->ij", 
                            einsum("lb,iklbcd->ikcd", ul1, t3_2), t2_1), ur1)   
	    + 0.5 * einsum("ijkb,jkab->ia", 
                      einsum("id,jkbd->ijkb", 
                            einsum("lc,ilcd->id", ul1, t2_1), t2_1), ur2)   
	    + 0.5 * einsum("ld,ilad->ia", 
                      einsum("jklb,jkbd->ld", 
                            einsum("lc,jkbc->jklb", ul1, ur2), t2_1), t2_1)   
	    + 0.5 * einsum("iklc,klac->ia", 
                      einsum("ib,klbc->iklc", ur1, t2_1), 
                            einsum("jd,jklacd->klac", ul1, t3_2))   
	    + 0.5 * einsum("il,la->ia", 
                      einsum("ikbc,klbc->il", ur2, t2_1), 
                            einsum("jd,jlad->la", ul1, t2_1))   
	    + 0.5 * einsum("kd,ikad->ia", 
                      einsum("kl,ld->kd", 
                            einsum("jlbc,jkbc->kl", ur2, t2_1), ul1), t2_1)   
	    - 1 * einsum("ij,ja->ia", 
                    einsum("ic,jc->ij", 
                          einsum("kb,ikbc->ic", ul1, t2_1), t1_2), ur1)   
	    - 1 * einsum("ijkc,jkac->ia", 
                    einsum("ikld,jlcd->ijkc", 
                          einsum("kb,ilbd->ikld", ul1, t2_1), t2_1), ur2)   
	    - 1 * einsum("ik,ka->ia", 
                    einsum("ib,kb->ik", ur1, t1_2), 
                          einsum("jc,jkac->ka", ul1, t2_1))   
	    - 1 * einsum("ijld,jlad->ia", 
                    einsum("ijkb,klbd->ijld", 
                          einsum("jc,ikbc->ijkb", ul1, ur2), t2_1), t2_1)   
	    - 1 * einsum("kc,ikac->ia", 
                    einsum("jb,jkbc->kc", ur1, t2_1), 
                          einsum("ld,iklacd->ikac", ul1, t3_2))   
	    - 1 * einsum("kc,ikac->ia", 
                    einsum("jb,jkbc->kc", 
                          einsum("ld,jlbd->jb", ul1, ur2), t2_1), t2_1)   
	    - 0.5 * einsum("jkcd,ijkacd->ia", 
                      einsum("jklc,ld->jkcd", 
                            einsum("lb,jkbc->jklc", ur1, t2_1), ul1), t3_2)   
	    - 0.5 * einsum("klbc,iklabc->ia", 
                      einsum("jl,jkbc->klbc", 
                            einsum("ld,jd->jl", ul1, ur1), t2_1), t3_2)   
	    - 0.25 * einsum("ijkl,jkla->ia", 
                       einsum("ilcd,jkcd->ijkl", t2_1, t2_1), 
                             einsum("lb,jkab->jkla", ul1, ur2))   
	    - 0.25 * einsum("ijkl,jkla->ia", 
                       einsum("ijbc,klbc->ijkl", ur2, t2_1), 
                             einsum("jd,klad->jkla", ul1, t2_1))  
)

    dm.vo = (
	    - 2 * einsum("ijab,jb->ai", ul2, ur1) 
	    + 1 * einsum("ik,ka->ai", 
                    einsum("jkbc,ijbc->ik", ur2, t2_1), ul1) 
	    + 1 * einsum("ik,ka->ai", 
                    einsum("jkbc,ijbc->ik", ur2, t2_2), ul1) 
	    + 1 * einsum("ijkb,jkab->ai", 
                    einsum("ic,jkbc->ijkb", ul1, ur2), t2_1)  
	    + 1 * einsum("ijkb,jkab->ai", 
                    einsum("ic,jkbc->ijkb", ul1, ur2), t2_2)  
	    + 1 * einsum("ka,ik->ai", 
                    einsum("jkab,jb->ka", ul2, ur1), p0_2_oo) 
	    + 1 * einsum("ib,ab->ai", 
                    einsum("ijbc,jc->ib", ul2, ur1), p0_2_vv)  
	    - 1 * einsum("ij,ja->ai", 
                    einsum("jb,ib->ij", ur1, t1_2), ul1)  
	    - 1 * einsum("ij,ja->ai", 
                    einsum("jb,ib->ij", ur1, p0_3_ov), ul1)  
	    - 1 * einsum("ij,ja->ai", 
                    einsum("ib,jb->ij", ul1, ur1), t1_2) 
	    - 1 * einsum("ij,ja->ai", 
                    einsum("ib,jb->ij", ul1, ur1), p0_3_ov)  
	    + 2 * einsum("jb,ijab->ai", 
                    einsum("kc,jkbc->jb", ul1, ur2), t2_1) 
	    + 2 * einsum("jb,ijab->ai", 
                    einsum("kc,jkbc->jb", ul1, ur2), t2_2) 
	    + 1 * einsum("jc,ijac->ai", 
                    einsum("jk,kc->jc", 
                          einsum("kb,jb->jk", ul1, ur1), t1_2), t2_1)  
	    + 1 * einsum("kb,ikab->ai", 
                    einsum("jk,jb->kb", 
                          einsum("jc,kc->jk", ul1, t1_2), ur1), t2_1)  
	    + 1 * einsum("lc,ilac->ai", 
                    einsum("kb,klbc->lc", 
                          einsum("jkbd,jd->kb", ul2, ur1), t2_1), t2_1)  
	    + 0.5 * einsum("il,la->ai", 
                      einsum("ikcd,klcd->il", 
                            einsum("jb,ijkbcd->ikcd", ur1, t3_2), t2_1), ul1)  
	    + 0.5 * einsum("klac,iklc->ai", 
                      einsum("jb,jklabc->klac", ur1, t3_2), 
                            einsum("id,klcd->iklc", ul1, t2_1))  
	    + 0.5 * einsum("ijlb,jlab->ai", 
                      einsum("id,jlbd->ijlb", 
                            einsum("kc,ikcd->id", ur1, t2_1), t2_1), ul2) 
	    + 0.5 * einsum("la,il->ai", 
                      einsum("kc,klac->la", ur1, t2_1), 
                            einsum("ijbd,jlbd->il", ul2, t2_1))
	    + 0.5 * einsum("kc,ikac->ai", 
                      einsum("jklb,jlbc->kc", 
                            einsum("jlbd,kd->jklb", ul2, ur1), t2_1), t2_1)  
	    - 1 * einsum("ik,ka->ai", 
                    einsum("ic,kc->ik", 
                          einsum("jb,ijbc->ic", ur1, t2_1), t1_2), ul1)  
	    - 1 * einsum("ka,ik->ai", 
                    einsum("jb,jkab->ka", ur1, t2_1), 
                          einsum("ic,kc->ik", ul1, t1_2)) 
	    - 1 * einsum("jkbc,ijkabc->ai", 
                    einsum("kc,jb->jkbc", 
                          einsum("ld,klcd->kc", ul1, t2_1), ur1), t3_2)  
	    - 1 * einsum("ijlb,jlab->ai", 
                    einsum("ijkd,klbd->ijlb", 
                          einsum("jc,ikcd->ijkd", ur1, t2_1), t2_1), ul2)  
	    - 1 * einsum("iklc,klac->ai", 
                    einsum("ijkb,jlbc->iklc", 
                          einsum("ijbd,kd->ijkb", ul2, ur1), t2_1), t2_1)  
	    - 0.5 * einsum("jkbc,ijkabc->ai", 
                      einsum("jklc,lb->jkbc", 
                            einsum("ld,jkcd->jklc", ul1, t2_1), ur1), t3_2)  
	    - 0.5 * einsum("jkbc,ijkabc->ai", 
                      einsum("jl,klbc->jkbc", 
                            einsum("ld,jd->jl", ul1, ur1), t2_1), t3_2)  
	    - 0.5 * einsum("lc,ilac->ai", 
                      einsum("jl,jc->lc", 
                            einsum("jkbd,klbd->jl", ul2, t2_1), ur1), t2_1) 
	    - 0.25 * einsum("ijkl,jkla->ai", 
                       einsum("ikcd,jlcd->ijkl", t2_1, t2_1), 
                             einsum("jlab,kb->jkla", ul2, ur1)) 
	    - 0.25 * einsum("iklc,klac->ai", 
                       einsum("ijkl,jc->iklc", 
                             einsum("ijbd,klbd->ijkl", ul2, t2_1), ur1), t2_1)  
)    
    return dm
# Ref: https://doi.org/10.1080/00268976.2013.859313
DISPATCH = {
    "isr0": s2s_tdm_isr0,
    "isr1s": s2s_tdm_isr0,  # Identical to ISR(0)
    "isr1": s2s_tdm_isr1,
    "isr2d": s2s_tdm_isr2,  # Identical to ISR(2)
    "isr2": s2s_tdm_isr2,
}


def state2state_transition_dm(method, ground_state, amplitude_from,
                              amplitude_to, intermediates=None):
    """
    Compute the state to state transition density matrix
    state in the MO basis using the intermediate-states representation.

    Parameters
    ----------
    method : str, IsrMethod
        The method to use for the computation (e.g. "isr2")
    ground_state : LazyMp
        The ground state upon which the excitation was based
    amplitude_from : AmplitudeVector
        The amplitude vector of the state to start from
    amplitude_to : AmplitudeVector
        The amplitude vector of the state to excite to
    intermediates : adcc.Intermediates
        Intermediates from the ADC calculation to reuse
    """
    if not isinstance(method, IsrMethod):
        method = IsrMethod(method)
    if not isinstance(ground_state, LazyMp):
        raise TypeError("ground_state should be a LazyMp object.")
    if not isinstance(amplitude_from, AmplitudeVector):
        raise TypeError("amplitude_from should be an AmplitudeVector object.")
    if not isinstance(amplitude_to, AmplitudeVector):
        raise TypeError("amplitude_to should be an AmplitudeVector object.")
    if intermediates is None:
        intermediates = Intermediates(ground_state)

    if method.name not in DISPATCH:
        raise NotImplementedError("state2state_transition_dm is not implemented "
                                  f"for {method.name}.")
    else:
        # final state is on the bra side/left (complex conjugate)
        # see ref https://doi.org/10.1080/00268976.2013.859313, appendix A2
        ret = DISPATCH[method.name](ground_state, amplitude_to, amplitude_from,
                                    intermediates)
        return ret.evaluate()
