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
import numpy as np
#import scipy.linalg as sp


class qed_matrix_from_diag_adc:
    def __init__(self, exstates, refstate):
        self.s2s = exstates.s2s_dipole_moments_qed
        self.tdm = exstates.transition_dipole_moments_qed
        self.h1 = exstates.qed_second_order_ph_ph_couplings
        self.coupl = np.array(refstate.coupling)
        self.freq = np.linalg.norm(refstate.frequency)
        self.full_freq = np.linalg.norm(np.real(refstate.freq_with_loss)) +\
            np.linalg.norm(np.imag(refstate.freq_with_loss)) * 1j
        self.n_adc = len(exstates.excitation_energy)
        self.exc_en = exstates.excitation_energy

    def loop_helper(self, s2s_tensor):
        ret = np.array([[self.coupl.dot(s2s)
                        for s2s in s2s_block]
                        for s2s_block in s2s_tensor])
        ret *= - np.sqrt(self.freq / 2) * np.sqrt(2 * self.freq)
        return ret

    def first_order_coupling(self):

        # build the blocks of the matrix

        tdm_block = np.empty(self.n_adc)

        for i, tdm in enumerate(self.tdm):
            tdm_block[i] = self.coupl.dot(tdm)
        tdm_block *= np.sqrt(2 * self.freq)

        s2s_block = self.loop_helper(self.s2s["qed_adc1_off_diag"])

        tdm_block *= - np.sqrt(self.freq / 2)

        if np.iscomplex(self.full_freq):
            self.freq = self.full_freq

        elec_block = np.diag(self.exc_en)
        phot_block = np.diag(self.exc_en + self.freq)

        # build the matrix

        matrix_upper = np.vstack((elec_block, tdm_block.reshape((1, self.n_adc)),
                                  s2s_block))
        matrix_middle = np.concatenate((tdm_block, np.array([self.freq]),
                                        np.zeros(self.n_adc)))
        matrix_lower = np.vstack((s2s_block, np.zeros((1, self.n_adc)),
                                  phot_block))

        matrix = np.hstack((matrix_upper,
                            matrix_middle.reshape((len(matrix_middle), 1)),
                            matrix_lower))

        #if np.iscomplex(self.full_freq):
        #    return sp.eig(matrix)
        #else:
        #    return sp.eigh(matrix)
        return matrix

    def second_order_coupling(self):

        # tdm part

        qed_adc2_tdm_vec = np.empty(self.n_adc)

        for i, tdm in enumerate(self.tdm):
            qed_adc2_tdm_vec[i] = self.coupl.dot(tdm)
        qed_adc2_tdm_vec *= - np.sqrt(self.freq / 2) * np.sqrt(2 * self.freq)

        # s2s_dipole parts of the ph_ph blocks

        qed_adc1_off_diag_block = self.loop_helper(self.s2s["qed_adc1_off_diag"])

        qed_adc2_diag_block = self.loop_helper(self.s2s["qed_adc2_diag"])
        # missing factor from state.s2s_dipole_moments_qed_adc2_diag
        # TODO: commit to one way of defining these factors
        # within the approx method
        qed_adc2_diag_block *= np.sqrt(self.freq / 2)

        qed_adc2_edge_block_couple = self.loop_helper(
            self.s2s["qed_adc2_edge_couple"])
        # missing factor from state.s2s_dipole_moments_qed_adc2_edge
        qed_adc2_edge_block_couple *= np.sqrt(self.freq)

        qed_adc2_edge_block_phot_couple = self.loop_helper(
            self.s2s["qed_adc2_edge_phot_couple"])
        # missing factor from state.s2s_dipole_moments_qed_adc2_edge
        qed_adc2_edge_block_phot_couple *= np.sqrt(self.freq)

        # s2s_dipole parts of the pphh_ph and ph_pphh blocks

        qed_adc2_ph_pphh_couple_block = self.loop_helper(
            self.s2s["qed_adc2_ph_pphh"])

        qed_adc2_pphh_ph_phot_couple_block = self.loop_helper(
            self.s2s["qed_adc2_pphh_ph"])

        # we still need the H_1 expectation value "as property"

        qed_adc2_couple_block = np.sqrt(self.freq / 2) *\
            self.h1["couple"]
        qed_adc2_phot_couple_block = np.sqrt(self.freq / 2) *\
            self.h1["phot_couple"]

        # build the blocks of the matrix

        if np.iscomplex(self.full_freq):
            self.freq = self.full_freq

        single_excitation_states = np.ones(self.n_adc)

        elec_block = np.diag(self.exc_en) + qed_adc2_diag_block

        phot_block = np.diag(self.exc_en) + qed_adc2_diag_block * 2 +\
            np.diag(single_excitation_states) * self.freq

        phot2_block = np.diag(self.exc_en) + qed_adc2_diag_block * 3 +\
            np.diag(single_excitation_states) * 2 * self.freq

        couple_block = qed_adc1_off_diag_block + qed_adc2_ph_pphh_couple_block +\
            qed_adc2_couple_block

        phot_couple_block = qed_adc1_off_diag_block +\
            qed_adc2_pphh_ph_phot_couple_block +\
            qed_adc2_phot_couple_block

        # build the matrix

        matrix_1 = np.vstack((elec_block, qed_adc2_tdm_vec.reshape((1, self.n_adc)),  # noqa: E501
                              phot_couple_block, np.zeros((1, self.n_adc)),
                              qed_adc2_edge_block_phot_couple))
        matrix_2 = np.concatenate((qed_adc2_tdm_vec, np.array([self.freq]),
                                   np.zeros(self.n_adc), np.array([0]),
                                   np.zeros(self.n_adc)))
        matrix_3 = np.vstack((couple_block, np.zeros((1, self.n_adc)), phot_block,
                              np.sqrt(2) * qed_adc2_tdm_vec.reshape((1, self.n_adc)),  # noqa: E501
                              np.sqrt(2) * phot_couple_block))
        matrix_4 = np.concatenate((np.zeros(self.n_adc), np.array([0]),
                                   np.sqrt(2) * qed_adc2_tdm_vec,
                                   2 * np.array([self.freq]), np.zeros(self.n_adc)))
        matrix_5 = np.vstack((qed_adc2_edge_block_couple, np.zeros((1, self.n_adc)),
                              np.sqrt(2) * couple_block, np.zeros((1, self.n_adc)),
                              phot2_block))

        matrix = np.hstack((matrix_1, matrix_2.reshape((len(matrix_2), 1)),
                            matrix_3, matrix_4.reshape((len(matrix_4), 1)),
                            matrix_5))

        #if np.iscomplex(self.full_freq):
        #    return sp.eig(matrix)
        #else:
        #    return sp.eigh(matrix)
        return matrix
