import unittest
import numpy as np
from adcc.misc import expand_test_templates
from adcc.testdata.cache import cache
from adcc.adc_pp.isrmatrix_vector_product import isrmatrix_vector_product
from adcc.OneParticleOperator import product_trace
from adcc.adc_pp.state2state_transition_dm import state2state_transition_dm

methods = ["adc0", "adc1", "adc2"]

@expand_test_templates(methods)
class TestIsrmatrixVectorProduct(unittest.TestCase):
    def base_test(self, system, method, kind, op_kind):
        state = cache.adc_states[system][method][kind]
        mp = state.ground_state
        if op_kind == "electric": # example of a symmetric operator
            dips = state.reference_state.operators.electric_dipole
        elif op_kind == "magnetic": # example of an asymmetric operator
            dips = state.reference_state.operators.magnetic_dipole
        else:
            raise NotImplementedError(
                    "Tests are only implemented for electric and magnetic dipole operators."
            )

        # computing Y_m @ B @ Y_n yields the state-to-state transition dipole moments (n->m)
        # (for n not equal to m)
        # they can either be obtained using isrmatrix_vector_product or
        # via the state-to-state transition density matrices
        # (the second method serves as a reference here)
            
        for excitation1 in state.excitations:
            product_vecs = isrmatrix_vector_product(
                    method, mp, dips, excitation1.excitation_vector
            )
            for excitation2 in state.excitations:
                s2s_tdm = [excitation2.excitation_vector @ pv for pv in product_vecs]
                tdm = state2state_transition_dm(
                        state.property_method,
                        state.ground_state,
                        excitation1.excitation_vector,
                        excitation2.excitation_vector,
                        state.matrix.intermediates
                )
                s2s_tdm_ref = np.array([product_trace(tdm, dip) for dip in dips])
                np.testing.assert_allclose(s2s_tdm, s2s_tdm_ref, atol=1e-12)

    #
    # General
    #
    def template_h2o_sto3g_singlets_elec(self, method):
        self.base_test("h2o_sto3g", method, "singlet", "electric")

    def template_h2o_def2tzvp_singlets_elec(self, method):
        self.base_test("h2o_def2tzvp", method, "singlet", "electric")

    def template_h2o_sto3g_triplets_elec(self, method):
        self.base_test("h2o_sto3g", method, "triplet", "electric")

    def template_h2o_def2tzvp_triplets_elec(self, method):
        self.base_test("h2o_def2tzvp", method, "triplet", "electric")

    def template_cn_sto3g_elec(self, method):
        self.base_test("cn_sto3g", method, "state", "electric")

    def template_cn_ccpvdz_elec(self, method):
        self.base_test("cn_ccpvdz", method, "state", "electric")

    def template_h2o_sto3g_singlets_mag(self, method):
        self.base_test("h2o_sto3g", method, "singlet", "magnetic")

    def template_h2o_sto3g_triplets_mag(self, method):
        self.base_test("h2o_sto3g", method, "triplet", "magnetic")

    def template_cn_sto3g_mag(self, method):
        self.base_test("cn_sto3g", method, "state", "magnetic")
