#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab
## ---------------------------------------------------------------------
##
## Copyright (C) 2020 by the adcc authors
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
import adcc
import pytest
from pytest import approx

from adcc import InputError
from .testdata_cache import testdata_cache


class TestWorkflow:
    def test_validate_state_parameters_rhf_pp(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("h2o_sto3g", case="gen")
        matrix.method = AdcMethod("adc2")

        assert (3, "any", None) == validate_state_parameters(
            matrix, n_states=3)
        assert (4, "singlet", None) == validate_state_parameters(
            matrix, n_states=4, kind="singlet")
        assert (2, "triplet", None) == validate_state_parameters(
            matrix, n_states=2, kind="triplet")
        assert (2, "triplet", None) == validate_state_parameters(
            matrix, n_triplets=2)
        assert (6, "singlet", None) == validate_state_parameters(
            matrix, n_singlets=6)

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_singlets=-2),   # Negative number of states requested
            dict(n_spin_flip=2),   # Invalid kind of states for RHF
            dict(n_states=2, n_singlets=2),      # States of two sorts
            dict(n_triplets=2, n_singlets=2),    # States of two sorts
            dict(n_states=2, n_spin_flip=2),     # States of two sorts
            dict(n_triplets=2, kind="singlet"),  # kind and n_ do not agree
            dict(n_states=2, kind="bla"),      # Kind invaled
            dict(n_states=2, kind="doublet"),  # Kind invalid for PP-ADC
            dict(n_states=2, is_alpha=True),   # Parameter only for IP/EA-ADC
            dict(n_states=2, is_alpha=False),  # Parameter only for IP/EA-ADC
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_uhf_pp(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("cn_sto3g", case="gen")
        matrix.method = AdcMethod("adc2")

        assert (3, "any", None) == validate_state_parameters(
            matrix, n_states=3, kind="any")
        assert (3, "any", None) == validate_state_parameters(
            matrix, n_states=3)
        assert (2, "spin_flip", None) == validate_state_parameters(
            matrix, n_spin_flip=2)

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_states=-2),     # Negative number of states requested
            dict(n_spin_flip=-3),  # Negative number of states requested
            dict(n_states=2, n_singlets=2),    # States of two sorts
            dict(n_triplets=2, n_singlets=2),  # States of two sorts
            dict(n_states=2, n_spin_flip=2),   # States of two sorts
            dict(n_spin_flip=2, kind="singlet"),  # kind and n_ do not agree
            dict(n_states=2, kind="bla"),      # Kind invalid
            dict(n_states=4, kind="singlet"),  # UHF with singlets
            dict(n_states=2, kind="triplet"),  # UHF with triplets
            dict(n_triplets=2),    # UHF with triplets
            dict(n_singlets=6),    # UHF with singlets
            dict(n_doublets=3),    # UHF with doublets (only restricted IP/EA)
            dict(n_states=2, is_alpha=True),    # Parameter only for IP/EA-ADC
            dict(n_states=2, is_alpha=False),   # Parameter only for IP/EA-ADC
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_rhf_ip(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("h2o_sto3g", case="gen")
        matrix.method = AdcMethod("ip-adc2") 

        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=False))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=True))  # restricted always beta
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_states=2, kind="doublet"))
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=2))
        assert (6, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=6, is_alpha=True))

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_doublets=-2),   # Negative number of states requested
            dict(n_states=2, kind="bla"),  # Kind invalid
            dict(n_singlets=2),    # Kind invalid for IP/EA-ADC
            dict(n_triplets=2),    # Kind invalid for IP/EA-ADC
            dict(n_spin_flip=2),   # Kind invalid for IP/EA-ADC
            dict(n_states=2, is_alpha="yes"),    # is_alpha not boolean
            dict(n_states=2, is_alpha=1),        # is_alpha not boolean
            dict(n_states=2, n_spin_flip=2),     # States of two sorts
            dict(n_doublets=2, kind="singlet"),  # kind and n_ do not agree
        ]

        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_uhf_ip(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("cn_sto3g", case="gen")
        matrix.method = AdcMethod("ip-adc2") 

        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, kind="any")
        assert (3, "any", False) == validate_state_parameters(
            matrix, n_states=3, is_alpha=False)
        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, is_alpha=True)

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_states=-2),     # Negative number of states requested
            dict(n_states=2, kind="bla"),  # Kind invalid
            dict(n_doublets=2),  # UHF with doublets
            dict(n_states=2, kind="doublet"),  # UHF with doublets
            dict(n_singlets=2),    # Kind invalid for IP/EA-ADC and UHF
            dict(n_triplets=2),    # Kind invalid for IP/EA-ADC and UHF
            dict(n_spin_flip=2),   # Kind invalid for IP/EA-ADC and UHF
            dict(n_states=2, is_alpha="yes"),    # is_alpha not boolean
            dict(n_states=2, is_alpha=1),        # is_alpha not boolean
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)
    
    def test_validate_state_parameters_rhf_ea(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("h2o_sto3g", case="gen")

        # IP
        matrix.method = AdcMethod("ea-adc2") 

        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=False))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=True))  # restricted always beta
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_states=2, kind="doublet"))
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=2))
        assert (6, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=6, is_alpha=True))

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_doublets=-2),   # Negative number of states requested
            dict(n_states=2, kind="bla"),  # Kind invalid
            dict(n_singlets=2),    # Kind invalid for IP/EA-ADC
            dict(n_triplets=2),    # Kind invalid for IP/EA-ADC
            dict(n_spin_flip=2),   # Kind invalid for IP/EA-ADC
            dict(n_states=2, is_alpha="yes"),    # is_alpha not boolean
            dict(n_states=2, is_alpha=1),        # is_alpha not boolean
            dict(n_states=2, n_spin_flip=2),     # States of two sorts
            dict(n_doublets=2, kind="singlet"),  # kind and n_ do not agree
        ]

        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_uhf_ea(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike
        from adcc.AdcMethod import AdcMethod

        # Build empty AdcMatrixlike object and assign ref_state and method
        matrix = AdcMatrixlike()
        matrix.reference_state = testdata_cache.refstate("cn_sto3g", case="gen")
        matrix.method = AdcMethod("ea-adc2") 

        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, kind="any")
        assert (3, "any", False) == validate_state_parameters(
            matrix, n_states=3, is_alpha=False)
        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, is_alpha=True)

        invalid_cases = [
            dict(),                # No states requested
            dict(n_states=0),      # No states requested
            dict(n_states=-2),     # Negative number of states requested
            dict(n_states=2, kind="bla"),  # Kind invalid
            dict(n_doublets=2),  # UHF with doublets
            dict(n_states=2, kind="doublet"),  # UHF with doublets
            dict(n_singlets=2),    # Kind invalid for IP/EA-ADC and UHF
            dict(n_triplets=2),    # Kind invalid for IP/EA-ADC and UHF
            dict(n_spin_flip=2),   # Kind invalid for IP/EA-ADC and UHF
            dict(n_states=2, is_alpha="yes"),    # is_alpha not boolean
            dict(n_states=2, is_alpha=1),        # is_alpha not boolean
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_construct_adcmatrix(self):
        from adcc.workflow import construct_adcmatrix

        #
        # Construction from hfdata
        #
        hfdata = testdata_cache._load_hfdata("h2o_sto3g")

        res = construct_adcmatrix(hfdata, method="adc3")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("adc3")
        assert res.method.adc_type == "pp"
        assert res.axis_blocks == ["ph", "pphh"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []

        res = construct_adcmatrix(hfdata, method="cvs-adc3", core_orbitals=1)
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("cvs-adc3")
        assert res.method.adc_type == "pp"
        assert res.axis_blocks == ["ph", "pphh"]
        assert res.mospaces.core_orbitals == [0, 7]
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []

        res = construct_adcmatrix(hfdata, method="adc2", frozen_core=1)
        assert res.method.adc_type == "pp"
        assert res.axis_blocks == ["ph", "pphh"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == [0, 7]
        assert res.mospaces.frozen_virtual == []

        res = construct_adcmatrix(hfdata, method="adc2", frozen_virtual=1)
        assert res.method.adc_type == "pp"
        assert res.axis_blocks == ["ph", "pphh"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == [6, 13]

        res = construct_adcmatrix(hfdata, method="adc2", frozen_virtual=1,
                                  frozen_core=1)
        assert res.method.adc_type == "pp"
        assert res.axis_blocks == ["ph", "pphh"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == [0, 7]
        assert res.mospaces.frozen_virtual == [6, 13]

        res = construct_adcmatrix(hfdata, method="ip-adc3")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("ip-adc3")
        assert res.method.adc_type == "ip"
        assert res.axis_blocks == ["h", "phh"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []

        res = construct_adcmatrix(hfdata, method="ea-adc2")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("ea-adc2")
        assert res.method.adc_type == "ea"
        assert res.axis_blocks == ["p", "pph"]
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []

        invalid_cases = [
            dict(),                   # Missing method
            dict(method="dadadad"),   # Unknown method
            dict(frozen_core=1),      # Missing method
            dict(frozen_virtual=3),   # Missing method
            dict(core_orbitals=4),    # Missing method
            dict(method="cvs-adc2"),  # No core_orbitals
            dict(method="cvs-adc2", frozen_core=1),
            dict(method="adc2", core_orbitals=3),  # Extra core parameter
            dict(method="adc2", core_orbitals=3, frozen_virtual=2),
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                construct_adcmatrix(hfdata, **case)

        #
        # Construction from LazyMp or ReferenceState
        #
        refst_ful = testdata_cache.refstate("h2o_sto3g", case="gen")
        refst_cvs = testdata_cache.refstate("h2o_sto3g", case="cvs")
        gs_ful, gs_cvs = adcc.LazyMp(refst_ful), adcc.LazyMp(refst_cvs)

        for obj in [gs_ful, refst_ful]:
            res = construct_adcmatrix(obj, method="adc2")
            assert isinstance(res, adcc.AdcMatrix)
            assert res.method == adcc.AdcMethod("adc2")

            with pytest.raises(InputError,
                               match=r"Cannot run a core-valence"):
                construct_adcmatrix(obj, method="cvs-adc2x")
            with pytest.warns(UserWarning,
                              match=r"^Ignored frozen_core parameter"):
                construct_adcmatrix(obj, frozen_core=3, method="adc1")
            with pytest.warns(UserWarning,
                              match=r"^Ignored frozen_virtual parameter"):
                construct_adcmatrix(obj, frozen_virtual=1, method="adc3")

        for obj in [gs_cvs, refst_cvs]:
            res = construct_adcmatrix(obj, method="cvs-adc2x")
            assert isinstance(res, adcc.AdcMatrix)
            assert res.method == adcc.AdcMethod("cvs-adc2x")

            with pytest.raises(InputError):
                construct_adcmatrix(obj)  # Missing method
            with pytest.raises(InputError, match=r"Cannot run a general"):
                construct_adcmatrix(obj, method="adc2")
            with pytest.warns(UserWarning,
                              match=r"^Ignored core_orbitals parameter"):
                construct_adcmatrix(obj, core_orbitals=2, method="cvs-adc1")

        #
        # Construction from ADC matrix
        #
        mtx_ful = adcc.AdcMatrix("adc2", gs_ful)
        mtx_cvs = adcc.AdcMatrix("cvs-adc2", gs_cvs)
        assert construct_adcmatrix(mtx_ful) == mtx_ful
        assert construct_adcmatrix(mtx_ful, method="adc2") == mtx_ful
        assert construct_adcmatrix(mtx_cvs) == mtx_cvs
        assert construct_adcmatrix(mtx_cvs, method="cvs-adc2") == mtx_cvs

        with pytest.warns(UserWarning, match=r"Ignored method parameter"):
            construct_adcmatrix(mtx_ful, method="adc3")
        with pytest.warns(UserWarning, match=r"^Ignored core_orbitals parameter"):
            construct_adcmatrix(mtx_cvs, core_orbitals=2)
        with pytest.warns(UserWarning, match=r"^Ignored frozen_core parameter"):
            construct_adcmatrix(mtx_ful, frozen_core=3)
        with pytest.warns(UserWarning,
                          match=r"^Ignored frozen_virtual parameter"):
            construct_adcmatrix(mtx_cvs, frozen_virtual=1)

    def test_diagonalise_adcmatrix_pp(self):
        from adcc.workflow import diagonalise_adcmatrix

        system = "h2o_sto3g"
        case = "gen"
        method = "adc2"
        kind = "singlet"

        refdata = testdata_cache.adcman_data(system, method=method, case=case)
        ref_singlets = refdata[kind]["eigenvalues"]
        n_states = min(len(ref_singlets), 3)

        matrix = adcc.AdcMatrix(method, testdata_cache.refstate(system, case=case))

        guesses = adcc.guesses_singlet(matrix, n_guesses=6, block="ph")
        res = diagonalise_adcmatrix(matrix, n_states=n_states, kind=kind,
                                    guesses=guesses)
        assert res.converged
        assert res.eigenvalues[:n_states] == approx(ref_singlets[:n_states])

        with pytest.raises(InputError):  # Too low tolerance
            # SCF tolerance = 1e-14 currently
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses,eigensolver="davidson",
                                        conv_tol=1e-15)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses, eigensolver="blubber")

        with pytest.raises(ValueError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        eigensolver="davidson",
                                        guesses=guesses)

    def test_diagonalise_adcmatrix_ip(self):
        from adcc.workflow import diagonalise_adcmatrix
        # pytest.skip("adcman referencedata not yet available")
        system = "h2o_sto3g"
        case = "gen"
        method = "ip-adc2"
        kind = "doublet"

        refdata = testdata_cache.adcman_data(system, method=method, case=case)
        ref_doublets = refdata[kind]["eigenvalues"]
        n_states = min(len(ref_doublets), 3)

        matrix = adcc.AdcMatrix(method, testdata_cache.refstate(system, case=case))

        guesses = adcc.guesses_doublet(matrix, n_guesses=6, block="h",
                                       is_alpha=True)
        res = diagonalise_adcmatrix(matrix, n_states=n_states, kind=kind,
                                    guesses=guesses, is_alpha=True)
        assert res.converged
        assert res.eigenvalues[:n_states] == approx(ref_doublets[:n_states])

        with pytest.raises(InputError):  # Too low tolerance
            # SCF tolerance = 1e-14 currently
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses,eigensolver="davidson",
                                        conv_tol=1e-15)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses, eigensolver="blubber")

        with pytest.raises(ValueError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        eigensolver="davidson",
                                        guesses=guesses)

    def test_diagonalise_adcmatrix_ea(self):
        from adcc.workflow import diagonalise_adcmatrix
        system = "h2o_sto3g"
        case = "gen"
        method = "ea-adc2"
        kind = "doublet"

        refdata = testdata_cache.adcman_data(system, method=method, case=case)
        ref_doublets = refdata[kind]["eigenvalues"]
        n_states = min(len(ref_doublets), 3)

        matrix = adcc.AdcMatrix(method, testdata_cache.refstate(system, case=case))

        guesses = adcc.guesses_doublet(matrix, n_guesses=6, block="p",
                                       is_alpha=True)
        res = diagonalise_adcmatrix(matrix, n_states=n_states, kind=kind,
                                    guesses=guesses, is_alpha=True)
        assert res.converged
        assert res.eigenvalues[:n_states] == approx(ref_doublets[:n_states])

        with pytest.raises(InputError):  # Too low tolerance
            # SCF tolerance = 1e-14 currently
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses,eigensolver="davidson",
                                        conv_tol=1e-15)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        guesses=guesses, eigensolver="blubber")

        with pytest.raises(ValueError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=9, kind=kind,
                                        eigensolver="davidson",
                                        guesses=guesses)

    def test_obtain_guesses_by_inspection_pp(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("adc2", ground_state)
        matrix1 = adcc.AdcMatrix("adc1", ground_state)

        # Test that the right number of guesses is returned ...
        for mat in [matrix1, matrix2]:
            for i in range(4, 9):
                res = obtain_guesses_by_inspection(mat, n_guesses=i,
                                                   kind="singlet")
                assert len(res) == i

        for i in range(4, 9):
            res = obtain_guesses_by_inspection(
                matrix2, n_guesses=i, kind="triplet", n_guesses_doubles=2)
            assert len(res) == i

        # Test right number of guesses if insufficient singles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=20, kind="singlet")
        assert len(res) == 20

        # Only doubles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=4, 
                                           kind="singlet",
                                           n_guesses_doubles=4)
        assert len(res) == 4

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=4, kind="any",
                                         n_guesses_doubles=2)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=40, kind="any")

    def test_obtain_guesses_by_inspection_ip(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("ip-adc2", ground_state)
        matrix1 = adcc.AdcMatrix("ip-adc1", ground_state)

        # Test that the right number of guesses is returned
        for i in range(4, 9):
            res = obtain_guesses_by_inspection(matrix2, n_guesses=i,
                                               kind="doublet",
                                               spin_change=-0.5, is_alpha=True)
            assert len(res) == i

        for i in range(2, 5):
            res = obtain_guesses_by_inspection(
                matrix1, n_guesses=i, kind="doublet",
                spin_change=-0.5, is_alpha=True)
            assert len(res) == i

        # Test right number of guesses if insufficient singles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=20,
                                           kind="doublet",
                                           spin_change=-0.5, is_alpha=True)
        assert len(res) == 20

        # Only doubles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=4, 
                                           kind="doublet",
                                           spin_change=-0.5, is_alpha=True,
                                           n_guesses_doubles=4)
        assert len(res) == 4

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=6, kind="any",
                                         spin_change=-0.5, is_alpha=True)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=2, kind="any",
                                         n_guesses_doubles=2, spin_change=-0.5,
                                         is_alpha=True)

    def test_obtain_guesses_by_inspection_ea(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("ea-adc2", ground_state)
        matrix1 = adcc.AdcMatrix("ea-adc1", ground_state)

        # Test that the right number of guesses is returned
        for i in range(4, 9):
            res = obtain_guesses_by_inspection(matrix2, n_guesses=i,
                                               kind="doublet",
                                               spin_change=0.5,
                                               is_alpha=True)
            assert len(res) == i

        for i in range(1, 2):
            res = obtain_guesses_by_inspection(
                matrix1, n_guesses=i, kind="doublet",
                spin_change=0.5, is_alpha=True)
            assert len(res) == i

        # Test right number of guesses if insufficient singles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=20,
                                           kind="doublet",
                                           spin_change=0.5, is_alpha=True)
        assert len(res) == 20

        # Only doubles guesses
        res = obtain_guesses_by_inspection(matrix2, n_guesses=4, 
                                           kind="doublet",
                                           spin_change=0.5, is_alpha=True,
                                           n_guesses_doubles=4)
        assert len(res) == 4

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=6, kind="any",
                                         spin_change=0.5, is_alpha=True)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=2, kind="any",
                                         n_guesses_doubles=2, spin_change=0.5,
                                         is_alpha=True)

    def test_construct_guesses_explicit(self):
        from adcc.workflow import construct_guesses

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("adc2", ground_state)

        res = construct_guesses(
            matrix=matrix,
            n_states=3,
            kind="singlet",
            spin_change=0,
            n_guesses=5
        )

        assert len(res) == 5

    def test_construct_guesses_davidson(self):
        from adcc.workflow import construct_guesses

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("adc2", ground_state)

        res = construct_guesses(
            matrix=matrix,
            n_states=2,
            kind="singlet",
            spin_change=0,
            n_guesses=None,
            eigensolver="davidson"
        )

        assert len(res) == 4

    def test_construct_guesses_lanczos(self):
        from adcc.workflow import construct_guesses

        refstate = testdata_cache.refstate("h2o_sto3g", case="gen")
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("adc2", ground_state)

        res = construct_guesses(
            matrix=matrix,
            n_states=2,
            kind="singlet",
            spin_change=0,
            n_guesses=None,
            eigensolver="lanczos"
        )

        assert len(res) == 2