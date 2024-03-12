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
from adcc.testdata.cache import cache


class TestWorkflow:
    def test_validate_state_parameters_rhf_pp(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike

        # Build empty AdcMatrixlike object and assign ref_state and type
        matrix = AdcMatrixlike()
        matrix.reference_state = cache.refstate["h2o_sto3g"]
        matrix.type = "pp"

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
            dict(n_states=2, kind="bla"),      # Kind invalid
            dict(n_states=2, kind="doublet"),  # Kind invalid for PP-ADC
            dict(n_states=2, is_alpha=True),   # Parameter only for IP/EA-ADC

        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_rhf_ip_ea(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike

        # Build empty AdcMatrixlike object and assign ref_state and type
        matrix = AdcMatrixlike()
        matrix.reference_state = cache.refstate["h2o_sto3g"]
        matrix.type = "ip"  # Does not matter if "ip" or "ea"

        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=True))
        assert (3, "any", True) == (validate_state_parameters(
            matrix, n_states=3, is_alpha=False))  # restricted always alpha
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_states=2, kind="doublet"))
        assert (2, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=2))
        assert (6, "doublet", True) == (validate_state_parameters(
            matrix, n_doublets=6, is_alpha=False))

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
            print(case)
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_uhf_pp(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike

        # Build empty AdcMatrixlike object and assign ref_state and type
        matrix = AdcMatrixlike()
        matrix.reference_state = cache.refstate["cn_sto3g"]
        matrix.type = "pp"

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
            dict(n_states=2, kind="bla"),      # Kind invaled
            dict(n_states=4, kind="singlet"),  # UHF with singlets
            dict(n_states=2, kind="triplet"),  # UHF with triplets
            dict(n_triplets=2),    # UHF with triplets
            dict(n_singlets=6),    # UHF with singlets
        ]
        for case in invalid_cases:
            with pytest.raises(InputError):
                validate_state_parameters(matrix, **case)

    def test_validate_state_parameters_uhf_ip_ea(self):
        from adcc.workflow import validate_state_parameters
        from adcc.AdcMatrix import AdcMatrixlike

        # Build empty AdcMatrixlike object and assign ref_state and type
        matrix = AdcMatrixlike()
        matrix.reference_state = cache.refstate["cn_sto3g"]
        matrix.type = "ip"  # Does not matter if "ip" or "ea"

        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, kind="any")
        assert (3, "any", True) == validate_state_parameters(
            matrix, n_states=3, is_alpha=True)
        assert (3, "any", False) == validate_state_parameters(
            matrix, n_states=3, is_alpha=False)

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
        hfdata = cache.hfdata["h2o_sto3g"]

        res = construct_adcmatrix(hfdata, method="adc3")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("adc3")
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []
        assert res.type == "pp"
        assert ["ph", "pphh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="cvs-adc3", core_orbitals=1)
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("cvs-adc3")
        assert res.mospaces.core_orbitals == [0, 7]
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []
        assert res.type == "pp"
        assert ["ph", "pphh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="adc2", frozen_core=1)
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == [0, 7]
        assert res.mospaces.frozen_virtual == []
        assert res.type == "pp"
        assert ["ph", "pphh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="adc2", frozen_virtual=1)
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == [6, 13]
        assert res.type == "pp"
        assert ["ph", "pphh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="adc2", frozen_virtual=1,
                                  frozen_core=1)
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == [0, 7]
        assert res.mospaces.frozen_virtual == [6, 13]
        assert res.type == "pp"
        assert ["ph", "pphh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="ip_adc3")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("ip_adc3")
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []
        assert res.type == "ip"
        assert ["h", "phh"] == res.axis_blocks

        res = construct_adcmatrix(hfdata, method="ea_adc2")
        assert isinstance(res, adcc.AdcMatrix)
        assert res.method == adcc.AdcMethod("ea_adc2")
        assert res.mospaces.core_orbitals == []
        assert res.mospaces.frozen_core == []
        assert res.mospaces.frozen_virtual == []
        assert res.type == "ea"
        assert ["p", "pph"] == res.axis_blocks

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
        refst_ful = cache.refstate["h2o_sto3g"]
        refst_cvs = cache.refstate_cvs["h2o_sto3g"]
        gs_ful, gs_cvs = adcc.LazyMp(refst_ful), adcc.LazyMp(refst_cvs)

        for obj in [gs_ful, refst_ful]:
            res = construct_adcmatrix(obj, method="adc2")
            assert isinstance(res, adcc.AdcMatrix)
            assert res.method == adcc.AdcMethod("adc2")
            assert res.type == "pp"

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

        refdata = cache.reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("adc2", adcc.LazyMp(cache.refstate["h2o_sto3g"]))

        res = diagonalise_adcmatrix(matrix, n_states=3, kind="singlet",
                                    eigensolver="davidson", spin_change=0)
        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets[:3])

        guesses = adcc.guesses_singlet(matrix, n_guesses=6, block="ph",
                                       spin_change=0)
        res = diagonalise_adcmatrix(matrix, n_states=3, kind="singlet",
                                    guesses=guesses, spin_change=0)
        ref_singlets = refdata["adc2"]["singlet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_singlets[:3])

        with pytest.raises(InputError):  # Too low tolerance
            res = diagonalise_adcmatrix(matrix, n_states=9, kind="singlet",
                                        eigensolver="davidson",
                                        conv_tol=1e-14)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=9, kind="singlet",
                                        eigensolver="blubber")

        with pytest.raises(InputError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=9, kind="singlet",
                                        eigensolver="davidson",
                                        guesses=guesses)

    def test_diagonalise_adcmatrix_ip(self):
        from adcc.workflow import diagonalise_adcmatrix

        refdata = cache.adcc_reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("ip_adc2",
                                adcc.LazyMp(cache.refstate["h2o_sto3g"]))

        res = diagonalise_adcmatrix(matrix, n_states=3, kind="doublet",
                                    eigensolver="davidson", spin_change=-0.5)
        ref_doublets = refdata["ip_adc2"]["doublet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_doublets[:3])

        guesses = adcc.guesses_doublet(matrix, n_guesses=5, block="h",
                                       spin_change=-0.5)
        res = diagonalise_adcmatrix(matrix, n_states=3, kind="doublet",
                                    guesses=guesses, spin_change=-0.5)
        assert res.converged
        assert res.eigenvalues == approx(ref_doublets[:3])

        with pytest.raises(InputError):  # Too low tolerance
            res = diagonalise_adcmatrix(matrix, n_states=5, kind="doublet",
                                        eigensolver="davidson",
                                        conv_tol=1e-14, spin_change=-0.5)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=5, kind="doublet",
                                        eigensolver="blubber", spin_change=-0.5)

        with pytest.raises(InputError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=6, kind="doublet",
                                        eigensolver="davidson",
                                        guesses=guesses, spin_change=-0.5)

    def test_diagonalise_adcmatrix_ea(self):
        from adcc.workflow import diagonalise_adcmatrix

        refdata = cache.adcc_reference_data["h2o_sto3g"]
        matrix = adcc.AdcMatrix("ea_adc2",
                                adcc.LazyMp(cache.refstate["h2o_sto3g"]))

        res = diagonalise_adcmatrix(matrix, n_states=2, kind="doublet",
                                    eigensolver="davidson", spin_change=0.5)
        ref_doublets = refdata["ea_adc2"]["doublet"]["eigenvalues"]
        assert res.converged
        assert res.eigenvalues == approx(ref_doublets[:2])

        guesses = adcc.guesses_doublet(matrix, n_guesses=2, block="p",
                                       spin_change=0.5)
        guesses += adcc.guesses_doublet(matrix, n_guesses=2, block="pph",
                                        spin_change=0.5)
        res = diagonalise_adcmatrix(matrix, n_states=2, kind="doublet",
                                    guesses=guesses, spin_change=0.5)
        assert res.converged
        assert res.eigenvalues == approx(ref_doublets[:2])

        with pytest.raises(InputError):  # Too low tolerance
            res = diagonalise_adcmatrix(matrix, n_states=5, kind="doublet",
                                        eigensolver="davidson",
                                        conv_tol=1e-14, spin_change=0.5)

        with pytest.raises(InputError):  # Wrong solver method
            res = diagonalise_adcmatrix(matrix, n_states=5, kind="doublet",
                                        eigensolver="blubber", spin_change=0.5)

        with pytest.raises(InputError):  # Too few guesses
            res = diagonalise_adcmatrix(matrix, n_states=6, kind="doublet",
                                        eigensolver="davidson",
                                        guesses=guesses, spin_change=0.5)

    def test_estimate_n_guesses_pp(self):
        from adcc.workflow import estimate_n_guesses

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("adc2", ground_state)

        # Check minimal number of guesses is 4 and at some point
        # we get more than four guesses
        assert 4 == estimate_n_guesses(matrix, n_states=1, singles_only=True)
        assert 4 == estimate_n_guesses(matrix, n_states=2, singles_only=True)
        for i in range(3, 20):
            assert i <= estimate_n_guesses(matrix, n_states=i,
                                           singles_only=True)

    def test_estimate_n_guesses_ip(self):
        from adcc.workflow import estimate_n_guesses

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("ip_adc2", ground_state)

        # Check minimal number of guesses is 4 and at some point
        # we get more than four guesses
        assert 4 == estimate_n_guesses(matrix, n_states=1, singles_only=True)
        assert 4 == estimate_n_guesses(matrix, n_states=2, singles_only=True)
        for i in range(3, 20):
            assert i <= estimate_n_guesses(matrix, n_states=i,
                                           singles_only=True)

        # Test different behaviour for IP-ADC(0/1)
        matrix = adcc.AdcMatrix("ip_adc0", ground_state)
        assert 4 == estimate_n_guesses(matrix, n_states=2, singles_only=True)
        assert 5 == estimate_n_guesses(matrix, n_states=5, singles_only=True)
        assert 10 == estimate_n_guesses(matrix, n_states=10, singles_only=True)

    def test_estimate_n_guesses_ea(self):
        from adcc.workflow import estimate_n_guesses

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix = adcc.AdcMatrix("ea_adc2", ground_state)

        # Check minimal number of guesses is 4 and at some point
        # we get more than four guesses
        assert 4 == estimate_n_guesses(matrix, n_states=1, singles_only=True)
        assert 4 == estimate_n_guesses(matrix, n_states=2, singles_only=True)
        for i in range(3, 20):
            assert i <= estimate_n_guesses(matrix, n_states=i,
                                           singles_only=True)

        # Test different behaviour for EA-ADC(0/1)
        matrix = adcc.AdcMatrix("ea_adc1", ground_state)
        assert 2 == estimate_n_guesses(matrix, n_states=1, singles_only=True)
        assert 2 == estimate_n_guesses(matrix, n_states=2, singles_only=True)
        assert 3 == estimate_n_guesses(matrix, n_states=3, singles_only=True)

    def test_obtain_guesses_by_inspection_pp(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("adc2", ground_state)
        matrix1 = adcc.AdcMatrix("adc1", ground_state)

        # Test that the right number of guesses is returned ...
        for mat in [matrix1, matrix2]:
            for i in range(4, 9):
                res = obtain_guesses_by_inspection(mat, n_guesses=i,
                                                   kind="singlet",
                                                   spin_change=0)
                assert len(res) == i

        for i in range(4, 9):
            res = obtain_guesses_by_inspection(
                matrix2, n_guesses=i, kind="triplet", n_guesses_doubles=2,
                spin_change=0)
            assert len(res) == i

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=4, kind="any",
                                         n_guesses_doubles=2, spin_change=0)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=40, kind="any",
                                         spin_change=0)

    def test_obtain_guesses_by_inspection_ip(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("ip_adc2", ground_state)
        matrix1 = adcc.AdcMatrix("ip_adc1", ground_state)

        # Test that the right number of guesses is returned
        for i in range(4, 9):
            res = obtain_guesses_by_inspection(matrix2, n_guesses=i,
                                               kind="doublet",
                                               spin_change=-0.5)
            assert len(res) == i

        for i in range(2, 5):
            res = obtain_guesses_by_inspection(
                matrix1, n_guesses=i, kind="doublet",
                spin_change=-0.5)
            assert len(res) == i

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=6, kind="any",
                                         spin_change=-0.5)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=2, kind="any",
                                         n_guesses_doubles=2, spin_change=-0.5)

    def test_obtain_guesses_by_inspection_ep(self):
        from adcc.workflow import obtain_guesses_by_inspection

        refstate = cache.refstate["h2o_sto3g"]
        ground_state = adcc.LazyMp(refstate)
        matrix2 = adcc.AdcMatrix("ea_adc2", ground_state)
        matrix1 = adcc.AdcMatrix("ea_adc1", ground_state)

        # Test that the right number of guesses is returned
        for i in range(4, 9):
            res = obtain_guesses_by_inspection(matrix2, n_guesses=i,
                                               kind="doublet",
                                               spin_change=0.5)
            assert len(res) == i

        for i in range(1, 2):
            res = obtain_guesses_by_inspection(
                matrix1, n_guesses=i, kind="doublet",
                spin_change=0.5)
            assert len(res) == i

        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=3, kind="any",
                                         spin_change=0.5)
        with pytest.raises(InputError):
            obtain_guesses_by_inspection(matrix1, n_guesses=2, kind="any",
                                         n_guesses_doubles=2, spin_change=0.5)
