#!/bin/bash

SOURCE="https://get.michael-herbst.com/adcc/testdata/0.3.0/"
DATAFILES=(
	cn_sto3g_hfdata.hdf5
	cn_sto3g_hfimport.hdf5
	cn_sto3g_reference_adc0.hdf5
	cn_sto3g_reference_adc1.hdf5
	cn_sto3g_reference_adc2.hdf5
	cn_sto3g_reference_adc2x.hdf5
	cn_sto3g_reference_adc3.hdf5
	cn_sto3g_reference_cvs_adc0.hdf5
	cn_sto3g_reference_cvs_adc1.hdf5
	cn_sto3g_reference_cvs_adc2.hdf5
	cn_sto3g_reference_cvs_adc2x.hdf5
	cn_sto3g_reference_cvs_adc3.hdf5
	cn_sto3g_reference_fc_adc2.hdf5
	cn_sto3g_reference_fc_fv_adc2.hdf5
	cn_sto3g_reference_fv_adc2x.hdf5
	cn_sto3g_reference_fv_cvs_adc2x.hdf5
	h2o_sto3g_hfdata.hdf5
	h2o_sto3g_hfimport.hdf5
	h2o_sto3g_reference_adc0.hdf5
	h2o_sto3g_reference_adc1.hdf5
	h2o_sto3g_reference_adc2.hdf5
	h2o_sto3g_reference_adc2x.hdf5
	h2o_sto3g_reference_adc3.hdf5
	h2o_sto3g_reference_cvs_adc0.hdf5
	h2o_sto3g_reference_cvs_adc1.hdf5
	h2o_sto3g_reference_cvs_adc2.hdf5
	h2o_sto3g_reference_cvs_adc2x.hdf5
	h2o_sto3g_reference_cvs_adc3.hdf5
	h2o_sto3g_reference_fc_adc2.hdf5
	h2o_sto3g_reference_fc_fv_adc2.hdf5
	h2o_sto3g_reference_fv_adc2x.hdf5
	h2o_sto3g_reference_fv_cvs_adc2x.hdf5
	h2s_sto3g_hfdata.hdf5
	h2s_sto3g_reference_fc_cvs_adc2.hdf5
	h2s_sto3g_reference_fc_fv_cvs_adc2x.hdf5
	hf3_631g_hfdata.hdf5
	hf3_631g_reference_adc0.hdf5
	hf3_631g_reference_adc1.hdf5
	hf3_631g_reference_adc2.hdf5
	hf3_631g_reference_adc2x.hdf5
	hf3_631g_reference_adc3.hdf5
)
DATAFILES_FULL=(
	cn_ccpvdz_hfdata.hdf5
	cn_ccpvdz_hfimport.hdf5
	cn_ccpvdz_reference_adc0.hdf5
	cn_ccpvdz_reference_adc1.hdf5
	cn_ccpvdz_reference_adc2.hdf5
	cn_ccpvdz_reference_adc2x.hdf5
	cn_ccpvdz_reference_adc3.hdf5
	cn_ccpvdz_reference_cvs_adc0.hdf5
	cn_ccpvdz_reference_cvs_adc1.hdf5
	cn_ccpvdz_reference_cvs_adc2.hdf5
	cn_ccpvdz_reference_cvs_adc2x.hdf5
	cn_ccpvdz_reference_cvs_adc3.hdf5
	h2o_def2tzvp_hfdata.hdf5
	h2o_def2tzvp_hfimport.hdf5
	h2o_def2tzvp_reference_adc0.hdf5
	h2o_def2tzvp_reference_adc1.hdf5
	h2o_def2tzvp_reference_adc2.hdf5
	h2o_def2tzvp_reference_adc2x.hdf5
	h2o_def2tzvp_reference_adc3.hdf5
	h2o_def2tzvp_reference_cvs_adc0.hdf5
	h2o_def2tzvp_reference_cvs_adc1.hdf5
	h2o_def2tzvp_reference_cvs_adc2.hdf5
	h2o_def2tzvp_reference_cvs_adc2x.hdf5
	h2o_def2tzvp_reference_cvs_adc3.hdf5
	h2s_6311g_hfdata.hdf5
	h2s_6311g_reference_adc2.hdf5
	h2s_6311g_reference_cvs_adc2x.hdf5
	h2s_6311g_reference_fc_adc2.hdf5
	h2s_6311g_reference_fc_cvs_adc2x.hdf5
	h2s_6311g_reference_fc_fv_adc2.hdf5
	h2s_6311g_reference_fc_fv_cvs_adc2x.hdf5
	h2s_6311g_reference_fv_adc2.hdf5
	h2s_6311g_reference_fv_cvs_adc2x.hdf5
)

if [ "$1" == "--full" ]; then
	DATAFILES=("${DATAFILES[@]}" "${DATAFILES_FULL[@]}")
fi

#
# -----
#

THISDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$THISDIR"
echo "Updating testdata ... please wait."

download() {
	if which wget &> /dev/null; then
		wget -w 1 -qN --show-progress $@
	else
		echo "wget not installed" >&2
		exit 1
	fi
}

download $(for file in ${DATAFILES[@]}; do echo $SOURCE/$file; done)

if which sha256sum &> /dev/null; then
	sha256sum --ignore-missing -c SHA256SUMS || exit 1
fi

exit 0
