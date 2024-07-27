#!/bin/bash

SOURCE="https://q-chem.de/adcc_testdata/0.5.0/"
DATAFILES=(
	ch2nh2_sto3g_hfdata.hdf5
	ch2nh2_sto3g_hfimport.hdf5
	cn_sto3g_hfdata.hdf5
	cn_sto3g_hfimport.hdf5
	cn_sto3g_adcc_reference_adc0.hdf5
	cn_sto3g_adcc_reference_adc1.hdf5
	cn_sto3g_adcc_reference_adc2.hdf5
	cn_sto3g_adcc_reference_adc2x.hdf5
	cn_sto3g_adcc_reference_adc3.hdf5
	cn_sto3g_adcc_reference_cvs_adc0.hdf5
	cn_sto3g_adcc_reference_cvs_adc1.hdf5
	cn_sto3g_adcc_reference_cvs_adc2.hdf5
	cn_sto3g_adcc_reference_cvs_adc2x.hdf5
	cn_sto3g_adcc_reference_cvs_adc3.hdf5
	cn_sto3g_adcc_reference_fc_adc2.hdf5
	cn_sto3g_adcc_reference_fc_fv_adc2.hdf5
	cn_sto3g_adcc_reference_fv_adc2x.hdf5
	cn_sto3g_adcc_reference_fv_cvs_adc2x.hdf5
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
	h2o_sto3g_adcc_reference_adc0.hdf5
	h2o_sto3g_adcc_reference_adc1.hdf5
	h2o_sto3g_adcc_reference_adc2.hdf5
	h2o_sto3g_adcc_reference_adc2x.hdf5
	h2o_sto3g_adcc_reference_adc3.hdf5
	h2o_sto3g_adcc_reference_cvs_adc0.hdf5
	h2o_sto3g_adcc_reference_cvs_adc1.hdf5
	h2o_sto3g_adcc_reference_cvs_adc2.hdf5
	h2o_sto3g_adcc_reference_cvs_adc2x.hdf5
	h2o_sto3g_adcc_reference_cvs_adc3.hdf5
	h2o_sto3g_adcc_reference_fc_adc2.hdf5
	h2o_sto3g_adcc_reference_fc_fv_adc2.hdf5
	h2o_sto3g_adcc_reference_fv_adc2x.hdf5
	h2o_sto3g_adcc_reference_fv_cvs_adc2x.hdf5
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
	h2s_sto3g_adcc_reference_fc_cvs_adc2.hdf5
	h2s_sto3g_adcc_reference_fc_fv_cvs_adc2x.hdf5
	h2s_sto3g_reference_fc_cvs_adc2.hdf5
	h2s_sto3g_reference_fc_fv_cvs_adc2x.hdf5
	hf3_631g_hfdata.hdf5
	hf3_631g_adcc_reference_adc0.hdf5
	hf3_631g_adcc_reference_adc1.hdf5
	hf3_631g_adcc_reference_adc2.hdf5
	hf3_631g_adcc_reference_adc2x.hdf5
	hf3_631g_adcc_reference_adc3.hdf5
	hf3_631g_reference_adc0.hdf5
	hf3_631g_reference_adc1.hdf5
	hf3_631g_reference_adc2.hdf5
	hf3_631g_reference_adc2x.hdf5
	hf3_631g_reference_adc3.hdf5
	methox_sto3g_hfdata.hdf5
	methox_sto3g_adcc_reference_adc0.hdf5
	methox_sto3g_adcc_reference_adc1.hdf5
	methox_sto3g_adcc_reference_adc2.hdf5
	methox_sto3g_adcc_reference_adc2x.hdf5
	methox_sto3g_adcc_reference_adc3.hdf5
	methox_sto3g_adcc_reference_cvs_adc0.hdf5
	methox_sto3g_adcc_reference_cvs_adc1.hdf5
	methox_sto3g_adcc_reference_cvs_adc2.hdf5
	methox_sto3g_adcc_reference_cvs_adc2x.hdf5
	methox_sto3g_adcc_reference_cvs_adc3.hdf5
	methox_sto3g_reference_adc0.hdf5
	methox_sto3g_reference_adc1.hdf5
	methox_sto3g_reference_adc2.hdf5
	methox_sto3g_reference_adc2x.hdf5
	methox_sto3g_reference_adc3.hdf5
	methox_sto3g_reference_cvs_adc0.hdf5
	methox_sto3g_reference_cvs_adc1.hdf5
	methox_sto3g_reference_cvs_adc2.hdf5
	methox_sto3g_reference_cvs_adc2x.hdf5
	methox_sto3g_reference_cvs_adc3.hdf5
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
	cn_ccpvdz_adcc_reference_adc0.hdf5
	cn_ccpvdz_adcc_reference_adc1.hdf5
	cn_ccpvdz_adcc_reference_adc2.hdf5
	cn_ccpvdz_adcc_reference_adc2x.hdf5
	cn_ccpvdz_adcc_reference_adc3.hdf5
	cn_ccpvdz_adcc_reference_cvs_adc0.hdf5
	cn_ccpvdz_adcc_reference_cvs_adc1.hdf5
	cn_ccpvdz_adcc_reference_cvs_adc2.hdf5
	cn_ccpvdz_adcc_reference_cvs_adc2x.hdf5
	cn_ccpvdz_adcc_reference_cvs_adc3.hdf5
	h2o_def2tzvp_hfdata.hdf5
	h2o_def2tzvp_hfimport.hdf5
	h2o_def2tzvp_adcc_reference_adc0.hdf5
	h2o_def2tzvp_adcc_reference_adc1.hdf5
	h2o_def2tzvp_adcc_reference_adc2.hdf5
	h2o_def2tzvp_adcc_reference_adc2x.hdf5
	h2o_def2tzvp_adcc_reference_adc3.hdf5
	h2o_def2tzvp_adcc_reference_cvs_adc0.hdf5
	h2o_def2tzvp_adcc_reference_cvs_adc1.hdf5
	h2o_def2tzvp_adcc_reference_cvs_adc2.hdf5
	h2o_def2tzvp_adcc_reference_cvs_adc2x.hdf5
	h2o_def2tzvp_adcc_reference_cvs_adc3.hdf5
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
	h2s_6311g_adcc_reference_adc2.hdf5
	h2s_6311g_adcc_reference_cvs_adc2x.hdf5
	h2s_6311g_adcc_reference_fc_adc2.hdf5
	h2s_6311g_adcc_reference_fc_cvs_adc2x.hdf5
	h2s_6311g_adcc_reference_fc_fv_adc2.hdf5
	h2s_6311g_adcc_reference_fc_fv_cvs_adc2x.hdf5
	h2s_6311g_adcc_reference_fv_adc2.hdf5
	h2s_6311g_adcc_reference_fv_cvs_adc2x.hdf5
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

# check which files exist
while read line; do
	list=($line)
	fname=${list[1]}
	[ -f "${fname}" ] && echo "$line" >> SHA256SUMS.filtered
done < SHA256SUMS
if which sha256sum &> /dev/null; then
	sha256sum -c SHA256SUMS.filtered || exit 1
fi
rm SHA256SUMS.filtered

exit 0
