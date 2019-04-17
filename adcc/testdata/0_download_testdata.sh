#!/bin/bash

SOURCE="https://get.michael-herbst.com/adcc/testdata/0.2.0/"
DATAFILES=(
	cn_sto3g_hfdata.hdf5
	cn_sto3g_reference_adc0.hdf5
	cn_sto3g_reference_adc1.hdf5
	cn_sto3g_reference_adc2.hdf5
	cn_sto3g_reference_adc2x.hdf5
	cn_sto3g_reference_adc3.hdf5
	cn_sto3g_reference_cvs_adc0.hdf5
	cn_sto3g_reference_cvs_adc1.hdf5
	cn_sto3g_reference_cvs_adc2.hdf5
	cn_sto3g_reference_cvs_adc2x.hdf5
	h2o_sto3g_hfdata.hdf5
	h2o_sto3g_reference_adc0.hdf5
	h2o_sto3g_reference_adc1.hdf5
	h2o_sto3g_reference_adc2.hdf5
	h2o_sto3g_reference_adc2x.hdf5
	h2o_sto3g_reference_adc3.hdf5
	h2o_sto3g_reference_cvs_adc0.hdf5
	h2o_sto3g_reference_cvs_adc1.hdf5
	h2o_sto3g_reference_cvs_adc2.hdf5
	h2o_sto3g_reference_cvs_adc2x.hdf5
	hf3_631g_hfdata.hdf5
	hf3_631g_reference_adc0.hdf5
	hf3_631g_reference_adc1.hdf5
	hf3_631g_reference_adc2.hdf5
	hf3_631g_reference_adc2x.hdf5
	hf3_631g_reference_adc3.hdf5
)

#
# -----
#

THISDIR=$(dirname "${BASH_SOURCE[0]}")
cd "$THISDIR"
echo "Updating testdata ... please wait."

download() {
	if which wget &> /dev/null; then
		wget -qN --show-progress $1
	else
		echo "wget not installed" >&2
		exit 1
	fi
}

for file in ${DATAFILES[@]}; do
	download $SOURCE/$file
done

if which sha256sum &> /dev/null; then
	sha256sum -c SHA256SUMS || exit 1
fi

touch .last_update
exit 0
