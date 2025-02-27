#!/bin/bash

# catch signals from the functions
trap "exit 1" TERM
export SCRIPT_PID=$$

SOURCE="https://wwwagdreuw.iwr.uni-heidelberg.de/adcc_test_data/0.7.0/"

SHAFILE="SHA256SUMS"

DATAFILES=(
    ch2nh2_sto3g_hfdata.hdf5
    ch2nh2_sto3g_hfimport.hdf5
    cn_sto3g_adcc_adc0.hdf5
    cn_sto3g_adcc_adc1.hdf5
    cn_sto3g_adcc_adc2.hdf5
    cn_sto3g_adcc_adc2x.hdf5
    cn_sto3g_adcc_adc3.hdf5
    cn_sto3g_adcc_mpdata.hdf5
    cn_sto3g_adcman_adc0.hdf5
    cn_sto3g_adcman_adc1.hdf5
    cn_sto3g_adcman_adc2.hdf5
    cn_sto3g_adcman_adc2x.hdf5
    cn_sto3g_adcman_adc3.hdf5
    cn_sto3g_adcman_mpdata.hdf5
    cn_sto3g_hfdata.hdf5
    cn_sto3g_hfimport.hdf5
    formaldehyde_sto3g_adcman_adc0.hdf5
    formaldehyde_sto3g_adcman_adc1.hdf5
    formaldehyde_sto3g_adcman_adc2.hdf5
    formaldehyde_sto3g_adcman_adc2x.hdf5
    formaldehyde_sto3g_adcman_adc3.hdf5
    h2o_sto3g_adcc_adc0.hdf5
    h2o_sto3g_adcc_adc1.hdf5
    h2o_sto3g_adcc_adc2.hdf5
    h2o_sto3g_adcc_adc2x.hdf5
    h2o_sto3g_adcc_adc3.hdf5
    h2o_sto3g_adcc_mpdata.hdf5
    h2o_sto3g_adcman_adc0.hdf5
    h2o_sto3g_adcman_adc1.hdf5
    h2o_sto3g_adcman_adc2.hdf5
    h2o_sto3g_adcman_adc2x.hdf5
    h2o_sto3g_adcman_adc3.hdf5
    h2o_sto3g_adcman_mpdata.hdf5
    h2o_sto3g_hfdata.hdf5
    h2o_sto3g_hfimport.hdf5
    hf_631g_adcc_adc0.hdf5
    hf_631g_adcc_adc1.hdf5
    hf_631g_adcc_adc2.hdf5
    hf_631g_adcc_adc2x.hdf5
    hf_631g_adcc_adc3.hdf5
    hf_631g_adcc_mpdata.hdf5
    hf_631g_adcman_adc0.hdf5
    hf_631g_adcman_adc1.hdf5
    hf_631g_adcman_adc2.hdf5
    hf_631g_adcman_adc2x.hdf5
    hf_631g_adcman_adc3.hdf5
    hf_631g_adcman_mpdata.hdf5
    hf_631g_hfdata.hdf5
    r2methyloxirane_sto3g_hfdata.hdf5
)
DATAFILES_FULL=(
    cn_ccpvdz_adcc_adc0.hdf5
    cn_ccpvdz_adcc_adc1.hdf5
    cn_ccpvdz_adcc_adc2.hdf5
    cn_ccpvdz_adcc_adc2x.hdf5
    cn_ccpvdz_adcc_adc3.hdf5
    cn_ccpvdz_adcc_mpdata.hdf5
    cn_ccpvdz_adcman_adc0.hdf5
    cn_ccpvdz_adcman_adc1.hdf5
    cn_ccpvdz_adcman_adc2.hdf5
    cn_ccpvdz_adcman_adc2x.hdf5
    cn_ccpvdz_adcman_adc3.hdf5
    cn_ccpvdz_adcman_mpdata.hdf5
    cn_ccpvdz_hfdata.hdf5
    cn_ccpvdz_hfimport.hdf5
    formaldehyde_ccpvdz_adcman_adc0.hdf5
    formaldehyde_ccpvdz_adcman_adc1.hdf5
    formaldehyde_ccpvdz_adcman_adc2.hdf5
    formaldehyde_ccpvdz_adcman_adc2x.hdf5
    formaldehyde_ccpvdz_adcman_adc3.hdf5
    h2o_def2tzvp_adcc_adc0.hdf5
    h2o_def2tzvp_adcc_adc1.hdf5
    h2o_def2tzvp_adcc_adc2.hdf5
    h2o_def2tzvp_adcc_adc2x.hdf5
    h2o_def2tzvp_adcc_adc3.hdf5
    h2o_def2tzvp_adcc_mpdata.hdf5
    h2o_def2tzvp_adcman_adc0.hdf5
    h2o_def2tzvp_adcman_adc1.hdf5
    h2o_def2tzvp_adcman_adc2.hdf5
    h2o_def2tzvp_adcman_adc2x.hdf5
    h2o_def2tzvp_adcman_adc3.hdf5
    h2o_def2tzvp_adcman_mpdata.hdf5
    h2o_def2tzvp_hfdata.hdf5
    h2o_def2tzvp_hfimport.hdf5
)

if ! which sha256sum &> /dev/null; then
    echo "sha256sum not installed" >&2
    exit 1
fi

# move in the folder of the script
SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
cd "$SCRIPT_DIR"

if [ ! -f "$SHAFILE" ] || [ ! -s "$SHAFILE" ]; then
    echo "$SHAFILE does not exist or is empty" >&2
    exit 1
fi

# add the datafiles for the full mode
if [ "$1" == "--full" ]; then
    DATAFILES=("${DATAFILES[@]}" "${DATAFILES_FULL[@]}")
fi


# check the SHAFILE for the file name and return the line
# containing the sha256sum and the filename
get_refshasum () {
    found=false
    while read line; do
        split=($line)
        if [ "${split[1]}" == "$1" ]; then
            found=true
            echo "$line"
            break
        fi
    done < "$SHAFILE"

    if ! $found; then
        echo "Hash for file $1 missing in $SHAFILE" >&2
        kill -s TERM $SCRIPT_PID
    fi
}
# download all given files from the server
download () {
    if which wget &> /dev/null; then
        wget -w 0.5 -q --show-progress --no-check-certificate $@
    else
        echo "wget not installed" >&2
        kill -s TERM $SCRIPT_PID
    fi
}


echo "Updating testdata ... please wait."

# Go though the files and identify files where the sha256sum
# does not match the one in the SHAFILE
FILES_TO_UPDATE=()
for file in "${DATAFILES[@]}"; do
    if [ -f "$file" ]; then
        if [ ! "$(sha256sum "$file")" == "$(get_refshasum "$file")" ]; then
            rm -f "$file"
            FILES_TO_UPDATE+=("$file")
        fi
    else
        FILES_TO_UPDATE+=("$file")
    fi
done

# exit if we don't have any missing/outdated files
if [ ${#FILES_TO_UPDATE[@]} == 0 ]; then
    echo "All files are already up to date."
    exit 0
fi

# download the remaining files
download $(for file in "${FILES_TO_UPDATE[@]}"; do echo "$SOURCE/$file"; done)
# and compare their shasums
for file in "${FILES_TO_UPDATE[@]}"; do
    if [ -f "$file" ]; then
        if [ ! "$(sha256sum "$file")" == "$(get_refshasum "$file")" ]; then
            echo "Wrong sha256sum for file $file" >&2
            exit 1
        fi
    else
        echo "missing test data file $file" >&2
        exit 1
    fi
done
