#!/bin/bash

if [ ! -f scripts/upload_to_anaconda.sh -o ! -f setup.py -o ! -f conda/meta.yaml.in ]; then
	echo "Please run from top dir of repository" >&2
	exit 1
fi
if [ -z "$ANACONDA_TOKEN" ]; then
	echo "Skipping build ... ANACONDA_TOKEN not set." >&2
	exit 1
fi

ADCC_VERSION=$(< setup.py awk '/__version__/ {print; exit}' | egrep -o "[0-9.]+")
ADCC_TAG=$(git tag --points-at $(git rev-parse HEAD))
if [[ "$ADCC_TAG" =~ ^v([0-9.]+)$ ]]; then
	LABEL=main
else
	LABEL=dev
	ADCC_VERSION="${ADCC_VERSION}.dev"
	ADCC_TAG=$(git rev-parse HEAD)
fi

echo -e "\n#"
echo "#-- Deploying tag/commit '$ADCC_TAG' (version $ADCC_VERSION) to label '$LABEL'"
echo -e "#\n"

set -eu
< conda/meta.yaml.in sed "s/@ADCC_VERSION@/$ADCC_VERSION/g;" > conda/meta.yaml
conda install conda-build anaconda-client --yes

# Setup channels for installing psi4 and pyscf and then build package
# conda config --append channels psi4/label/dev
conda config --append channels pyscf

# Running build and deployment
conda build conda --user adcc --token $ANACONDA_TOKEN --label $LABEL
