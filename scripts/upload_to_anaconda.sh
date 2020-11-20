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

# TODO: remove
whoami
echo $(which conda)

PYREGEX="$1"
set -eu

# Build conda/meta.yaml and conda_build_config.yaml
if [ "$PYREGEX" ]; then
	# Apply version restriction regex
	 < conda/conda_build_config.yaml awk -v "vregex=$PYREGEX" '
		/^[a-z_]+:/ {pr = 0}  # Switch on printing for any new yaml keys
		pr == 1 && $0 ~ "^[[:space:]]+- " vregex {print; next}  # print matches
		pr == 0               # Print if pr is 0 (unset or printing enabled)
		/^python:/ {pr=1}     # Switch off printing if we have entered python:
	' > conda_build_config.yaml.temp
	mv conda_build_config.yaml.temp conda/conda_build_config.yaml
fi
< conda/meta.yaml.in sed "s/@ADCC_VERSION@/$ADCC_VERSION/g;" > conda/meta.yaml

# Install requirements and setup channels
conda install conda-build anaconda-client --yes
# conda config --append channels psi4/label/dev
# conda config --append channels pyscf

# Running build and deployment
conda build conda -c defaults -c conda-forge --user adcc --token $ANACONDA_TOKEN --label $LABEL
