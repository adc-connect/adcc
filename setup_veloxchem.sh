BASEDIR=$(cd ../../../Forschung_Wissenschaft/Gator/gator; pwd)
for dir in external/VeloxChemMP/build/python .; do
	export PYTHONPATH="$PYTHONPATH:$(readlink -f "$BASEDIR/$dir")"
done
