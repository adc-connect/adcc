BASEDIR=$(cd ../VeloxChemMP; pwd)
for dir in build/python .; do
	export PYTHONPATH="$PYTHONPATH:$(readlink -f "$BASEDIR/$dir")"
done
