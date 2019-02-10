for dir in ../build/src/adcc; do
	export PYTHONPATH="$PYTHONPATH:$(readlink -f "$dir")"
done
