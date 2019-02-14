# adcc
A python-based framework for running ADC calculations.

## Installation
Once we have figured out how to do it, it should be as simple as
```
pip install adcc
```
Right now, the package is not uploaded to pypi, however.


## Development setup use cases
### Access to adccore source code
Having cloned this `adcc` repository,
proceed to clone `adccore` **into the `adcc` directory**,
that is, run
```
git clone https://path/to/adccore adccore
```
inside the directory, where this **README.md** file is found.

Then build, test and (optionally) install `adcc`.
```
./setup.py test   # Builds and tests adcc
pip install .     # Installs adcc (optional)
```

### Without access to adccore source code
Download the tarball of the compiled `adccore` library
into the `external/adccore` directory:
```
cd ./external/adccore
wget " https://path/to/binary/adccore-VERSION.tar.gz"
tar xzf adccore.tar.gz
cd ../..
```

Afterwards proceed as above, that is:
```
./setup.py test   # Builds and tests adcc
pip install .     # Installs adcc (optional)
```

## Commands of the `setup.py` script worth knowing
The following commands of the `setup.py` script are very useful
for development and worth knowing:

- `setup.py build_ext`: Build the `C++` part of `adcc` in the current directory.
  This includes `adccore` in case you have the source code repository set up as
  described above.
- `setup.py test`: Run the `adcc` unit tests via `pytest`. Implies `build_ext`.
