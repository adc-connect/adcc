# adcc

## Access to adccore source code
```
git clone https://path/to/adccore adccore
./adccore/build.sh -j 4 install

pip install .
```

## No access to adccore source code
```
cd ./external/adccore
wget " https://path/to/binary/adccore.tar.gz"
tar xzf adccore.tar.gz
cd ../..

pip install .
```

