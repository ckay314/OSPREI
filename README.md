# OSPREI
OSPREI is a suite of models that simulate the Sun-to-Earth (or satellite) behavior of CMEs. 'OSPREI.py' is the main function, and is called by passing a text file to it on the command line.
```
python OSPREI.py OSPtestcase.txt
```
OSPREI runs ForeCAT for coronal CME deflection and rotation, ANTEATR (PARADE) for interplanetary CME propagation, expansion, and deformation, and FIDO for in situ profiles. Systematic output files are generated, which can then be automatically processed into user-friendly figures.
```
python processOSPREI.py OSPtestcase.txt
```

## Before Running OSPREI

## Input Parameters

## Component Details