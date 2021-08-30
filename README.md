# OSPREI
OSPREI is a suite of models that simulate the Sun-to-Earth (or satellite) behavior of CMEs. `OSPREI.py` is the main function, and is called by passing a text file to it on the command line.
```
python OSPREI.py OSPtestcase.txt
```
OSPREI runs ForeCAT for coronal CME deflection and rotation, ANTEATR (PARADE) for interplanetary CME propagation, expansion, and deformation, and FIDO for in situ profiles. Systematic output files are generated, which can then be automatically processed into user-friendly figures.
```
python processOSPREI.py OSPtestcase.txt
```

## Before Running OSPREI
Before the main simulation can be performed the solar background must be set up. Specifically, the background magnetic field must created using a PFSS model. First, a magnetogram must be acquired. We recommend using an HMI synchronic magnetogram for the day of the eruption (or the day before if the eruption occurs before the time of the magnetogram). These are available from the JSOC catalog. 

Once the fits file is downloaded, the script `sync2synop.py` shifts the magnetogram to the standard Carrington rotation frame. The HMI synchcronic magnetograms have the newest data in the first 120° of longitude. This function pulls the Carrington longitude for the time of observation from the fits header and shifts things accordingly. We save the fits files as `HMIYYYYMMDDsync.fits` in a specific folder. These first few lines should be changed to make the file path match your system, naming format, and specific date.

The next step is to run `harmonics.py`, which uses the magnetogram to calculate the harmonic coefficients for the PFSS background. Again, the `fits.open()` portion should be changed to match your folder system. The script is set up to pull in the date corresponding to the magnetogram, simply to pull in the correct file name. The format can match whatever you are using, it doesn't need to be in a specific date format, it's just a text string. This script will take a few minutes to run and should show progress as it loops through the user set number of harmonic coefficients.

The final step for the magnetic background is to run `PFSS.py`. This script pulls in a date string at the python call, just to find the correct text file containing the harmonic coefficients. Within the `initcoeffs` function, the `fname=` should be set to pull in from the correct path and file name. This script will take some time to run as it calculated the magnetic field vector on a 0.5°x0.5°x0.1 Rs grid between 1 Rs and 2.5 Rs, probably 30 minutes to an hour depending on the computer. This creates two large pickles (python binary data files), each about 630 MB. OSPREI (or at least the ForeCAT portion) will interpolate the background magnetic field from these pickles rather than calculating the vector on the fly during the simulation. This only needs to be ran once for a specific day/simulation.

We also include a pair of scripts that help with finding a CME's initial position, which can be used to initiate OSPREI/ForeCAT. `selectregion.py` opens up a PFSS pickle and displays the full map. The file path needs to be changed and it pulls in the date string from the command line. It is set to show the magnetic field strength at 1.05 Rs, which we find to be a good height for visualizing the polarity inversion line (PIL). Any lower and there tends to be 'ringing' effects from the PFSS model, any higher and active regions become less complex. It may be appropriate to switch to a higher height for quiet sun/streamer belt/stealth CMEs. Using the built in python window, one can zoom in and find the indices that form a nice rectangle around the source region. These indices can be passed to `ForeCATARPILER.py`, the ForeCAT Active Region PIL Extraction Routine, which will fit the PIL and determine a latitude, longitude, and tilt. This routine is called without any additional information on the command line but the `fitPIL` command should be passed the date string, x1, x2, y1, y2 (where x and y are the horizontal and vertical indices). This routine is largely untested and should be considered a beta program, but it gives a good automatically-generated match to what one would visually identify as a PIL.

## Input Parameters

## Component Details
