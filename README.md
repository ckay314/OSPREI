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
OSPREI is full specified using an text file where each input is assigned using specific terms. Here we list the terms, their meaning, and how their values can be identified for a case. Parameters that are listed in bold are required, all others have 'reasonable' defaults that will be used if they are not specified, which are shown in brackets.
- **date** - The date of the eruption, in the format YYYYMMDD.
- **time** - The time of the start of the eruption, in the format HH:MM. This is the start of the simulation, the moment the CME begins moving, which may be before when the first motion is observed.
- **suffix** - A name to add as a subscript to the save files. OSPREI automatically generates file names based on the date and groups them in folders. This allows for different configurations for the same case or multiple CMEs on the same day with different tags and avoids overwriting things. 
- models - Selects whether to run all of OSPREI (ALL), the individual components (FC, ANT, FIDO), or some combination (IP - ANT+FIDO, noB - FC+ANT). The inputs should always correspond to the values at the start of the simulation, whether that is at the solar surface, the far corona, or near 1 AU. *[ALL]*
- nRuns - The number of simulation runs. If it is set greater than 1 then the program looks for a second file with the same name as the input file but extension ".ens" that sets the parameters varied in the ensemble and their range. *[1]*
- PFSSscale - Option to uniformly scale the PFSS magnetic field by a constant factor. There are known scaling differences between magnetograms from different observatories or one might try and mimic a different star by scaling the solar magnetogram. This should not typically be used for a normal case but the option exists. *[1]*
- CMEr - The initial radius of the front of the CME. *[1.1 Rs]*
- **CMElat** - The initial latitude of the CME nose in degrees.
- **CMElon** - The initial longitude of the CME nose in degrees. If running ForeCAT, this needs to be in the Carrington coordinates reference frame at the time corresponding to the start of the simulation to place the CME in the PFSS background. If not running ForeCAT then other reference frames can be used as long as it used for all provided longitudes.
- **CMEtilt** The initial tilt or orientation of the CME, measured in degrees counterclockwise from solar west.
- **CMEvr** The coronal velocity of the CME in km/s. This is the maximum speed achieved in the corona and either sets the final value of a coronal three-phase propagation model and the initial speed for interplanetary propagation.
- **CMEAW** - The face-on angular width of the CME. Technically this is the half width but most reconstructions refer to the half width as the angular width.
- CMEAWp - The edge-on or perpendicular angular width. This defaults to one-third of the full angular width if nothing is provided as a reasonable but largely unjustified value. *[0.33 CMEAW]*
- CMEM - The mass of the CME at the end of the corona. It increases to this value during coronal propagation then remains constant. *[1e15 g]*
- CMEdelAx - The aspect ratio of the CME's central axis (radial width divided by perpendicular width) at the start of the simulation. Realistically, this is currently hard if not impossible to get a observationally measure. Likely the CME is not perfectly circular (1) or very squished (near 0), but the exact value in between is a guess. *[0.75]*
- CMEdelCS - 
- FCrmax
- FCvrmin 
- FCraccel1
- FCraccel2
- SWCd
- **SWv**
- **SWn**
- **SWB**
- FRBscale
- FRTscale
- FRpol
- Gamma
- IVDf
- SatLat
- SatLon
- SatR
- SatRot
- includeSIT
- ObsDataFile



## Component Details
