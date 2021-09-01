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

## Component Details
OSPREI is made of three separate models - ForeCAT, ANTEATR, and FIDO. ForeCAT uses the PFSS background to determine the external magnetic forces on a CME. These forces cause the CME to deflect in latitude and longitude and the net torque determines a rotation. The output of the model is the final latitude, longitude, and tilt of the CME in the outer corona (as well as the profiles of these parameters versus distance). The ForeCAT model is split between several different files. `ForeCAT.py` is the main wrapper function, `CME_class.py` creates a CME object that holds all the useful information about the CME and contains functions for updating it, `ForceFields.py` contains functions for calculating the background magnetic field and forces, and `ForeCAT_functions.py` contains a few random helper functions.

ANTEATR takes the ForeCAT CME and propagates it to the final satellite distance. Like most arrival time models, ANTEATR includes a drag force that causes the CME speed to approach that of the background solar wind. It also includes CME expansion and deformation from the internal thermal and magnetic forces, which was added when ANTEATR was updated to ANTEATR-PARADE. The output is the final CME speed (both propagation and expansion), size, and shape (and their profiles with distance) as well as the arrival time and internal thermal and magnetic properties of the CME. This component is fully contained in `PARADE.py`.

FIDO takes the evolved CME from ANTEATR with the position and orientation from ForeCAT and passes the CME over a synthetic spacecraft. The relative location of the spacecraft within the CME determines the in situ magnetic field vector and velocity. It also calculates the Kp index from these values. This component is fully contained in `FIDO.py`.

After running OSPREI, the results can be automatically processed using `processOSPREI.py`. This takes the standardized output files, processes them into a results object, then makes an assortment of figures depending on whether a single simulation or a full ensemble was run. Some of the figures are very straightforward, such as profiles of values versus distance or histograms. Others require more significant computation, particularly for the contour plots and ensemble correltation plots, and should potentially be commented out and not run if you want to make quick figures while testing things. New figures can be created using the results object that contains all the simulation data in an easily accessible format.

## Input Parameters
OSPREI is full specified using an text file where each input is assigned using specific terms. Here we list the terms, their meaning, and how their values can be identified for a case. Only NUMBER parameters are absolutely required and labeled as such, all others have 'reasonable' defaults that will be used if they are not specified, which are shown in italicized brackets. This is a full list of every input parameter, many of these would not need to be set for most simulations.

- `date` - The date of the eruption, in the format YYYYMMDD. *[REQUIRED]*
- `time` - The time of the start of the eruption, in the format HH:MM. This is the start of the simulation, the moment the CME begins moving, which may be before when the first motion is observed. *[REQUIRED]*
- `suffix` - A name to add as a subscript to the save files. OSPREI automatically generates file names based on the date and groups them in folders. This allows for different configurations for the same case or multiple CMEs on the same day with different tags and avoids overwriting things. *[REQUIRED]*
- `models` - Selects whether to run all of OSPREI (ALL), the individual components (FC, ANT, FIDO), or some combination (IP - ANT+FIDO, noB - FC+ANT). The inputs should always correspond to the values at the start of the simulation, whether that is at the solar surface, the far corona, or near 1 AU. *[ALL]*
- `nRuns` - The number of simulation runs. If it is set greater than 1 then the program looks for a second file with the same name as the input file but extension ".ens" that sets the parameters varied in the ensemble and their range. *[1]*
- `PFSSscale` - Option to uniformly scale the PFSS magnetic field by a constant factor. There are known scaling differences between magnetograms from different observatories or one might try and mimic a different star by scaling the solar magnetogram. This should not typically be used for a normal case but the option exists. *[1]*
- `CMEr` - The initial radius of the front of the CME. *[1.1 Rs]*
- `CMElat` - The initial latitude of the CME nose in degrees. *[REQUIRED]*
- `CMElon` - The initial longitude of the CME nose in degrees. If running ForeCAT, this needs to be in the Carrington coordinates reference frame at the time corresponding to the start of the simulation to place the CME in the PFSS background. If not running ForeCAT then other reference frames can be used as long as it used for all provided longitudes. *[REQUIRED]*
- `CMEtilt` The initial tilt or orientation of the CME, measured in degrees counterclockwise from solar west. *[REQUIRED]*
- `CMEvr` The coronal velocity of the CME in km/s. This is the maximum speed achieved in the corona and either sets the final value of a coronal three-phase propagation model and the initial speed for interplanetary propagation. *[REQUIRED]*
- `CMEAW` - The face-on angular width of the CME. Technically this is the half width but most reconstructions refer to the half width as the angular width. *[REQUIRED]*
- `CMEAWp` - The edge-on or perpendicular angular width. This defaults to one-third of the full angular width if nothing is provided as a reasonable but largely unjustified value. *[0.33 CMEAW]*
- `CMEM` - The mass of the CME at the end of the corona. It increases to this value during coronal propagation then remains constant. *[1e15 g]*
- `CMEdelAx` - The aspect ratio of the CME's central axis (radial width divided by perpendicular width) at the start of the simulation. Realistically, this is currently hard if not impossible to get a observationally measure. Likely the CME is not perfectly circular (1) or very squished (near 0), but the exact value in between is a guess. *[0.75]*
- `CMEdelCS` - The aspect ratio of the CME's cross section. This defaults to perfectly circular in the corona, which is a reasonable guess. *[1]*
- `FCrmax` - The radial distance at which ForeCAT ends and ANTEATR begins. We default to 10 Rs for "normal" CMEs as little deflection or rotation tends to happen beyond this distances. Slow, streamer blowout CMEs exhibit more motion at farther distances so the default should not be used for these cases. *[10 Rs]*
- `FCvrmin` - The radial speed of the CME during the initial slow rise phase of the coronal propagation. *[50 km/s]*
- `FCraccel1` - The radial distance where the CME's coronal propagation transitions from slow rise to rapid acceleration. *[1.3 Rs]*
- `FCraccel2` - The radial distance where the CME's coronal propagation transitions from rapid acceleration to constant propagation. *[5 Rs]*
- `SWCd` - The drag coefficient for the background solar wind. *[1]*
- `SWv` - The background solar wind speed at the location of the satellite. All solar wind properties have defaults corresponding to the average OMNI values at 1 AU over about a decade. *[400 km/s]*
- `SWn` - The background solar wind density at the location of the satellite. *[5 cm^-3]*
- `SWB` - The background solar wind magnetic field strength at the location of the satellite. *[6.9 nT]*
- `FRBscale` - The ratio of the magnetic field strength of the CME flux rope (at the central axis) relative to the background solar wind at the start of the ANTEATR simulation. It can be positive or negative to control the direction of the axial field relative to the CME geometry. *[2]*
- `FRTscale` - The ratio of the temperature of the CME relative to the background solar wind at the start of the ANTEATR simulation. *[2]*
- `FRpol` - The handedness or polarity of the CME flux rope (1 or -1 for right or left-handed). *[1]*
- `Gamma` - The adiabatic index of the flux rope, which allows the thermal expansion to vary between isothermal (1) and adiabatic (1.67). *[1.33]*
- `IVDf` - Parameter controlling the initial velocity decomposition, or how a propagation speed translates into expansion speeds, at the beginning of ANTEATR. If set to 0 then there is self-similar expansion, if set to 1 then it convects out with the solar wind. *[0.5]*
- `SatLat` - The latitude of the satellite or other body of interest. *[0°]*
- `SatLon` - The longitude of the satellite or other body of interest. *[0°]*
- `SatR` - The radial distance of the satellite or other body of interest. *[213 Rs]*
- `SatRot` - The orbital speed of the satellite or other body of interest. *[2.8e-6 rad/s]*
- `includeSIT` - Whether or not to include the sheath in front of the flux rope in FIDO. It does require a CME moving faster than the background solar wind at the time of arrival. Can only be set to True or False. *[False]*
- `ObsDataFile` - Name of a text file containing observational data to compare to in the figures.

Of the non-required input parameters, we strongly suggest at least making educated guesses for CMEAWp, CMEM, FRBscale, FRTscale, FRpol, and the solar wind properties rather than resorting to defaults. These CME properties likely scale with CME size/speed so it is likely better to use values larger than the defaults for fast CMEs and smaller than for slow CMEs.

