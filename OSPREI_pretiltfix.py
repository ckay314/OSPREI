import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os

# Import all the OSPREI files, make this match your system
mainpath = '/Users/ckay/OSPREI/' #MTMYS
# ForeCAT 
sys.path.append(os.path.abspath(mainpath+'ForeCAT')) #MTMYS
from ForeCAT import *
import CME_class as CC
from ForeCAT_functions import readinputfile, calc_dist, calc_SW
# ANTEATR
sys.path.append(os.path.abspath(mainpath+'ANTEATR')) #MTMYS
from funcANTEATR import *
# FIDO/FIDO-SIT
sys.path.append(os.path.abspath(mainpath+'FIDO')) #MTMYS
from FIDO import *

# Useful constant
global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees


# Initial OSPREI setup ---------------------------------------------|
# Make use of ForeCAT function to read in vars----------------------|
# ------------------------------------------------------------------|
input_values, allinputs = FC.readinputfile()
try:
    date = input_values['date']
except:
    sys.exit('Need name of magnetogram/date to run!!!')
    
# Pull in other values from allinputs
# possible_vars = ['suffix', 'nRuns']   
# Set defaults for these values
suffix = ''
nRuns  = 1 
for i in range(len(allinputs)):
    temp = allinputs[i]
    if temp[0][:-1] == 'suffix':
        suffix = temp[1]
    elif temp[0][:-1] == 'nRuns':
        nRuns = int(temp[1])

thisName = date+suffix
print('Running '+str(nRuns)+' full OSPREI simulation(s) for '+thisName)

# See if we have a directory for output, create if not
dir = mainpath+thisName
if not os.path.exists(dir):
    os.mkdir(dir)


# ForeCAT portion --------------------------------------------------|
# ------------------------------------------------------------------|
# ------------------------------------------------------------------|

# init ForeCAT gives input params
# CME_params = [shapeA, shapeB, rstart]
# init_pos = [ilat, ilon, tilt]    
CME_params, ipos, rmax = initForeCAT(input_values)
CMEarray = []
for i in range(nRuns):
    # this works but doesn't save anything
    # right now tilt changes in opposite direction because of whole
    # reciprical thing
    thisPos = ipos + np.array([0,0,30*i])
    CME = initCME(CME_params, thisPos)
    #CME, path = runForeCAT(CME, rmax, path=True)
    CME = runForeCAT(CME, 10)
    CMEarray.append(CME)
    # Final position
    #print CME.points[CC.idcent][1]

    # Print full path
    #for i in range(len(path[0,:])):
    #    print path[0,i], path[1,i], path[2,i], path[3,i]



# ANTEATR portion --------------------------------------------------|
# ------------------------------------------------------------------|
# ------------------------------------------------------------------|

# ANTEATR takes inputs
# invec = [CMElat, CMElon, CMEtilt, CMEvel0, CMEmass, CMEAW, CMEA, CMEB, vSW, SWrho0, Cd]
# Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth!
# which we can generate from the ForeCAT data

# Pull in ANTEATR values from the input file
ANTinputs = getANTinputs(allinputs)
Epos, Cd = processANTinputs(ANTinputs)

# Assume that the given lon is the time of eruption, need to add in
# the orbit during the ForeCAT run
Erotrate = 360/365./24/60
Elons = [Epos[1]+Erotrate*CMEarray[i].t for i in range(nRuns)]

ANTvrs = []
ANTts  = []
ANTeLons = []
for i in range(nRuns):
    Epos[1] = Elons[i]
    # CME parameters from CME object
    CME = CMEarray[i]
    # CME position
    CMEr, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
    # Tilt adjustment
    #tilt = CME.tilt
    tilt = (90-CME.tilt+3600.) % 360. # in between 0 and 360
    if tilt > 180: tilt -=360.

    # Calculate vr
    vr = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
    # Mass
    mass = CME.M/1e15
    # CME shape
    cmeAW = CME.ang_width*radeg
    cmeA, cmeB = CME.shape_ratios[0], CME.shape_ratios[1]

    # Get solar wind values using the ForeCAT model
    HCSdist = calc_dist(CME.cone[1,1],CME.cone[1,2])
    SWnv = calc_SW(Epos[2],HCSdist)
    nSW, vSW = SWnv[0]/1.6727e-24, SWnv[1]/1e5

    # Package up invec, run ANTEATR
    invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeA, cmeB, vSW, nSW, Cd]
    ATresults = getAT(invec,CMEr,Epos, silent=True)

    # Add in ForeCAT time to ANTEATR time
    TotTime = ATresults[0]+CME.t/60./24
    # Assign other outputs to correct variables
    Epos[1] = ATresults[4] # start FIDO when nose dist = Earth dist, earliest possible impact
    CMEvr = ATresults[1]
    rCME = ATresults[2]
    
    # Store things to pass to FIDO ensembles
    ANTeLons.append(ATresults[4])
    ANTvrs.append(ATresults[1])
    ANTts.append(TotTime)

    print ('Contact after '+"{:.2f}".format(TotTime)+' days with velocity '+"{:.2f}".format(ATresults[1])+' km/s when nose reaches '+"{:.2f}".format(rCME) + ' Rsun')







# FIDO portion -----------------------------------------------------|
# ------------------------------------------------------------------|
# ------------------------------------------------------------------|
# Essentially just copying the relevant parts of runFIDO()
# First set up additional parameters and set to not launch GUI
# take allinputs and add relevant to input_values
input_values = read_more_inputs(allinputs, input_values)
input_values['Launch_GUI'] = 'False'
input_values['No_Plot'] = 'True' # will do our own
setupOptions(input_values, silent=True)
CMEB0 = float(input_values['CME_B0'])
CMEH  = float(input_values['CME_pol'])
try:
    CMEstart = float(input_values['CME_start'])
except:
    CMEstart = TotTime
try:
    CMEstop = float(input_values['CME_stop'])
except:
    CMEstop = 0. # it will ignore this value, but needs to be passed something
try:
    vexp = float(input_values['vExp'])
except:
    vexp = 0.  # set to zero if not given, will ignore if not using vexp expansion

for i in range(nRuns):    
    # set up input array   
    # this is all the same as done before ANTEATR, better to repeat or save values?
    Epos[1] = ANTeLons[i]
    # CME parameters from CME object
    CME = CMEarray[i]
    CMEr, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
    # Tilt adjustment
    #tilt = CME.tilt
    tilt = (90-CME.tilt+3600.) % 360. # in between 0 and 360
    if tilt > 180: tilt -=360.
    # Mass
    mass = CME.M/1e15
    # CME shape
    cmeAW = CME.ang_width*radeg
    cmeA, cmeB = CME.shape_ratios[0], CME.shape_ratios[1]
    
    # order is FFlat [0], FFlon0 [1], CMElat [2], CMElon [3], CMEtilt [4], CMEAW [5]
    # CMESRA [6], CMESRB [7], CMEvr [8], CMEB0 [9], CMEH [10], tshift [11], start [12], end [13], vexp[14]
    inps =[Epos[0], Epos[1], CMElat, CMElon, tilt,  cmeAW, cmeA, cmeB, ANTvrs[i],CMEB0, CMEH, 0., ANTts[i], 0., vexp]
    # sheath params are empty array, not using for now
    Bout, tARR = run_case(inps, [])

    cols = ['k', 'b','r', 'k']  
    fig = plt.figure()
    for i in range(len(Bout)):
        plt.plot(tARR, Bout[i], linewidth=3, color=cols[i])
    plt.show()
