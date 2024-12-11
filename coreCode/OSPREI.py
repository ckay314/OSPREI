import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import datetime
from scipy.interpolate import CubicSpline

# Import path names from file
myPaths = dict(np.genfromtxt('myPaths.txt', dtype=str))
mainpath = myPaths['mainpath:']
codepath = myPaths['codepath:']
magpath  = myPaths['magpath:']
propath  = myPaths['processpath:']

# Previous hardcoded version
# Import all the OSPREI files, make this match your system
#mainpath = '/Users/ckay/OSPREI/' #MTMYS
#codepath = mainpath + 'coreCode/'
#magpath  ='/Users/ckay/OSPREI/PickleJar/'
sys.path.append(os.path.abspath(codepath)) 

# Can add other folders as needed if you store anything somewhere else
# (e.g. satFiles)
# only necessary if folders aren't in same place as where you run
#sys.path.append(os.path.abspath('/User/ckay/OSPREI/exampleFiles'))

from ForeCAT import *
import CME_class as CC
from ForeCAT_functions import readinputfile, calc_dist, calc_SW
import ForceFields 
#from funcANTEATR import *
#from PARADE import *
from ANT_PUP import *
#import FIDO as FIDO

# Useful constant
global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

# pick a number to seed the random number generator used by ensembles
np.random.seed(20220310)

# turn off all printing
allSilent = False

global sheathParams
# hack for adding in sheath on combo 
sheathParams = [0.,0.,0.]



def setupOSPREI(logInputs=False, inputPassed='noFile'):
    # Initial OSPREI setup ---------------------------------------------|
    # Make use of ForeCAT function to read in vars----------------------|
    # ------------------------------------------------------------------|
    global input_values, allinputs, date, noDate
    input_values, allinputs = FC.readinputfile(inputPassed=inputPassed)
    if logInputs:
        add2inputlog(input_values)
    checkInputs()
    
    noDate = False
    try:
        date = input_values['date']
    except:
        date = '99999999'
        noDate = True
        #sys.exit('Need name of magnetogram/date to run!!!')    
            
    # Pull in other values from allinputs
    # possible_vars = ['suffix', 'nRuns']   
    # Set defaults for these values
    global suffix, nRuns
    # these are values its convenient to read early for processOSPREI
    global time, satPos, Sat_rot, ObsDataFile, mass, useFCSW, flagScales, flag1DSW, doPUP, doMH, isSat, simYaw
    global obsFRstart, obsFRend, obsShstart, vSW, MHarea, MHdist, satPath, FRpol
    suffix = ''
    nRuns  = 1
    satPos = [0,0,213]
    shape  = [1,0.15] 
    Sat_rot = 360./365.25/24/60/60.
    ObsDataFile = None
    useFCSW = False
    flagScales = False
    flag1DSW = False
    doPUP   = False
    doMH    = False
    simYaw  = False
    isSat   = True
    obsFRstart, obsFRend, obsShstart = [None], [None], [None]
    mass = 5.
    FRpol = 1
    # Read in values from the text file
    for i in range(len(allinputs)):
        temp = allinputs[i]
        if temp[0][:-1] == 'suffix':
            suffix = temp[1]
        elif temp[0][:-1] == 'nRuns':
            nRuns = int(temp[1])
        elif temp[0][:-1] == 'models':
            models = temp[1]
        elif temp[0][:-1] == 'time':
            time = temp[1]
        elif temp[0][:-1] == 'SatLat':
            satPos[0] = float(temp[1])
        elif temp[0][:-1] == 'SatLon':
            satPos[1] = float(temp[1])
        elif temp[0][:-1] == 'SatR':
            satPos[2] = float(temp[1])
        elif temp[0][:-1] == 'SatRot':
            Sat_rot = float(temp[1])
        elif temp[0][:-1] == 'ObsDataFile':
            ObsDataFile = temp[1]
        elif temp[0][:-1] == 'doPUP':
            if temp[1] == 'True':
                doPUP = True
        elif temp[0][:-1] == 'CMEM':
            mass = float(temp[1])
        elif temp[0][:-1] == 'useFCSW':
            if temp[1] == 'True':
                useFCSW = True   
        elif temp[0][:-1] == 'flagScales':
            if temp[1] == 'True':
                flagScales = True
        elif temp[0][:-1] == 'SWfile':
            flag1DSW  = True
            global SWfile
            SWfile = temp[1]
        elif temp[0][:-1] == 'doMH':
            if temp[1] == 'True':
                doMH = True
        elif temp[0][:-1] == 'simYaw':
            if temp[1] == 'True':
                simYaw = True
        elif temp[0][:-1] == 'isSat':
            if temp[1] == 'True':
                isSat = True
        # Needed for processOSP metrics
        elif temp[0][:-1] == 'obsFRstart':
            obsFRstart = float(temp[1])
        elif temp[0][:-1] == 'obsFRend':
            obsFRend = float(temp[1])
        elif temp[0][:-1] == 'obsShstart':
            obsShstart = float(temp[1])
        # Needed for enlilesque plot
        elif temp[0][:-1] == 'SWv':
            vSW = float(temp[1])
        elif temp[0][:-1] == 'MHarea':
            MHarea = float(temp[1])
        elif temp[0][:-1] == 'MHdist':
            MHdist = float(temp[1])
        elif temp[0][:-1] == 'satPath':  
            satPath = temp[1]
            isSat = True
        # dunno why this was missing
        elif temp[0][:-1] == 'FRpol':
             FRpol = int(temp[1])
    
    # check if we have a magnetogram name for ForeCAT or if passed only the date
    global pickleName
    if models in ['FC', 'ALL', 'All']:
        if 'FCmagname:' in allinputs:
            pickleName = allinputs[np.where(allinputs == 'FCmagname:')[0]][0,1]
        elif not noDate:
            pickleName = date
            print ('Assuming pickles are named by date ', date)
        else:
            sys.exit('Need name of magnetogram/date to run!!!') 
    
    if not noDate:        
        # get the actual time for the given string
        yr  = int(date[0:4])
        mon = int(date[4:6])
        day = int(date[6:])
        # Check if we were given a time
        try:
            hrs, mins = int(time[:2]), int(time[3:])
        except:
            hrs, mins = 0, 0
        global dObj, Doy
        dObj = datetime.datetime(yr, mon, day,hrs,mins)
        dNewYear = datetime.datetime(yr, 1, 1, 0,0)
        DoY = (dObj - dNewYear).days + (dObj - dNewYear).seconds/3600./24.
        if not allSilent:
            print('Simulation starts at '+dObj.strftime('%Y %b %d %H:%M '))
                        
    global thisName
    if noDate:
        thisName = suffix
    else:
        thisName = date+suffix
    if not allSilent:
        print('Running '+str(nRuns)+' OSPREI simulation(s) for '+thisName)

    # See if we have a directory for output, create if not
    # I save things in individual folders, but you can dump it whereever
    # by changing Dir
    global Dir
    if noDate:
        Dir = mainpath+'NoDate'
    else:
        Dir = mainpath+date
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    
    
    # check if running not running FC and not provided rCME
    # need to switch default to ANT value of 21.5 Rs
    if not doFC and (float(input_values['CMEr']) < 10):
        if doANT:
            input_values['CMEr'] = 21.5
        else:
            input_values['CMEr'] = satPos[2]
    if not allSilent:        
        print( 'Running: ')
        print( 'ForeCAT: ', doFC)
        print( 'ANTEATR: ', doANT)
        print( 'FIDO:    ', doFIDO)
        print('')

def add2inputlog(input_values):
    # Enable Reproducibility with the Input Keeper Assistant - ERIKA mode
    # make the line we want to add
    outline = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S") + ',  '
    for item in input_values:
        outline += (item+': '+str(input_values[item]) + ', ')
    outline = outline[:-2]
    
    # check if file exists to start
    logFile = 'inputs.log'
    # get the date to find the right output folder
    try:
        date = input_values['date']
    except:
        date = 'NoDate'
    # write in an existing file or start a new one    
    if os.path.isfile(mainpath+date+'/'+logFile):
        fOut = open(mainpath+date+'/'+logFile, 'a')
        fOut.write('\n'+outline)
        fOut.close()
    else:
        os.makedirs(mainpath+date+'/', exist_ok=True)
        fOut = open(mainpath+date+'/'+logFile, 'w')
        fOut.write(outline)
        fOut.close()

def checkInputs(printNonCrit=False):    
    # Run through all the input parameters and check that everything is in appropriate range
    # Set to defaults if not provided or exit if must be explicitly provided
    
    global models
    if 'models' in input_values:
        models = input_values['models']
        if models not in ['ALL', 'All', 'noB', 'IP', 'FC', 'ANT', 'FIDO']:
            sys.exit('Models keyword not understood, select from ALL/noB/IP/FC/ANT/FIDO')
    else:
        print('models keyword not provided, assuming ALL')
        models = 'ALL'    
    # Check to see if we are running all components or just part
    # flags are ALL = all three, IP = ANTEATR+FIDO, ANT = ANTEATR, FIDO = FIDO    
    # FC = ForeCAT only, noB = ForeCAT+ANTEATR
    global doFC, doANT, doFIDO
    doFC, doANT, doFIDO = True, True, True
    if models in ['IP', 'FIDO', 'ANT']: doFC = False
    if models in ['ANT', 'noB']: doFIDO = False
    if models == 'FIDO': doANT = False
    if models == 'FC': doANT, doFIDO, = False, False
    global startpoint
    if doFIDO:
        startpoint = 'FIDO'
    if doANT:
        startpoint = 'ANT'
    if doFC:
        startpoint = 'FC'
    
    # Required parameters --------------------------------------------------------------------------------
    if 'CMElat' in input_values:
        CMElat = float(input_values['CMElat'])
        if np.abs(CMElat) > 90.:
            sys.exit('CME latitude (CMElat) must be within +/- 90 deg') 
    else:
        sys.exit('CME latitude (CMElat) required to run')
        
    if 'CMElon' in input_values:
        CMElon = float(input_values['CMElon'])
        if (CMElon < -180) or (CMElon > 360) :
            sys.exit('CME longitude (CMElon) must be within [-180, 180] or [0,360] deg') 
    else:
        sys.exit('CME longitude (CMElon) required to run')
        
    if 'CMEtilt' in input_values:
        CMEtilt = float(input_values['CMEtilt'])
        if np.abs(CMEtilt) > 90.:
            sys.exit('CME tilt (CMEtilt) must be within +/- 90') 
    else:
        sys.exit('CME tilt (CMEtilt) required to run')

    if 'CMEvr' in input_values:
        CMEvr = float(input_values['CMEvr'])
        if np.abs(CMEvr) < 0.:
            sys.exit('CME coronal velocity (CMEvr) must > 0 km/s') 
    else:
        sys.exit('CME coronal velocity (CMEvr) required to run')

    if 'CMEAW' in input_values:
        CMEAW = float(input_values['CMEAW'])
        if (CMEAW < 5) or (CMEAW > 180) :
            sys.exit('CME Angular Width (CMEAW) must be within [5, 180] degrees') 
    else:
        sys.exit('CME Angular Width (CMEAW) required to run')
    
    if 'CMEyaw' in input_values:
        CMElat = float(input_values['CMElat'])
        if np.abs(CMElat) > 90.:
            sys.exit('CME yaw (CMEyaw) must be within +/- 90 deg') 
    

    # Important parameters with defaults ----------------------------------------------------------------
    if 'CMEr' in input_values:
        CMEr = float(input_values['CMEr'])
        if startpoint == 'FC':
            if (CMEr < 1.05) or (CMEr > 2.5):
                sys.exit('Initial CME front distance (CMEr) must be within [1.05, 2.5] Rs for ForeCAT start') 
        elif startpoint == 'ANT':
            if (CMEr < 10) or (CMEr > 500):
                sys.exit('Initial CME front distance (CMEr) must be within [10, 50] Rs for ANTEATR start') 
        # FIDO can start wherever but at least check that units are semi reasonable
        # and within Jupiter distance?
        if (CMEr < 1.05) or (CMEr > 1100):
            sys.exit('Initial CME front distance (CMEr) must be within [1.05, 1100] Rs for FIDO start') 
    else:
        if startpoint == 'FC':
            input_values['CMEr'] = str(1.1)    
        elif startpoint == 'ANT':
            input_values['CMEr'] = str(21.5)
        elif startpoint == 'FIDO':
            if 'SatR' in input_values:
                input_values['CMEr'] = input_values['SatR']
            elif 'satPath' in input_values:
                # set it to temp value we know we will replace later?
                input_values['CMEr'] = input_values['9999']       
    
    if 'CMEAWp' in input_values:
        CMEAWp = float(input_values['CMEAWp'])
        if (CMEAWp < 5) or (CMEAWp > 90) :
            sys.exit('CME perpendicular Angular Width (CMEAWp) must be within [5, 90] degrees') 
    else:
        if printNonCrit:
            print('No perpendicular CME Angular Width (CMEAWp) given. Setting at 1/3 AW')
        input_values['CMEAWp'] = str(1/3*float(input_values['CMEAW']))
        
    if 'CMEdelAx' in input_values:
        CMEdelAx = float(input_values['CMEdelAx'])
        if (CMEdelAx < 0.1) or (CMEdelAx > 1.5) :
            sys.exit('CME Axial Aspect Ratio (CMEdelAx) must be within [0.1, 1.5]') 
    else:
        input_values['CMEdelAx'] = str(0.75)
    
    if 'CMEdelCS' in input_values:
        CMEdelCS = float(input_values['CMEdelCS'])
        if (CMEdelCS < 0.1) or (CMEdelCS > 1.5) :
            sys.exit('CME CS Aspect Ratio (CMEdelCS) must be within [0.1, 1.5]') 
    else:
        input_values['CMEdelCS'] = str(1)
    
    if 'CMEM' in input_values:
        CMEM = float(input_values['CMEM'])
        if (CMEM < 0.1):
            sys.exit('CME mass (CMEM) must be > 0.1 (input units 1e15 g)') 
    else:
        if printNonCrit:
            print ('Calculating CME mass from velocity since not explicitly given')
        # Updated to LLAMACoRe v1.0 relation
        CMEM = 0.010 * float(input_values['CMEvr']) +0.16
        if CMEM > 0:
            input_values['CMEM'] = str(CMEM)
            if not allSilent:
                print ('Using '+str(CMEM) +' x10^15 g for CMEM')
        else:
            sys.exit('Cannot calc CME mass from vr, vr too small (<146 km/s) and produces negative mass with given regression')
    

    # Typically unused ForeCAT parameters ----------------------------------------------------------------
    if 'FCrmax' in input_values:
        FCrmax = float(input_values['FCrmax'])
        if (FCrmax < 10) or (FCrmax > 21.5) :
            sys.exit('ForeCAT rmax (FCrmax) must be within [10, 21.5] Rs') 
    else:
        input_values['FCrmax'] = str(21.5)        
        
    if 'FCraccel1' in input_values:
        FCraccel1 = float(input_values['FCraccel1'])
        if (FCraccel1 < 1) or (FCraccel1 > 2) :
            sys.exit('ForeCAT raccel1 (FCraccel1) must be within [1, 2] Rs') 
    else:
        input_values['FCraccel1'] = str(1.3)        
    
    if 'FCraccel2' in input_values:
        FCraccel2 = float(input_values['FCraccel2'])
        if (FCraccel2 < 1.5) or (FCraccel2 > 21.5) :
            sys.exit('ForeCAT raccel2 (FCraccel2) must be within [1.5, 21.5] Rs') 
        if (FCraccel2 > float(input_values['FCrmax'])):
            sys.exit('ForeCAT raccel2 greater than ForeCAT rmax, needs to reach full speed before/at end of ForeCAT')
    else:
        input_values['FCraccel2'] = str(10)        
        
    if 'FCvrmin' in input_values:
        FCvrmin = float(input_values['FCvrmin'])
        if (FCvrmin < 10) or (FCvrmin > 200) :
            sys.exit('ForeCAT initial velocity (FCvrmin) must be within [10, 200] km/s') 
    else:
        input_values['FCvrmin'] = str(50)        
        
    if 'FCAWmin' in input_values:
        FCAWmin = float(input_values['FCAWmin'])
        if (FCAWmin < 1) or (FCAWmin > 20) :
            sys.exit('ForeCAT initial AW (FCAWmin) must be within [1, 20] deg') 
    else:
        input_values['FCAWmin'] = str(5)        
        
    if 'FCAWr' in input_values:
        FCAWr = float(input_values['FCAWr'])
        rlim = (float(input_values['FCrmax']) - 1)/4.61 # limit so reaches ~full size before end of ForeCAT
        if (FCAWr < 0.1) or (FCAWr > rlim) :
            sys.exit('ForeCAT expansion length scale (FCAWr) must be within [0.1, (FCrmax - 1)/4.61] Rs') 
    else:
        input_values['FCAWr'] = str(1)     
        
    if 'FCrmaxM' in input_values:
        FCrmaxM = float(input_values['FCrmaxM'])
        if  (FCrmaxM > float(input_values['FCrmax'])) :
            sys.exit('ForeCAT initial AW (FCrmaxM) must be < FCrmax Rs') 
    else:
        input_values['FCrmaxM'] = input_values['FCrmax']    
           
    if 'SWCdp' in input_values:
        SWCdp = float(input_values['SWCdp'])
        if (SWCdp < 0) or  (SWCdp > 10):
            sys.exit('ForeCAT initial AW (SWCdp) must be in [0, 10]') 
    else:
        input_values['SWCdp'] = str(1) 
        
    if 'FCNtor' in input_values:
        FCNtor = float(input_values['FCNtor'])
        if (FCNtor < 5) or  (FCNtor > 25):
            sys.exit('Number of grid points along ForeCAT toroidal axis (FCNtor) must be in [5, 25]') 
    else:
        input_values['FCNtor'] = str(15) 

    if 'FCNpol' in input_values:
        FCNpol = float(input_values['FCNpol'])
        if (FCNpol < 5) or  (FCNpol > 25):
            sys.exit('Number of grid points along ForeCAT poloidal axis (FCNpol) must be in [5, 25]') 
    else:
        input_values['FCNpol'] = str(13) 

    if 'PFSSscale' in input_values:
        PFSSscale = float(input_values['PFSSscale'])
        if (PFSSscale < 0.1) or  (PFSSscale > 1000):
            sys.exit('Uniform scaling of PFSS model (PFSSscale) must be in [0.1, 1000]') 
    else:
        input_values['PFSSscale'] = str(1) 


    # Flux rope parameters --------------------------------------------------------------------------------
    if 'FRpol' in input_values:
        if input_values['FRpol'] not in ['1', '+1', '-1']:
            sys.exit('FRpol input not understood. Set to 1 (right-handed) or -1 (left-handed)')
    else:
        if float(input_values['CMElat']) > 0:
            input_values['FRpol'] = '-1'
        else:
            input_values['FRpol'] = '1'
        if printNonCrit:
            print('Flux rope handedness not given. Setting based on hemisphere (Bothmer-Schwenn, north = negative) but this is only statistically valid')
    
    if 'FRtau' in input_values:
        FRtau = float(input_values['FRtau'])
        if (FRtau < 0.1) or  (FRtau > 3):
            sys.exit('Flux rope tau (FRtau) must be in [0.1, 3]') 
    else:
        input_values['FRtau'] = str(1) 

    if 'FRCnm' in input_values:
        FRCnm = float(input_values['FRCnm'])
        if (FRCnm < 0.5) or (FRCnm > 3):
            sys.exit('Flux rope Cnm (FRCnm) must be in [0.5, 3]') 
    else:
        input_values['FRCnm'] = str(1.927) 
    
        
    # ANTEATR parameters --------------------------------------------------------------------------------
    if 'IVDf1' in input_values:
        IVDf1 = float(input_values['IVDf1'])
        if (IVDf1 < 0) or  (IVDf1 > 1):
            sys.exit('Initial velocity decomposition 1 (IVDf1) must be in [0, 1]') 
    else:
        input_values['IVDf1'] = str(0.5) 

    if 'IVDf2' in input_values:
        IVDf2 = float(input_values['IVDf2'])
        if (IVDf2 < 0) or  (IVDf2 > 1):
            sys.exit('Initial velocity decomposition 2 (IVDf2) must be in [0, 1]') 
    else:
        input_values['IVDf2'] = str(0.5) 

    if 'IVDf' in input_values:
        IVDf = float(input_values['IVDf'])
        if (IVDf < 0) or  (IVDf > 1):
            sys.exit('Initial velocity decomposition (IVDf) must be in [0, 1]') 
    else:
        input_values['IVDf'] = str(0.5) 
                
    if 'Gamma' in input_values:
        Gamma = float(input_values['Gamma'])
        if (Gamma < 1) or  (Gamma > 1.67):
            sys.exit('Adiabatic index (Gamma) must be in [1, 1.67]') 
    else:
        input_values['Gamma'] = str(1.33) 
    
    if 'doPUP' in input_values:
        doPUP = input_values['doPUP']
        if doPUP not in ['True', 'False']:
            sys.exit('Inclusion of PUP in ANTEATR (doPUP) can only be True or False') 
    
    flagScales = False
    if 'flagScales' in input_values:
        fS = input_values['flagScales']
        if fS not in ['True', 'False']:
            sys.exit('Option to flag inputs as scaled B and T (flagScales) can only be True or False')
        if fS == 'True':
            flagScales = True 
    
    # Include ability to calculate B/T from empirical relations if not given (or told to scale)
    if not flagScales:
        dtor = 3.14159 / 180
        rFront = float(input_values['FCrmax']) * 7e10
        AW = float(input_values['CMEAW']) * dtor
        AWp = float(input_values['CMEAWp']) * dtor
        delAx = float(input_values['CMEdelAx'])
        delCS = float(input_values['CMEdelCS'])
        v = float(input_values['CMEvr'])
        Cnm = float(input_values['FRCnm'])
        tau = float(input_values['FRtau'])
        mass = float(input_values['CMEM'])
        
        # Function to estimate length of torus for on weird ellipse/parabola hybrid
        lenCoeffs = [0.61618, 0.47539022, 1.95157615]
        lenFun = np.poly1d(lenCoeffs)
        
        # need CME R and len to convert phiflux to B0
        rCSp = np.tan(AWp) / (1 + delCS * np.tan(AWp)) * rFront
        rCSr = delCS * rCSp
        Lp = (np.tan(AW) * (rFront - rCSr) - rCSr) / (1 + delAx * np.tan(AW))  
        Ltorus = lenFun(delAx) * Lp
        # Ltorus needs to include legs
        rCent = rFront - delAx*Lp - rCSp
        Lleg = np.sqrt(Lp**2 + rCent**2) - 7e10 # dist from surface
        Ltot = Ltorus + 2 * Lleg 
        avgR = (0.5*rCSp * Lleg * 2 + rCSp * Ltorus) / (Lleg*2 + Ltorus)
        
        if 'FRB' in input_values:
            FRB = float(input_values['FRB'])
            if (np.abs(FRB) < 5) or  (np.abs(FRB) > 15000):
                if float(input_values['CMEr']) < 30.:
                    sys.exit('Flux rope |B| (FRB) must be in [500, 15000] nT') 
                
        else:
            KE = 0.5 * mass*1e15 * (v*1e5)**2 /1e31
            phiflux = np.power(10, np.log10(KE / 0.19) / 1.87)*1e21
            B0 = phiflux * Cnm * (delCS**2 + 1) / avgR / Ltot / delCS**2 *1e5
            Bcent = delCS * tau * B0 
            if (Bcent < 500) or  (Bcent > 20000):
                sys.exit('Cannot calculate a reasonable default flux rope B using empirical scaling. Please provide FRB')
            else:
                if not allSilent:
                    print('Using ', Bcent, ' nT for FRB')
                input_values['FRB'] = str(Bcent)

        if 'FRT' in input_values:
            FRT = float(input_values['FRT'])
            if (FRT < 2e4) or  (FRT > 30e6):
                sys.exit('Flux rope T (FRT) must be in [5e4, 3e6] nT') 
        else:
            vSheath = 0.129 * v + 376
            vIS = (vSheath + 51.73) / 1.175
            vExp = 0.175 * vIS -51.73
            logTIS = 3.07e-3 * vIS +3.65
            FRT = np.power(10, logTIS) * np.power(215*7e10/rFront, 0.7)
            if (FRT < 5e4) or  (FRT > 3e6):
                sys.exit('Cannot calculate a reasonable default flux rope T using empirical scaling. Please provide FR T')
            else:
                if not allSilent:
                    print('Using ', FRT, ' K for FRT')
                input_values['FRT'] = str(FRT)
        
        
    # Satellite Parameters ----------------------------------------------------------------
    hasSatPath = False
    isMulti = False
    if 'satPath' in input_values:
        hasSatPath = True
        if input_values['satPath'][-4:] == 'sats':
            isMulti = True
        
    if 'SatLon' in input_values:
        SatLon = float(input_values['SatLon'])
        if (SatLon < -180) or  (SatLon > 360):
            sys.exit('Satellite longitude (SatLon) must be in [-180, 180] or [0, 360] deg') 
    elif not isMulti:
        sys.exit('Initial satellite longitude (SatLon) is required. Must be in Carrington coordinates if ForeCAT is used.') 
    
    if not hasSatPath:
        if 'SatLat' in input_values:
            SatLat = float(input_values['SatLat'])
            if (SatLat < -90) or (SatLat > 90) :
                sys.exit('Initial satellite latitude (SatLat) must be within [-90, 90] deg') 
        else:
            sys.exit('Initial satellite latitude (SatLat) required to run')
        
        if 'SatR' in input_values:
            SatR = float(input_values['SatR'])
            if (SatR < 0) or (SatR > 1100) :
                sys.exit('Satellite distance (SatR) must be within [0, 1100] Rs') 
        else:
            sys.exit('Initial satellite distance (SatR) required to run')
        
        if 'SatRot' in input_values:
            SatRot = float(input_values['SatRot'])
            if (SatRot < -0.00417) or (SatRot > 0.00417) :
                sys.exit('Satellite distance (SatRot) must be within +/-0.00417 deg/s (1 full orbit per day)') 
        else:
            input_values['SatRot'] = str(0.0000114)
            if printNonCrit:
                print('Satellite orbital speed (SatRot) assumed same as Earth')


    # Solar Wind Parameters ----------------------------------------------------------------
    if 'SWCd' in input_values:
        SWCd = float(input_values['SWCd'])
        if (SWCd < 0) or  (SWCd > 10):
            sys.exit('Solar wind drag coefficient (SWCd) must be in [0, 10]') 
    else:
        input_values['SWCd'] = str(1) 
    
    # Get satellite radius, if near 1 AU force SW params to be reasonable values
    if 'SatR' in input_values:
        satR = float(input_values['SatR'])
    else:
        satR = 99999  
    
    # if has satPath good chance not 1 AU so don't check SW params    
    if not hasSatPath and (np.abs(satR-215) < 15.):
        if 'SWn' in input_values:
            SWn = float(input_values['SWn'])
            if (SWn < 0) or  (SWn > 50):
                sys.exit('Solar wind number density (SWn) must be in [0.1, 50] cm^-3 if satR is near 1 AU') 
        else:
            input_values['SWn'] = str(7.5)
             
        if 'SWv' in input_values:
            SWv = float(input_values['SWv'])
            if (SWv < 50) or  (SWv > 800):
                sys.exit('Solar wind radial velocty (SWv) must be in [50, 800] km/s if satR is near 1 AU') 
        else:
            input_values['SWv'] = str(350)
            
        if 'SWB' in input_values:
            SWB = float(input_values['SWB'])
            if (np.abs(SWB) < 0.1) or  (np.abs(SWB) > 50):
                sys.exit('Solar wind magnetic field strength (SWB) must be in [0.1, 50] nT if satR is near 1 AU') 
        else:
            input_values['SWB'] = str(6.9)
        
        if 'SWT' in input_values:
            SWT = float(input_values['SWT'])
            if (SWT < 1e4) or  (SWT > 5e5):
                sys.exit('Solar wind temperature (SWT) must be in [1e4, 5e5] K if satR is near 1 AU') 
        else:
            input_values['SWT'] = str(75000)
            
    if 'SWR' in input_values:
        SWR = float(input_values['SWR'])
        if (SWR < 1) or  (SWR > 1000):
            sys.exit('Distance for solar wind values (SWR) must be in [1, 10000] Rs')    
    
    # General simulation parameters ------------------------------------------------
    if 'date' in input_values:
        strlen = len(input_values['date'])
        if strlen != 8:
            sys.exit('date must be given as YYYYMMDD')
    else:
        if printNonCrit:
            print ('No date given, running in arbitrary time')

    if 'time' in input_values:
        strlen = len(input_values['time'])
        if (strlen != 5) & (input_values['time'][2] != ':'):
            sys.exit('date must be given as HH:MM (24 hr time)')
    else:
        input_values['time'] = '00:00'
        if printNonCrit:
            print ('No time given, starting at 00:00')
        
    if 'nRuns' in input_values:
        nRuns = int(input_values['nRuns'])
        if (nRuns < 0) or  (nRuns > 200):
            sys.exit('Number of simulation runs (nRuns) must be in [0, 200]') 
    else:
        input_values['nRuns'] = str(1) 
        
    # TorF only... 'FCRotCME' 'saveData', 'printData','flagScales'   'isSat',
    if 'FCRotCME' in input_values:
        FCRotCME = input_values['FCRotCME']
        if FCRotCME not in ['True', 'False']:
            sys.exit('Inclusion of rotation in ANTEATR (FCRotCME) can only be True or False') 
    
    if 'saveData' in input_values:
        saveData = input_values['saveData']
        if saveData not in ['True', 'False']:
            sys.exit('Option to save data for individual runs (saveData) can only be True or False') 

    if 'printData' in input_values:
        printData = input_values['printData']
        if printData not in ['True', 'False']:
            sys.exit('Option to print data for individual runs (printData) can only be True or False') 
                
    if 'isSat' in input_values:
        isSat = input_values['isSat']
        if isSat not in ['True', 'False']:
            sys.exit('Option to save data in satellite coords vs GSE (isSat) can only be True or False') 

    if 'L0' in input_values:
        SatLon = float(input_values['SatLon'])
        if (SatLon < -180) or  (SatLon > 360):
            sys.exit('Satellite longitude shift (L0) must be in [-180, 180] or [0, 360] deg') 
        
    if 'SunR' in input_values:
        SunR = float(input_values['SunR'])
        if (SunR < 7e8) or  (SunR > 7e12):
            sys.exit('Stellar/solar radius (SunR) must be in [7e8, 7e12] cm (0.01 Rs, 100 Rs)') 
    else:
        input_values['SunR'] = str(7e10) 

    if 'SunRotRate' in input_values:
        SunRotRate = float(input_values['SunRotRate'])
        if (np.abs(SunRotRate) < 0.00175):
            sys.exit('Stellar/solar rotation rate (SunRotRate) must in +/- 0.00175 rad/s (one hour or longer period)') 
    else:
        input_values['SunRotRate'] = str(2.8e-6) 
        
    if 'SunRss' in input_values:
        SunRss = float(input_values['SunRss'])
        if (SunRss < 1.5) or  (SunRss > 5):
            sys.exit('Source surface height (SunRss) must in [1.5, 5] Rs') 
    else:
        input_values['SunRss'] = str(2.5) 
    
    hasFRstart, hasFRend, hasShstart = False, False, False    
    if 'obsFRstart' in input_values:
        oFRs = float(input_values['obsFRstart'])
        if (oFRs < 0) or  (oFRs >= 360):
            sys.exit('Observed flux rope start (obsFRstart) must in [0, 360) fraction DoY') 
        else:
            hasFRstart = True

    if 'obsFRend' in input_values:
        oFRe = float(input_values['obsFRend'])
        if (oFRe < 0) or  (oFRe >= 360):
            sys.exit('Observed flux rope end (obsFRend) must in [0, 360) fraction DoY') 
        else:
            hasFRend = True
    
    if 'obsShstart' in input_values:
        oShs = float(input_values['obsShstart'])
        if (oShs < 0) or  (oShs >= 360):
            sys.exit('Observed shock/sheath start (obsShstart) must in [0, 360) fraction DoY') 
        else:
            hasShstart = True

    if hasFRstart and not hasFRend:
        sys.exit('Provided FRstart but not FRend. Need both or neither.')

    if hasFRend and not hasFRstart:
        sys.exit('Provided FRend but not FRstart. Need both or neither.')
            
    if hasFRstart and hasFRend:
        if float(input_values['obsFRend']) < float(input_values['obsFRstart']):
            sys.exit('Flux rope end (obsFRend) is before flux rope start (obsFRstart). Please fix')

    if hasFRstart and hasShstart:
        if float(input_values['obsFRstart']) < float(input_values['obsShstart']):
            sys.exit('Flux rope start (obsFRstart) is before sheath start (obsShstart). Please fix')
    
        
    # MEOW-HiSS parameters ----------------------------------------------------------
    doMH = False
    if 'doMH' in input_values:
        doMH = input_values['doMH']
        if doMH in ['True', 'False']:
            if doMH == 'True':
                doMH = True
            elif doMH == 'False':
                doMH = False
        else:
            sys.exit('Flag for MEOW-HiSS (doMH) must be either True or False')
    
    if doMH:
        if 'MHarea' in input_values:
            MHarea = float(input_values['MHarea'])
            if (MHarea < 50) or  (MHarea > 2500):
                sys.exit('Coronal hole area (MHarea) must be in [50, 2500] 10^8 km^2') 
        else:
            sys.exit('Coronal hole area (MHarea) must be provided if MEOW-HiSS is included. If you only know the front distance, 800 (10^8 km^2) is a good average size to start with.')

        if 'MHdist' in input_values:
            MHdist = float(input_values['MHdist'])
            if (MHdist < -0.4) or  (MHdist > 1.5):
                sys.exit('Initial HSS front distance (MHdist) must be in [-0.4, 1.5] AU') 
        else:
            sys.exit('Initial HSS front distance (MHdist) must be provided if MEOW-HiSS is included')
    
    # Yaw check
    if 'simYaw' in input_values:
        simYaw = input_values['simYaw']
        if simYaw in ['True', 'False']:
            if simYaw == 'True':
                simYaw = True
        else:
            sys.exit('Flag for simulating yaw rotation (simYaw) must be either True or False')
    
        
    # bonus check - are lons near one another    
    if not hasSatPath:
        CMElon = float(input_values['CMElon'])
        satLon = float(input_values['SatLon'])
        
        if (CMElon > 300) and (satLon < 0):
            satLon += 360.
            input_values['SatLon'] = str(satLon)
        if (CMElon < 0) and (satLon > 360):
            CMElon += 360.
            input_values['CMElon'] = str(CMElon)

def setupEns():
    # All the possible parameters one could in theory want to vary
    possible_vars =  ['CMElat', 'CMElon', 'CMEtilt', 'CMEyaw', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB', 'PFSSscale', 'IVDf1', 'IVDf2', 'IVDf', 'Gamma', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWT', 'FRB', 'FRtau', 'FRCnm', 'FRT', 'MHarea', 'MHdist']
    print( 'Determining parameters varied in ensemble...')
    EnsData = np.genfromtxt(FC.fprefix+'.ens', dtype=str, encoding='utf8')
    # Make a dictionary containing the variables and their uncertainty
    global EnsInputs
    EnsInputs = {}
    for i in range(len(EnsData)):
        #temp = EnsData[i]
        if len(EnsData) > 2:
            temp = EnsData[i]
        elif len(EnsData) == 2:
            if len(EnsData[i]) == 2: 
                temp = EnsData[i]
            else:
                temp = EnsData
        if temp[0][:-1] in possible_vars:
            EnsInputs[temp[0][:-1]] = float(temp[1])
    # Display what parameters will be varied
    EnsVarStr = ''
    for item in EnsInputs.keys():
        EnsVarStr += item +', '
    print( 'Ensembles will vary '+EnsVarStr[:-2] )
    # Convert the uncertainties into sigmas for Gaussian distributions
    # 3 sigma = 99.7 percent of distribution
    for item in EnsInputs.keys(): EnsInputs[item] = EnsInputs[item]/3.
    # Open up a file for the tracking of ensemble member parameters
    global ensembleFile, orderedEnsKeys
    ensembleFile = open(Dir+'/EnsembleParams'+thisName+'.dat', 'w')
    # Add a header and the seed values
    # Have to do ForeCAT values first then non ForeCAT based on how 
    # genEnsMem runs.  
    # nSW and vSW variations are only captured in ANTEATR output
    outstr1 = 'RunID '
    outstr2 = '0000 '
    orderedEnsKeys = []
    for item in EnsInputs.keys():
        if item not in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWT', 'FRB', 'FRT', 'IVDf1', 'IVDf2', 'IVFf', 'Gamma', 'MHarea', 'MHdist']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
            orderedEnsKeys.append(item)
    for item in EnsInputs.keys():
        if item in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWT',  'FRB', 'FRT', 'IVDf1', 'IVDf2', 'IVFf', 'Gamma', 'MHarea', 'MHdist']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
            orderedEnsKeys.append(item)
    orderedEnsKeys = np.array(orderedEnsKeys)
    ensembleFile.write(outstr1+'\n')
    ensembleFile.write(outstr2+'\n')
            
def genEnsMem(runnum=0):
    # Vary parameters for all models at the same time
    outstr = str(runnum)
    outstr = outstr.zfill(4) + '   '
    flagAccel = False
    flagExp   = False
    new_pos = [float(input_values['CMElat']), float(input_values['CMElon']), float(input_values['CMEtilt'])]
    for item in orderedEnsKeys:
        # Sort out what variable we adjust for each param
        # The lambda functions will auto adjust to new global values in them
        if item == 'CMElat':
            new_pos[0] = np.random.normal(loc=float(input_values['CMElat']), scale=EnsInputs['CMElat'])
            if new_pos[0] >= 90:
                new_pos[0] = 89.99
            elif new_pos[0] <= -90:
                new_pos[0] = -89.99
            outstr += '{:4.2f}'.format(new_pos[0]) + ' '
        if item == 'CMElon':
            new_pos[1] = np.random.normal(loc=float(input_values['CMElon']), scale=EnsInputs['CMElon'])
            outstr += '{:4.2f}'.format(new_pos[1]) + ' '
        if item == 'CMEtilt':
            new_pos[2] = np.random.normal(loc=float(input_values['CMEtilt']), scale=EnsInputs['CMEtilt'])
            outstr += '{:4.2f}'.format(new_pos[2]) + ' '
        if item == 'SWCdp':
            FC.Cd = np.random.normal(loc=float(input_values['SWCdp']), scale=EnsInputs['SWCdp'])
            # Make sure drag coeff is positive if inappropriate uncertainty is used
            if FC.Cd < 0: FC.Cd = 0
            outstr += '{:4.3f}'.format(FC.Cd) + ' '           
        if item == 'CMEdelAx':
            FC.deltaAx = np.random.normal(loc=float(input_values['CMEdelAx']), scale=EnsInputs['CMEdelAx'])
            outstr += '{:5.4f}'.format(FC.deltaAx) + ' '
        if item == 'CMEdelCS':
            FC.deltaCS = np.random.normal(loc=float(input_values['CMEdelCS']), scale=EnsInputs['CMEdelCS'])
            outstr += '{:5.4f}'.format(FC.deltaCS) + ' '
        if item == 'CMEr':
            FC.rstart = np.random.normal(loc=float(input_values['CMEr']), scale=EnsInputs['CMEr'])
            outstr += '{:5.4f}'.format(FC.rstart) + ' '
        if item == 'FCraccel1':
            FC.rga = np.random.normal(loc=float(input_values['FCraccel1']), scale=EnsInputs['FCraccel1'])
            outstr += '{:5.4f}'.format(FC.rga) + ' '
            flagAccel = True
        if item == 'FCraccel2':
            FC.rap = np.random.normal(loc=float(input_values['FCraccel2']), scale=EnsInputs['FCraccel2'])
            outstr += '{:5.4f}'.format(FC.rap) + ' '
            flagAccel = True
        if item == 'FCvrmin':
            FC.vmin = np.random.normal(loc=float(input_values['FCvrmin']), scale=EnsInputs['FCvrmin'])*1e5
            outstr += '{:5.2f}'.format(FC.vmin/1e5) + ' '
            flagAccel = True
        if item == 'CMEvr':
            FC.vmax = np.random.normal(loc=float(input_values['CMEvr']), scale=EnsInputs['CMEvr'])*1e5
            outstr += '{:6.2f}'.format(FC.vmax/1e5) + ' '
            flagAccel = True
        if item == 'FCAWr':
            FC.awR = np.random.normal(loc=float(input_values['FCAWr']), scale=EnsInputs['FCAWr'])
            outstr += '{:4.3f}'.format(FC.awR) + ' '
            flagExp = True
        if item == 'FCAWmin':
            FC.aw0 = np.random.normal(loc=float(input_values['FCAWmin']), scale=EnsInputs['FCAWmin'])
            outstr += '{:5.2f}'.format(newaw0) + ' '
            flagExp = True
        if item == 'CMEAW':
            FC.awM = np.random.normal(loc=float(input_values[item]), scale=EnsInputs[item])
            outstr += '{:5.2f}'.format(FC.awM) + ' '
            flagExp = True
        if item == 'CMEAWp':
            FC.AWp = np.random.normal(loc=float(input_values[item]), scale=EnsInputs[item])
            outstr += '{:5.2f}'.format(FC.AWp) + ' '
            flagExp = True
            FC.AWratio = lambda R_nose: (FC.AWp/FC.awM)*(1. - np.exp(-(R_nose-1.)/FC.awR))
        if item == 'FCrmaxM':
            FC.rmaxM = np.random.normal(loc=float(input_values['FCrmaxM']), scale=EnsInputs['FCrmaxM'])
            outstr += '{:5.2f}'.format(FC.rmaxM) + ' '
        if item == 'CMEM':
            FC.max_M = np.random.normal(loc=float(input_values['CMEM']), scale=EnsInputs['CMEM'])* 1e15
            # Set somewhat arbitrary lower bound, definitely don't want negative mass
            if FC.max_M < 1e14: FC.max_M = 1e14
            outstr += '{:5.2f}'.format(FC.max_M/1e15) + ' '        
        if item == 'PFSSscale':
            FC.PFSSscale = np.random.normal(loc=float(input_values['PFSSscale']), scale=EnsInputs['PFSSscale'])
            if FC.PFSSscale < 0: FC.PFSSscale = 0.01 # set minimum
            outstr += '{:5.2f}'.format(FC.PFSSscale) + ' '
    
    if flagAccel:
        FC.a_prop = (FC.vmax**2 - FC.vmin **2) / 2. / (FC.rap - FC.rga)
    if flagExp:
        FC.user_exp = lambda R_nose: FC.aw0 + (FC.awM-FC.aw0)*(1. - np.exp(-(R_nose-1.)/FC.awR))
        FC.AWratio = lambda R_nose: (FC.AWp/FC.awM)*(1. - np.exp(-(R_nose-1.)/FC.awR))
            
            
    CME = initCME([FC.deltaAx, FC.deltaCS, FC.rstart], new_pos)

    # add non ForeCAT vars
    if 'SWCd' in input_values: CME.Cd = float(input_values['SWCd'])
    if 'SWn' in input_values: CME.nSW = float(input_values['SWn'])    
    if 'SWv' in input_values: CME.vSW = float(input_values['SWv'])
    if 'SWB' in input_values: CME.BSW = float(input_values['SWB'])    
    if 'SWT' in input_values: CME.TSW = float(input_values['SWT'])    
    if 'FRtau'  in input_values: CME.tau = float(input_values['FRtau'])
    if 'FRCnm'  in input_values: CME.cnm = float(input_values['FRCnm'])
    if 'FRB' in input_values: CME.FRBtor = float(input_values['FRB'])    
    if 'FRT' in input_values: CME.FRT = float(input_values['FRT'])
    if 'IVDf1' in input_values: CME.IVDfs[0] = float(input_values['IVDf1'])
    if 'IVDf2' in input_values: CME.IVDfs[1] = float(input_values['IVDf2'])
    if 'IVDf' in input_values: 
        CME.IVDfs[0] = float(input_values['IVDf'])
        CME.IVDfs[1] = float(input_values['IVDf'])
    if 'Gamma' in input_values: CME.gamma = float(input_values['Gamma'])
    if 'MHdist' in input_values: CME.MHdist = float(input_values['MHdist'])    
    if 'MHarea' in input_values: CME.MHarea = float(input_values['MHarea'])    
    if 'CMEyaw' in input_values: CME.yaw = float(input_values['CMEyaw'])    
    
    # add changes to non ForeCAT things onto the CME object
    for item in orderedEnsKeys:
        if item == 'SWCd':
            CME.Cd = np.random.normal(loc=float(input_values['SWCd']), scale=EnsInputs['SWCd'])
            if CME.Cd <0: CME.Cd = 0
            outstr += '{:5.3f}'.format(CME.Cd) + ' '
        if item == 'SWn':
            CME.nSW = np.random.normal(loc=float(input_values['SWn']), scale=EnsInputs['SWn'])
            if CME.nSW < 0: CME.nSW = 0
            outstr += '{:6.2f}'.format(CME.nSW) + ' '
        if item == 'SWv':
            CME.vSW = np.random.normal(loc=float(input_values['SWv']), scale=EnsInputs['SWv'])
            outstr += '{:6.2f}'.format(CME.vSW) + ' '
        if item == 'SWB':
            CME.BSW = np.random.normal(loc=float(input_values['SWB']), scale=EnsInputs['SWB'])
            outstr += '{:6.2f}'.format(CME.BSW) + ' '
        if item == 'SWT':
            CME.TSW = np.random.normal(loc=float(input_values['SWT']), scale=EnsInputs['SWT'])
            outstr += '{:8.0f}'.format(CME.TSW) + ' '
        if item == 'FRB':
            CME.FRBtor = np.random.normal(loc=float(input_values['FRB']), scale=EnsInputs['FRB'])
            outstr += '{:5.2f}'.format(CME.FRBtor) + ' '
        if item == 'FRtau':
            CME.tau = np.random.normal(loc=float(input_values['FRtau']), scale=EnsInputs['FRtau'])
            outstr += '{:5.2f}'.format(CME.tau) + ' '
        if item == 'FRCnm':
            CME.cnm = np.random.normal(loc=float(input_values['FRCnm']), scale=EnsInputs['FRCnm'])
            outstr += '{:5.2f}'.format(CME.cnm) + ' '
        if item == 'FRT':
            CME.FRT = np.random.normal(loc=float(input_values['FRT']), scale=EnsInputs['FRT'])
            outstr += '{:5.2f}'.format(CME.FRT) + ' '
        if item == 'IVDf1':
            CME.IVDfs[0] = np.random.normal(loc=float(input_values['IVDf1']), scale=EnsInputs['IVDf1'])
            # force f2 to match f1 unless reset by explicitly including IVDf2 in ensemble
            if 'IVDf2' not in EnsInputs.keys():
                CME.IVDfs[1] = CME.IVDfs[0]
            outstr += '{:6.2f}'.format(CME.IVDfs[0]) + ' '
        if item == 'IVDf2':
            CME.IVDfs[1] = np.random.normal(loc=float(input_values['IVDf2']), scale=EnsInputs['IVDf2'])
            outstr += '{:6.2f}'.format(CME.IVDfs[1]) + ' '
        if item == 'IVDf':
            CME.IVDfs[0] = np.random.normal(loc=float(input_values['IVDf']), scale=EnsInputs['IVDf'])
            CME.IVDfs[1] = CME.IVDfs[0]
            outstr += '{:6.2f}'.format(CME.IVDfs[0]) + ' '
        if item == 'Gamma':
            CME.gamma = np.random.normal(loc=float(input_values['Gamma']), scale=EnsInputs['Gamma'])
            outstr += '{:6.2f}'.format(CME.gamma) + ' '
        if item == 'MHdist':
            CME.MHdist = np.random.normal(loc=float(input_values['MHdist']), scale=EnsInputs['MHdist'])
            outstr += '{:6.3f}'.format(CME.MHdist) + ' '
        if item == 'MHarea':
            CME.MHarea = np.random.normal(loc=float(input_values['MHarea']), scale=EnsInputs['MHarea'])
            outstr += '{:6.2f}'.format(CME.MHarea) + ' '  
        if item == 'CMEyaw':
            CME.yaw = np.random.normal(loc=float(input_values['CMEyaw']), scale=EnsInputs['CMEyaw'])
            outstr += '{:6.2f}'.format(CME.yaw) + ' '  
                  
    ensembleFile.write(outstr+'\n')  
    # used to set CME.vs here, seems unnecessary b/c done in other parts of code       
    return CME
                        
def makeSatPaths(fname, tzero, Clon0=0):
    # tzero is a date time object for whenever 0 simulation time is
    # Clon0 is the Carrington longitude at time 0 
    # it allows us to given lons in reference to the CME lon which is given
    # with respect to the noninertial Carrington coord system at t0
    # (basically a way to shift the inertial frame in lon)
    data = np.genfromtxt(fname, dtype=str, encoding='utf8')
    delts = []
    for i in range(len(data)):
        thisdatestr = str(data[i][0])+' '+str(data[i][1])
        if len(thisdatestr) == 19:
            thisdate = datetime.datetime.strptime(thisdatestr, '%Y-%m-%d %H:%M:%S')
        elif len(thisdatestr) == 16:
            thisdate = datetime.datetime.strptime(thisdatestr, '%Y-%m-%d %H:%M')
        delts.append((thisdate - tzero).total_seconds())
    satrs   = np.array(data[:,2]).astype(float) * 1.49e13 / rsun # convert to cm then rsun
    satlats = np.array(data[:,3]).astype(float)
    satlons = np.array(data[:,4]).astype(float) % 360.
    idx = np.where(np.abs(delts) == np.min(np.abs(delts)))[0]
    lon0 = satlons[idx[0]]
    satlons -= lon0 # sat lons now shifted to movement in lon from t0
    satlons += Clon0 # sat lons now relative to Clon at t0
    fR = CubicSpline(delts, satrs)
    fLat = CubicSpline(delts, satlats)
    fLon = CubicSpline(delts, satlons)
    # return functions that are R, lat, lon as a function of seconds since start of sim
    return fR, fLat, fLon
    
def goForeCAT(makeRestart=False):
    # ForeCAT portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|

    # Open a file to save the ForeCAT output
    ForeCATfile = open(Dir+'/ForeCATresults'+thisName+'.dat', 'w')

    # init ForeCAT gives input params
    # CME_params = [shapeA, shapeB, rstart]
    # init_pos = [ilat, ilon, tilt]    
    global iparams, ipos
    ipos, rmax = initForeCAT(input_values, magpath, pickleName)

    # option to force PFSS so that it will scale as R^2 from 2.5 to satPos
    if False:
        if ('SWB' in input_values) & (('PFSSscale' not in input_values)):
            latID = int(satPos[0]*2 + 179)
            lonID = int(satPos[1]*2)%720
            fullBvec = ForceFields.B_high[-1,latID,lonID][3]
            fullBvec = np.mean(np.abs(ForceFields.B_high[-1,latID-10:latID+11,lonID-20:lonID+21][3]))
            BSS = fullBvec * (2.5/satPos[2])**2 *1e5
            FC.PFSSscale = float(input_values['SWB']) / BSS  
            print (FC.PFSSscale)
        
    CME = initCME([FC.deltaAx, FC.deltaCS, FC.rstart], ipos)
    # add any ANTEATR/FIDO params to the seed case (genEnsMem will add for other cases)
    if 'SWn' in input_values: CME.nSW = float(input_values['SWn'])
    if 'SWv' in input_values: CME.vSW = float(input_values['SWv'])
    if 'SWB' in input_values: CME.BSW = float(input_values['SWB'])
    if 'SWT' in input_values: CME.TSW = float(input_values['SWT'])
    if 'SWCd' in input_values: CME.Cd = float(input_values['SWCd'])
    if 'FRB'  in input_values: CME.FRBtor = float(input_values['FRB'])
    if 'FRtau'  in input_values: CME.tau = float(input_values['FRtau'])
    if 'FRCnm'  in input_values: CME.cnm = float(input_values['FRCnm'])
    if 'FRT'  in input_values: CME.FRT = float(input_values['FRT'])
    if 'IVDf1'  in input_values: CME.IVDfs[0] = float(input_values['IVDf1'])
    if 'IVDf2'  in input_values: CME.IVDfs[1] = float(input_values['IVDf2'])
    if 'Gamma'  in input_values: CME.gamma = float(input_values['Gamma'])
    if 'MHarea'    in input_values: CME.MHarea = float(input_values['MHarea'])
    if 'MHdist'    in input_values: CME.MHdist = float(input_values['MHdist'])
    
    
    for i in range(nRuns):
        
        print('Running ForeCAT simulation '+str(i+1)+' of '+str(nRuns))
    
        # Make a new ensemble member
        if i > 0:
            CME = genEnsMem(runnum = i)
            if ('SWB' in input_values) & (('PFSSscale' not in input_values)):
                FC.PFSSscale = 1.
                latID = int(satPos[0]*2 + 179)
                lonID = int(satPos[1]*2)%720
                fullBvec = ForceFields.B_high[-1,latID,lonID]
                BSS = fullBvec[3] * (2.5/satPos[2])**2 *1e5
                FC.PFSSscale = CME.BSW / BSS

        # Run ForeCAT
        CME, path = runForeCAT(CME, rmax, path=True, silent=allSilent)
        
        # add a few useful 1 AU SW parameters to the CME object
        if useFCSW:
            HCSdist = calc_dist(CME.cone[1,1],CME.cone[1,2])
            SWnv = calc_SW(213,HCSdist)
            nSW, vSW = SWnv[0]/1.6727e-24, SWnv[1]/1e5
            CME.nSW = nSW 
            CME.vSW = vSW 
            # get B
            latID = int(CME.cone[1,1]*2 + 179)
            lonID = int(CME.cone[1,2]*2)%720
            fullBvec = FC.PFSSscale*ForceFields.B_high[-1,latID,lonID]
            inorout = np.sign(np.dot(CME.rhat,fullBvec[:3])/fullBvec[3])
            CME.BSW = inorout*fullBvec[3] * (2.5/213)**2 *1e5
            # calc alfven speed
            CME.vA = np.abs(CME.BSW/1e5) / np.sqrt(4*3.14159 * CME.nSW * 1.67e-24) / 1e5
        # Save the CME in the array
        CMEarray.append(CME)
        # Write the simulation results to a file
        for j in range(len(path[0,:])):
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = [path[0,j], path[1,j], path[2,j], path[3,j], path[4,j], path[5,j], path[6,j], path[7,j],  path[8,j],  path[9,j],  path[10,j],  path[11,j],  path[12,j],  path[13,j], path[14,j], path[15,j], path[16,j], path[17,j]]
            for iii in outstuff:
                outprint = outprint +'{:4.3f}'.format(iii) + ' '
            ForeCATfile.write(outprint+'\n')   
        
        # write a file to restart ANTEATR/FIDO from current CME values
        if makeRestart:      
            # update current inps with things that have changed in ForeCAT to pass to
            currentInps['CMEr']   = path[1,-1]
            currentInps['CMElat'] = CME.cone[1,1]
            currentInps['CMElon'] = CME.cone[1,2]
            currentInps['CMEtilt'] = CME.tilt
            currentInps['CMEAW']  = path[5,-1]
            currentInps['CMEAWp'] = path[6,-1]
            currentInps['SatLon'] = satPos[1]+Sat_rot*CME.t
            nowTime = dObj + datetime.timedelta(seconds=CME.t*60)
            currentInps['date'] = nowTime.strftime('%Y')+nowTime.strftime('%m')+nowTime.strftime('%d')
            currentInps['time'] = nowTime.strftime("%H:%M")
            # SW params if using those...
            genNXTtxt(CME, num=str(i), tag='IP')
               
    ForeCATfile.close()
    
def makeCMEarray():
    # Called if FC not ran
    global ipos
    if models in ['FIDO', 'ANT', 'IP']:
        ipos, rmax = initForeCAT(input_values, magpath, None, skipPkl=True)  
        rmax = float(input_values['CMEr'])
    else:
        ipos, rmax = initForeCAT(input_values)
    # wipe the FC functions to constants, assume we're above height of variations
    FC.user_mass = lambda x: float(input_values['CMEM']) * 1e15
    FC.user_exp  = lambda x: float(input_values['CMEAW'])
    FC.AWratio   = lambda x: float(input_values['CMEAWp']) / float(input_values['CMEAW'])
    # initiate the CME
    CME = initCME([FC.deltaAx, FC.deltaCS,  FC.rstart], ipos)
    # add non ForeCAT vars
    if 'SWCd' in input_values: CME.Cd = float(input_values['SWCd'])
    if 'FRB'  in input_values: CME.FRBtor = float(input_values['FRB'])
    if 'FRtau'  in input_values: CME.tau = float(input_values['FRtau'])
    if 'FRCnm'  in input_values: CME.cnm = float(input_values['FRCnm'])
    if 'FRT'  in input_values: CME.FRT = float(input_values['FRT'])
    if 'IVDf1'  in input_values: CME.IVDfs[0] = float(input_values['IVDf1'])
    if 'IVDf2'  in input_values: CME.IVDfs[1] = float(input_values['IVDf2'])
    if 'IVDf'  in input_values: 
            CME.IVDfs[0] = float(input_values['IVDf'])
            CME.IVDfs[1] = float(input_values['IVDf'])
    if 'Gamma'  in input_values: CME.gamma = float(input_values['Gamma'])
    if 'SWn'    in input_values: CME.nSW = float(input_values['SWn'])
    if 'SWv'    in input_values: CME.vSW = float(input_values['SWv'])
    if 'SWB'    in input_values: CME.BSW = float(input_values['SWB'])
    if 'SWT'    in input_values: CME.TSW = float(input_values['SWT'])
    if 'MHarea'    in input_values: CME.MHarea = float(input_values['MHarea'])
    if 'MHdist'    in input_values: CME.MHdist = float(input_values['MHdist'])
    if 'CMEyaw'    in input_values: CME.yaw = float(input_values['CMEyaw'])
    
    # Move to end of ForeCAT distance    
    CME = move2corona(CME, rmax)
         
    global CMEarray
    CMEarray = [CME]
    
    # Repeat process with ensemble variation
    for i in range(nRuns-1):
        CME = genEnsMem(runnum=i+1)
        CME = move2corona(CME, rmax)
        CMEarray.append(CME)
        
def move2corona(CME, rmax):
    # Need to take the CMEs which are generated in low corona for ForeCAT and
    # move to outer corona for ANTEATR/FIDO
    # Pull in values
    CME.points[CC.idcent][1,0] = rmax
    CME.AW = FC.user_exp(rmax) * dtor
    CME.AWp = FC.AWratio(rmax) * CME.AW
    CME.calc_lens()

    # set vr, don't really care about correct XYZ orientation, ANTEATR just needs magnitude
    CME.vels[0,:] = CME.vels[0,:]*0
    CME.vels[0,0] = float(input_values['CMEvr']) * 1e5
	# determine new mass
    CME.M = float(input_values['CMEM'])*1e15

     # determine new position of grid points with updated ang_width and cone pos
    CME.calc_points()
    return CME
    
def goANTEATR(makeRestart=False, satPathIn=False):
    # ANTEATR portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|
    
    # Sort out if doing one satellite or multi and separate names if needed
    if satPathIn:
        satNames = np.array(satPathIn[-1])
        satPath  = satPathIn[:-1]
    else:
        satPath = False
        satNames = np.array(['sat1'])
    nSats = len(satNames)
    

    # Open a file to save the ANTEATR output
    ANTEATRfile = open(Dir+'/ANTEATRresults'+thisName+'.dat', 'w')
    if doPUP:
        PUPfile = open(Dir+'/PUPresults'+thisName+'.dat', 'w')
    SITfiles = []    
    FIDOfiles = []
    # Will either open one for each given name or a single one if don't have sat names    
    for satName in satNames:
        # reset the generic 'sat1' to nothing if only doing 1 case with temp name
        if (len(satNames) == 1) and (satName == 'sat1'):
            satName = ''
        if doPUP:
            SITfile = open(Dir+'/SITresults'+thisName+satName+'.dat', 'w')
            SITfiles.append(SITfile)
        if doFIDO:
            FIDOfile = open(Dir+'/FIDOresults'+thisName+satName+'.dat', 'w')
            FIDOfiles.append(FIDOfile)
    SITfiles = np.array(SITfiles)
    FIDOfiles = np.array(FIDOfiles)

    # ANTEATR takes inputs
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, FRB, nSW, vSW, BSW, Cd, tau, cnm]         <-this is outdated
    # SatVars0 = [Satlat, Satlon, Satradius] -> technically doesn't have to be Earth!
    # which we can generate from the ForeCAT data

    # Pull in ANTEATR values from the input file
    global SatVars0
    SatVars0, Cd, mySW = processANTinputs(input_values, hasPath=satPath)

    # Assume that the given lon is the time of eruption, need to add in
    # the orbit during the ForeCAT run
    SatRotRate = SatVars0[3]
    SatLons = [SatVars0[1]+SatRotRate*CMEarray[i].t for i in range(nRuns)]

    global impactIDs, SWv, SWn, ANTCMErs
    ANTCMErs = {}
    impactIDs = []
    
    actualPath = False
    # Check if we're using satPath to pass multi simple satellites
    if satPath:
        try:
            a = float(satPath[0][0])
        except:
            actualPath = True
    
    # Loop over all the CMEs    
    for i in range(nRuns):
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr0, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        
        # reset path functions for t0 at start of ANTEATR
        # need to do for each satellite if > 1
        if satPath:           
            if actualPath:
                satfs = [[] for i in range(len(satPath))]
                satfs[0] = [lambda x: satPath[0][0](x + CME.t*60), lambda x: satPath[0][1](x + CME.t*60), lambda x: satPath[0][2](x + CME.t*60)]
                if nSats >= 2:
                    satfs[1] = [lambda x: satPath[1][0](x + CME.t*60), lambda x: satPath[1][1](x + CME.t*60), lambda x: satPath[1][2](x + CME.t*60)] 
                if nSats >= 3:
                    satfs[2] = [lambda x: satPath[2][0](x + CME.t*60), lambda x: satPath[2][1](x + CME.t*60), lambda x: satPath[2][2](x + CME.t*60)]
                if nSats >= 4:
                    satfs[3] = [lambda x: satPath[3][0](x + CME.t*60), lambda x: satPath[3][1](x + CME.t*60), lambda x: satPath[3][2](x + CME.t*60)]
                if nSats >= 5:
                    satfs[4] = [lambda x: satPath[4][0](x + CME.t*60), lambda x: satPath[4][1](x + CME.t*60), lambda x: satPath[4][2](x + CME.t*60)]
                if nSats >= 6:
                    satfs[5] = [lambda x: satPath[5][0](x + CME.t*60), lambda x: satPath[5][1](x + CME.t*60), lambda x: satPath[5][2](x + CME.t*60)]
                if nSats >= 7:
                    satfs[6] = [lambda x: satPath[6][0](x + CME.t*60), lambda x: satPath[6][1](x + CME.t*60), lambda x: satPath[6][2](x + CME.t*60)]
                if nSats >= 8:
                    satfs[7] = [lambda x: satPath[7][0](x + CME.t*60), lambda x: satPath[7][1](x + CME.t*60), lambda x: satPath[7][2](x + CME.t*60)]
                if nSats >= 9:
                    satfs[8] = [lambda x: satPath[8][0](x + CME.t*60), lambda x: satPath[8][1](x + CME.t*60), lambda x: satPath[8][2](x + CME.t*60)]
                if nSats >= 10:
                    satfs[9] = [lambda x: satPath[9][0](x + CME.t*60), lambda x: satPath[9][1](x + CME.t*60), lambda x: satPath[9][2](x + CME.t*60)]
            
            
                # Only give it the names if 2 or more satellites  
                if nSats >= 2:             
                    satfs.append(satNames)
     
        # Set up initial satellite(s) position
        # Given vals are at OSPREI time 0, possibly need to include ForeCAT time now
        if satPath:
            if actualPath:
                myParams = []
                for ii in range(len(satPath)):
                    thisParam = [satfs[ii][0](CMEarray[i].t*60), satfs[ii][1](CMEarray[i].t*60), satfs[ii][2](CMEarray[i].t*60)*7e10,0]
                    myParams.append(thisParam)
                myParams.append(satNames)
            else:
                myParams = []
                for ii in range(len(satPath)):
                     thisParam = np.copy(satPath[ii])
                     thisParam[1] = thisParam[1]+thisParam[3]*CMEarray[i].t 
                     thisParam[2] = thisParam[2] * 7e10
                     myParams.append(thisParam)
                myParams.append(satNames)
        else:
            myParams = SatVars0
            myParams[1] = SatLons[i]
            myParams = [[myParams[0], myParams[1], myParams[2]*7e10, myParams[3]], ['sat1']]

        # Pull in this CME's parameters for input array
        # Tilt 
        tilt = CME.tilt
        # Calculate vr
        vr = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
        # Mass
        mass = CME.M/1e15
        # CME shape
        cmeAW = CME.AW*radeg
        cmeAWp = CME.AWp*radeg
        deltax = CME.deltaAx
        deltap = CME.deltaCS
        deltaCA = CME.deltaCSAx
        IVDfs = CME.IVDfs
        

        # Add in ensemble variation if desired
        if (i > 0) and useFCSW:
            if 'SWn' in EnsInputs: CME.nSW = np.random.normal(loc=CME.nSW, scale=EnsInputs['SWn'])
            if 'SWv' in EnsInputs: CME.vSW = np.random.normal(loc=CME.vSW, scale=EnsInputs['SWv'])
            if 'SWB' in EnsInputs: CME.BSW = np.random.normal(loc=CME.BSW, scale=EnsInputs['SWB'])              
            if 'SWT' in EnsInputs: CME.TSW = np.random.normal(loc=CME.TSW, scale=EnsInputs['SWT'])              
        FRB, tau, cnm, FRT = CME.FRBtor, CME.tau, CME.cnm, CME.FRT
        
        # Option to recalc B if running ensembles that vary AW and we are using that to calc them
        if False:
            rFront = CMEr0 * 7e10
            rCSp = np.tan(cmeAWp*dtor) / (1 + deltap * np.tan(cmeAWp*dtor)) * rFront
            rCSr = deltap * rCSp
            Lp = (np.tan(cmeAW*dtor) * (rFront - rCSr) - rCSr) / (1 + deltax * np.tan(cmeAW*dtor))  
            Ltorus = lenFun(deltax) * Lp
    
            # Ltorus needs to include legs
            rCent = rFront - deltax*Lp - rCSp
            Lleg = np.sqrt(Lp**2 + rCent**2) - 7e10 # dist from surface
            Ltot = Ltorus + 2 * Lleg 
            avgR = (0.5*rCSp * Lleg * 2 + rCSp * Ltorus) / (Lleg*2 + Ltorus)
    
            # B from KE
            KE = 0.5 * mass*1e15 * (vr*1e5)**2 /1e31
            phiflux = np.power(10, np.log10(KE / 0.19) / 1.87)*1e21
            B0 = phiflux * cnm * (deltap**2 + 1) / avgR / Ltot / deltap**2 *1e5
            FRB = deltap * tau * B0
        
        # Package up invec, run ANTEATR        
        gamma = CME.gamma
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, FRB, Cd, tau, cnm, FRT, gamma, CME.yaw]
        SWvec = [CME.nSW, CME.vSW, np.abs(CME.BSW), CME.TSW]

        # check if given SW 1D profiles
        if flag1DSW:
            SWvec = SWfile
        MHin = False
        if doMH:
            MHin = [CME.MHarea, CME.MHdist]
            
        # check if given distance of SW measurements, otherwise ANT will assume first sat loc
        SWR = None
        if 'SWR' in input_values:
            SWR = float(input_values['SWR'])    
        
        # SW polarity in or out - THINK THIS WAS FLIPPED FOR ISSI/PROPOSAL CASE?
        inorout = np.sign(CME.BSW) 
        # high fscales = more convective like
        isSilent = allSilent
        
        if actualPath:      
            ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, satfs=satfs, flagScales=flagScales, doPUP=doPUP, MEOWHiSS=MHin, aFIDOinside=doFIDO, inorout=inorout, simYaw=simYaw, SWR=SWR, CMEH=FRpol)
        else:
            ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, flagScales=flagScales, doPUP=doPUP, MEOWHiSS=MHin, aFIDOinside=doFIDO, inorout=inorout, simYaw=simYaw, SWR=SWR, CMEH=FRpol, sheathParams=sheathParams)
            
        # Check if miss or hit  
        # Need to account for mix of hits and misses with multi TO DO!!
        if ATresults[0][0] not in [9999, 8888]:
            impactIDs.append(i)
            if (1 in PUPresults[11]):
                CME.hasSheath = True
                
            for sat in satNames:
                thisSum = outSum[sat]
                pidx = np.where(satNames == sat)[0]
                # check if input for this sat
                if thisSum[-1] != -9999:
                    thisvs = vsArr[sat]
                    FRhit = thisSum[0] / 24.
                    FRidx = np.min(np.where(ATresults[0] >= FRhit))
                    thisParams = myParams[pidx[0]]
                    thisAngs = angArr[sat]
                    # get improved v estimates using FIDO code
                    vFvec, vExvec = getvCMEframe(1., thisAngs[0], thisAngs[1], ATresults[5][FRidx], ATresults[6][FRidx], vsArr[sat])
                    temp0 = roty(vFvec, -CME.yaw)
                    temp = rotx(temp0, -(90.-tilt))
                    temp2 = roty(temp, CMElat - thisParams[0]) 
                    vInSitu = rotz(temp2, CMElon - thisParams[1])
                    vF = vInSitu[0] / 1e5
                    
                    # this is vExp for axis and cs not the same as measured vexp....
                    temp0 = roty(vExvec, -CME.yaw)
                    temp = rotx(temp0, -(90.-tilt))
                    temp2 = roty(temp, CMElat - thisParams[0]) 
                    vExVec = rotz(temp2, CMElon - thisParams[1])
                    vEx = vExVec[0] / 1e5
            
                    # Can add in ForeCAT time to ANTEATR time
                    FCATtime = CME.t/60./24
                    # Now want time of FR first contact, which is not idx -1 if do fullContact/FIDO
                    hitIdx = -1
                    if doFIDO:
                        regs = FIDOresults[sat][7]
                        hitTime = FIDOresults[sat][0][np.min(np.where(regs == 1))]/24
                        hitIdx = np.min(np.where(ATresults[0] >= hitTime))
                    TotTime  = ATresults[0][hitIdx]+FCATtime
                    rCME     = ATresults[1][hitIdx]
                    CMEAW    = ATresults[3][hitIdx]
                    CMEn     = ATresults[10][hitIdx]
                    logT     = ATresults[11][hitIdx]
                    
                    if not allSilent:
                        print (str(i)+' Contact after '+"{:.2f}".format(TotTime)+' days with front velocity '+"{:.2f}".format(vF)+' km/s (expansion velocity ' +"{:.2f}".format(vEx)+' km/s) when nose reaches '+"{:.2f}".format(rCME) + ' Rsun and angular width '+"{:.0f}".format(CMEAW)+' deg and estimated duration '+"{:.0f}".format(thisSum[5])+' hr')
                    
                    if not allSilent:
                        if not noDate:
                            dImp = dObj + datetime.timedelta(days=TotTime)
                            print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
                        print ('   Density: ', CMEn, '  Temp:  ', np.power(10,logT))
                        
                # Only sheath impact
                elif thisSum[0] != -9999:
                    # Can add in ForeCAT time to ANTEATR time
                    FCATtime = CME.t/60./24
                    regs = FIDOresults[sat][7]
                    hitTime = FIDOresults[sat][0][np.min(np.where(regs == 0))]/24 
                    hitIdx = np.min(np.where(ATresults[0] >= hitTime))
                    endTime = FIDOresults[sat][0][np.max(np.where(regs == 0))]/24 
                    ShDur = 24 * (endTime - hitTime)
                    if not allSilent:
                        print (str(i)+' Contact with sheath after '+"{:.2f}".format(hitTime+ FCATtime)+' days. No FR impact. Sheath duration of ' + '{:.2f}'.format(ShDur) + ' hr.')
                        if not noDate:
                            dImp = dObj + datetime.timedelta(days=hitTime)
                            print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
                            
                # Should work for either just sheath or full impact
                if thisSum[0] != -9999:                          
                    # Record FIDO/PUP results specific to each satellite
                    if doFIDO:
                        # Save FIDO profiles
                        theseFIDOres = FIDOresults[sat]
                        for j in range(len(theseFIDOres[0])):
                            outprint = str(i)
                            outprint = outprint.zfill(4) + '   '
                            Btot = np.sqrt(theseFIDOres[1][j]**2 + theseFIDOres[2][j]**2 + theseFIDOres[3][j]**2)
                            if isSat:
                                outstuff = [theseFIDOres[0][j]/24+FCATtime, Btot, theseFIDOres[1][j], theseFIDOres[2][j], theseFIDOres[3][j], theseFIDOres[4][j], theseFIDOres[5][j], theseFIDOres[6][j], theseFIDOres[7][j]]
                            else:
                                # GSE coords, flip R,T to xy
                                outstuff = [theseFIDOres[0][j]/24+FCATtime, Btot, -theseFIDOres[1][j], -theseFIDOres[2][j], theseFIDOres[3][j], theseFIDOres[4][j], theseFIDOres[5][j], theseFIDOres[6][j], theseFIDOres[7][j]]    
                            for iii in outstuff:
                                outprint = outprint +'{:6.3f}'.format(iii) + ' '
                            FIDOfiles[pidx[0]].write(outprint+'\n')
                
                    if doPUP:
                        # Save sheath/shock properties at first contact
                        #outstuff = id [dur, comp, Mach, n, vsheath, B, vshock] + 
                        # PUP 0 vShock, 1 r, 2 Ma, 3 wid, 4 dur, 5 mass, 6 dens 7 temp 8 theta 9 B 10 vt 11 init
                        # get idx appropriate for each CME
                        shIdx = hitIdx #np.min(np.where(PUPresults[11,:] == 1))
                        outstuff = [PUPresults[4,shIdx], PUPresults[1,shIdx], PUPresults[2,shIdx], PUPresults[6,shIdx], ATresults[2,shIdx], PUPresults[9,shIdx], PUPresults[0,shIdx], PUPresults[7,shIdx]]
                        outprint = str(i)
                        outprint = outprint.zfill(4) + '   '
                        for iii in outstuff:
                            outprint = outprint +'{:6.3f}'.format(iii) + ' '
                        SITfiles[pidx[0]].write(outprint+'\n')
                    
            # Save results for the full simulation (satellite independent)
                          
            # For ANTEATR, save CME id number (necessary? matches other file formats)
            # total time, velocity at impact, nose distance, Elon at impact, Elon at 213 Rs
            
            # old->CME: 0 t, 1 r, 2 vFront, 3 AW, 4 AWp, 5 delAx, 6 delCS, 7 delCA, 8 B, 9 Cnm, 10 n, 11 Temp, 12 yaw, 13 yaw v, 14 reg, 15 vEdge 16 vCent 17  vrr 18 vrp 19 vLr 20 vLp 21 vXCent]
            for j in range(len(ATresults[0])):
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = [ATresults[0,j], ATresults[1,j], ATresults[3,j], ATresults[4,j], ATresults[5,j], ATresults[6,j], ATresults[7,j], ATresults[2,j], ATresults[15,j], ATresults[16,j], ATresults[17,j], ATresults[18,j], ATresults[19,j], ATresults[20,j], ATresults[8,j], ATresults[9,j], tau, ATresults[10,j], ATresults[11,j], ATresults[12,j], ATresults[13,j], ATresults[14,j], ATresults[15,j]]
                for iii in outstuff:
                    outprint = outprint +'{:6.3f}'.format(iii) + ' '
                ANTEATRfile.write(outprint+'\n')
                
            # save PUP profile (if doing)
            if doPUP:
                if CME.hasSheath:
                    for j in range(len(PUPresults[0])):
                        outprint = str(i)
                        outprint = outprint.zfill(4) + '   '
                        outstuff = [PUPresults[0,j], PUPresults[1,j], PUPresults[2,j], PUPresults[3,j], PUPresults[4,j], PUPresults[5,j], PUPresults[6,j], PUPresults[7,j], PUPresults[8,j], PUPresults[9,j], PUPresults[10,j], PUPresults[11,j]]
                        for iii in outstuff:
                            outprint = outprint + '{:6.3f}'.format(iii) + ' '
                        PUPfile.write(outprint+'\n')
                else:
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    outstuff = np.zeros(12)+8888
                    for iii in outstuff:
                        outprint = outprint + '{:6.3f}'.format(iii) + ' '
                    PUPfile.write(outprint+'\n')
                
                
        elif ATresults[0][0] == 8888:
            print ('ANTEATR-PARADE forces unstable')
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = np.zeros(23)+8888
            for iii in outstuff:
                outprint = outprint +'{:6.3f}'.format(iii) + ' '
            ANTEATRfile.write(outprint+'\n')
            if doPUP:
                # PUPfile
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = np.zeros(12)+8888
                for iii in outstuff:
                    outprint = outprint + '{:6.3f}'.format(iii) + ' '
                PUPfile.write(outprint+'\n')
                # SITfile
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = np.zeros(8)+8888
                for iii in outstuff:
                    outprint = outprint + '{:6.3f}'.format(iii) + ' '
                SITfile.write(outprint+'\n')
            if doFIDO:
                for afile in FIDOfiles:
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    outstuff = np.zeros(9)+8888
                    for iii in outstuff:
                        outprint = outprint + '{:6.3f}'.format(iii) + ' '
                    afile.write(outprint+'\n')
                
        else:
            print('Miss')
               
    ANTEATRfile.close()  
    if doPUP: 
        PUPfile.close()
        for afile in SITfiles:
            afile.close()
    if doFIDO:
        for afile in FIDOfiles:
            afile.close()
    
    
# This have been updated since FIDO has now been integrated with ANTEATR
# We now run ANTEATR with all the forces turned off if want only FIDO
# This allows all SW values to be left at defaults since they are not used
def goFIDO(satPathIn=False):
    # ANTEATR portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|
    
    # Sort out if doing one satellite or multi and separate names if needed
    if satPathIn:
        satNames = np.array(satPathIn[-1])
        satPath  = satPathIn[:-1]
    else:
        satPath = False
        satNames = np.array(['sat1'])
    nSats = len(satNames)
    

    # Open a file to save the ANTEATR output
    FIDOfiles = []
    # Will either open one for each given name or a single one if don't have sat names    
    for satName in satNames:
        if (len(satNames) == 1) and (satName == 'sat1'):
            satName = ''
        
        # reset the generic 'sat1' to nothing if only doing 1 case with temp name
        FIDOfile = open(Dir+'/FIDOresults'+thisName+satName+'.dat', 'w')
        FIDOfiles.append(FIDOfile)
    FIDOfiles = np.array(FIDOfiles)
    
    # ANTEATR takes inputs
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, FRB, nSW, vSW, BSW, Cd, tau, cnm]         <-this is outdated
    # SatVars0 = [Satlat, Satlon, Satradius] -> technically doesn't have to be Earth!
    # which we can generate from the ForeCAT data

    # Pull in ANTEATR values from the input file
    global SatVars0
    SatVars0, Cd, mySW = processANTinputs(input_values, hasPath=satPath)

    # Assume that the given lon is the time of eruption, need to add in
    # the orbit during the ForeCAT run
    SatRotRate = SatVars0[3]
    SatLons = [SatVars0[1]+SatRotRate*CMEarray[i].t for i in range(nRuns)]

    global impactIDs, SWv, SWn, ANTCMErs
    ANTCMErs = {}
    impactIDs = []
    
    actualPath = False
    # Check if we're using satPath to pass multi simple satellites
    if satPath:
        try:
            a = float(satPath[0][0])
        except:
            actualPath = True
    
    # Loop over all the CMEs    
    for i in range(nRuns):
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr0, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        
        # reset path functions for t0 at start of ANTEATR
        # need to do for each satellite if > 1
        if satPath:
            
            if actualPath:
                satfs = [[] for i in range(len(satPath))]
                satfs[0] = [lambda x: satPath[0][0](x + CME.t*60), lambda x: satPath[0][1](x + CME.t*60), lambda x: satPath[0][2](x + CME.t*60)]
                if nSats >= 2:
                    satfs[1] = [lambda x: satPath[1][0](x + CME.t*60), lambda x: satPath[1][1](x + CME.t*60), lambda x: satPath[1][2](x + CME.t*60)] 
                if nSats >= 3:
                    satfs[2] = [lambda x: satPath[2][0](x + CME.t*60), lambda x: satPath[2][1](x + CME.t*60), lambda x: satPath[2][2](x + CME.t*60)]
                if nSats >= 4:
                    satfs[3] = [lambda x: satPath[3][0](x + CME.t*60), lambda x: satPath[3][1](x + CME.t*60), lambda x: satPath[3][2](x + CME.t*60)]
                if nSats >= 5:
                    satfs[4] = [lambda x: satPath[4][0](x + CME.t*60), lambda x: satPath[4][1](x + CME.t*60), lambda x: satPath[4][2](x + CME.t*60)]
                if nSats >= 6:
                    satfs[5] = [lambda x: satPath[5][0](x + CME.t*60), lambda x: satPath[5][1](x + CME.t*60), lambda x: satPath[5][2](x + CME.t*60)]
                if nSats >= 7:
                    satfs[6] = [lambda x: satPath[6][0](x + CME.t*60), lambda x: satPath[6][1](x + CME.t*60), lambda x: satPath[6][2](x + CME.t*60)]
                if nSats >= 8:
                    satfs[7] = [lambda x: satPath[7][0](x + CME.t*60), lambda x: satPath[7][1](x + CME.t*60), lambda x: satPath[7][2](x + CME.t*60)]
                if nSats >= 9:
                    satfs[8] = [lambda x: satPath[8][0](x + CME.t*60), lambda x: satPath[8][1](x + CME.t*60), lambda x: satPath[8][2](x + CME.t*60)]
                if nSats >= 10:
                    satfs[9] = [lambda x: satPath[9][0](x + CME.t*60), lambda x: satPath[9][1](x + CME.t*60), lambda x: satPath[9][2](x + CME.t*60)]
            
            
                # Only give it the names if 2 or more satellites  
                if nSats >= 2:             
                    satfs.append(satNames)
     
        # Set up initial satellite(s) position
        # Given vals are at OSPREI time 0, possibly need to include ForeCAT time now
        if satPath:
            if actualPath:
                myParams = []
                for ii in range(len(satPath)):
                    thisParam = [satfs[ii][0](CMEarray[i].t*60), satfs[ii][1](CMEarray[i].t*60), satfs[ii][2](CMEarray[i].t*60)*7e10,0]
                    myParams.append(thisParam)
                myParams.append(satNames)
            else:
                myParams = []
                for ii in range(len(satPath)):
                     thisParam = np.copy(satPath[ii])
                     thisParam[1] = thisParam[1]+thisParam[3]*CMEarray[i].t 
                     thisParam[2] = thisParam[2] * 7e10
                     myParams.append(thisParam)
                myParams.append(satNames)
        else:
            myParams = SatVars0
            myParams[1] = SatLons[i]
            myParams = [[myParams[0], myParams[1], myParams[2]*7e10, myParams[3]], ['sat1']]

        # Pull in this CME's parameters for input array
        # Tilt 
        tilt = CME.tilt
        # Calculate vr
        vr = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
        # Mass
        mass = CME.M/1e15
        # CME shape
        cmeAW = CME.AW*radeg
        cmeAWp = CME.AWp*radeg
        deltax = CME.deltaAx
        deltap = CME.deltaCS
        deltaCA = CME.deltaCSAx
        IVDfs = CME.IVDfs
        

        # Add in ensemble variation if desired
        if (i > 0) and useFCSW:
            if 'SWn' in EnsInputs: CME.nSW = np.random.normal(loc=CME.nSW, scale=EnsInputs['SWn'])
            if 'SWv' in EnsInputs: CME.vSW = np.random.normal(loc=CME.vSW, scale=EnsInputs['SWv'])
            if 'SWB' in EnsInputs: CME.BSW = np.random.normal(loc=CME.BSW, scale=EnsInputs['SWB'])              
            if 'SWT' in EnsInputs: CME.TSW = np.random.normal(loc=CME.TSW, scale=EnsInputs['SWT'])              
        FRB, tau, cnm, FRT = CME.FRBtor, CME.tau, CME.cnm, CME.FRT
        
        # Package up invec, run ANTEATR        
        gamma = CME.gamma
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, FRB, Cd, tau, cnm, FRT, gamma, CME.yaw]
        SWvec = [CME.nSW, CME.vSW, np.abs(CME.BSW), CME.TSW]
        
        # check if given SW 1D profiles
        if flag1DSW:
            SWvec = SWfile
        MHin = False
        if doMH:
            MHin = [CME.MHarea, CME.MHdist]
        
        # SW polarity in or out - THINK THIS WAS FLIPPED FOR ISSI/PROPOSAL CASE?
        inorout = np.sign(CME.BSW) 
        # high fscales = more convective like
        
        isSilent = False
        if actualPath:            
            ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, satfs=satfs, flagScales=flagScales, doPUP=doPUP, MEOWHiSS=MHin, aFIDOinside=doFIDO, inorout=inorout, thermOff=True, csOff=True, axisOff=True, dragOff=True)
        else:
            ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, flagScales=flagScales, doPUP=doPUP, MEOWHiSS=MHin, aFIDOinside=doFIDO, inorout=inorout, thermOff=True, csOff=True, axisOff=True, dragOff=True)
            
                      
        # Check if miss or hit  
        # Need to account for mix of hits and misses with multi TO DO!!
        if ATresults[0][0] not in [9999, 8888]:
            impactIDs.append(i)
            
            for sat in satNames:
                pidx = np.where(satNames == sat)[0]
                thisSum = outSum[sat]
                # check if input for this sat
                if thisSum[0] != -9999:
                    # Save FIDO profiles
                    theseFIDOres = FIDOresults[sat]
                
                    for j in range(len(theseFIDOres[0])):
                        outprint = str(i)
                        outprint = outprint.zfill(4) + '   '
                        Btot = np.sqrt(theseFIDOres[1][j]**2 + theseFIDOres[2][j]**2 + theseFIDOres[3][j]**2)
                        if isSat:
                            outstuff = [theseFIDOres[0][j]/24, Btot, theseFIDOres[1][j], theseFIDOres[2][j], theseFIDOres[3][j], theseFIDOres[4][j], theseFIDOres[5][j], theseFIDOres[6][j], theseFIDOres[7][j]]
                        else:
                            # GSE coords, flip R,T to xy
                            outstuff = [theseFIDOres[0][j]/24, Btot, -theseFIDOres[1][j], -theseFIDOres[2][j], theseFIDOres[3][j], theseFIDOres[4][j], theseFIDOres[5][j], theseFIDOres[6][j], theseFIDOres[7][j]]    
                        for iii in outstuff:
                            outprint = outprint +'{:6.3f}'.format(iii) + ' '
                        FIDOfiles[pidx[0]].write(outprint+'\n')
                else:
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    outstuff = np.zeros(9)+8888
                    for iii in outstuff:
                        outprint = outprint + '{:6.3f}'.format(iii) + ' '
                    FIDOfile.write(outprint+'\n')
               
        elif ATresults[0][0] == 8888:
            for afile in FIDOfiles:
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = np.zeros(9)+8888
                for iii in outstuff:
                    outprint = outprint + '{:6.3f}'.format(iii) + ' '
                afile[pidx[0]].write(outprint+'\n')
                
        else:
            print('Miss')
               
    for afile in FIDOfiles:
        afile.close()

def genNXTtxt(CME, num='', tag=''):
    # Create a file that could be used to restart a run from a midpoint location
    print ('Saving restart in nxt'+tag+thisName+num+'.txt')
    NXTtxt = open(Dir+'/nxt'+tag+thisName+num+'.txt', 'w')
    for key in currentInps:
        if key == 'models':
            NXTtxt.write('models: '+ tag+' \n')
        else:
            NXTtxt.write(key+': '+str(currentInps[key])+' \n')
    NXTtxt.close()
    
def thankslambdas(satfiles, satPos0, dObj, nSats):
    satPaths = []
    
    satRfA, satLatfA, satLonfA = makeSatPaths(satfiles[0], dObj, Clon0=satPos0[0][1])
    satPaths.append([satLatfA, satLonfA, satRfA])
    if nSats >= 2:
        satRfB, satLatfB, satLonfB = makeSatPaths(satfiles[1], dObj, Clon0=satPos0[1][1])
        satPaths.append([satLatfB, satLonfB, satRfB])
    if nSats >= 3:
        satRfC, satLatfC, satLonfC = makeSatPaths(satfiles[2], dObj, Clon0=satPos0[2][1])
        satPaths.append([satLatfC, satLonfC, satRfC])
    if nSats >= 4:
        satRfD, satLatfD, satLonfD = makeSatPaths(satfiles[3], dObj, Clon0=satPos0[3][1])
        satPaths.append([satLatfD, satLonfD, satRfD])
    if nSats >= 5:
        satRfE, satLatfE, satLonfE = makeSatPaths(satfiles[4], dObj, Clon0=satPos0[4][1])
        satPaths.append([satLatfE, satLonfE, satRfE])
    if nSats >= 6:
        satRfF, satLatfF, satLonfF = makeSatPaths(satfiles[5], dObj, Clon0=satPos0[5][1])
        satPaths.append([satLatfF, satLonfF, satRfF])
    if nSats >= 7:
        satRfG, satLatfG, satLonfG = makeSatPaths(satfiles[6], dObj, Clon0=satPos0[6][1])
        satPaths.append([satLatfG, satLonfG, satRfG])
    if nSats >= 8:
        satRfH, satLatfH, satLonfH = makeSatPaths(satfiles[7], dObj, Clon0=satPos0[7][1])
        satPaths.append([satLatfH, satLonfH, satRfH])
    if nSats >= 9:
        satRfI, satLatfI, satLonfI = makeSatPaths(satfiles[8], dObj, Clon0=satPos0[8][1])
        satPaths.append([satLatfI, satLonfI, satRfI])
    if nSats == 10:
        satRfJ, satLatfJ, satLonfJ = makeSatPaths(satfiles[9], dObj, Clon0=satPos0[9][1])
        satPaths.append([satLatfJ, satLonfJ, satRfJ])
        
    return satPaths

def satPathWrapper(satPath, checkSatlen=True):
    satNames = []
    satPos0  = []
    satfiles = []
    #havePaths = True
    if satPath[-5:] == '.sats':
        satFile = np.genfromtxt(satPath, dtype='unicode', delimiter=' ')
        nSats = len(satFile)
        if (nSats > 10) and checkSatlen:
            sys.exit('Can only run 10 or fewer satellites')
        nItems = len(satFile[0])
        if len(satFile.shape) == 1:
            nItems = len(satFile)
            nSats = 1
            satFile = [satFile] # nest to make looping over Sats happy
        if nItems not in [5,6,9]:
            sys.exit('.sats file should be SatName Lat0 Lon0 R0 Orbit/PathFile [ObsData SheathStart FRStart FRend (Optional)]')
            
        if '.' in satFile[0][4]:
            havePaths = True
        else:
            havePaths = False
             
        if havePaths:
            for i in range(nSats):
                if (satFile[i][4][-4:] != '.sat'):
                    sys.exit('Need to provide satellite files as .sat')
                    
        for i in range(nSats):
            thisSat = satFile[i]
            satNames.append(thisSat[0])
            satPos0.append([float(thisSat[1]), float(thisSat[2]), float(thisSat[3]), 0])
            if not havePaths:
                satPos0[-1][3] = float(thisSat[4])
            else:
                satfiles.append(thisSat[4])
    else:
        nSats = 1
        satNames.append('sat1')
        satPos0.append([satPos[0], satPos[1], satPos[2]*7e10, 0])
        satfiles.append(satPath) 
        if '.' in satPath:
            havePaths = True
        else:
            havePaths = False

    if havePaths:
        satPaths = thankslambdas(satfiles, satPos0, dObj, nSats)
    else:
        satPaths = satPos0
    satPaths.append(satNames)
    
    return satPaths

def runOSPREI(inputPassed='noFile'):
    setupOSPREI(logInputs=True, inputPassed=inputPassed)
    
    if nRuns > 1: setupEns()

    global CMEarray
    CMEarray = []
    
    global currentInps
    currentInps = {}
    for key in input_values: currentInps[key] = input_values[key]
    
    global doSatPath
    doSatPath = False
    
    if 'satPath' in input_values:
        satPath = input_values['satPath']
        satPaths = satPathWrapper(satPath)

    if doFC:
        goForeCAT(makeRestart=False)        
    else:
        # Fill in CME array for ANTEATR or FIDO
        makeCMEarray()

    if doANT: 
        if 'satPath' in input_values:
            goANTEATR(makeRestart=False, satPathIn=satPaths)
        else:
            goANTEATR(makeRestart=False)
            
    # Option for FIDO only when not running ANTEATR
    # (actually just run ANTEATR w/no forces near arrival)
    if doFIDO and not doANT: 
        goFIDO(satPathIn=doSatPath)

    if nRuns > 1: ensembleFile.close()

if __name__ == '__main__':
    runOSPREI()
