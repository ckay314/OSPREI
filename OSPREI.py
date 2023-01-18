import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import datetime
from scipy.interpolate import CubicSpline


# Import all the OSPREI files, make this match your system
mainpath = '/Users/ckay/OSPREI/' #MTMYS
codepath = mainpath + 'codes/'
magpath  ='/Users/ckay/PickleJar/'
sys.path.append(os.path.abspath(codepath)) 

from ForeCAT import *
import CME_class as CC
from ForeCAT_functions import readinputfile, calc_dist, calc_SW
import ForceFields 
#from funcANTEATR import *
#from PARADE import *
from ANT_PUP import *
import FIDO as FIDO

# Useful constant
global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

# pick a number to seed the random number generator used by ensembles
np.random.seed(202201262)

def setupOSPREI():
    # Initial OSPREI setup ---------------------------------------------|
    # Make use of ForeCAT function to read in vars----------------------|
    # ------------------------------------------------------------------|
    global input_values, allinputs, date, noDate
    input_values, allinputs = FC.readinputfile()
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
    global time, satPos, Sat_rot, ObsDataFile, mass, useFCSW, flagScales, flag1DSW, doPUP, doMH, isSat
    global obsFRstart, obsFRend, obsShstart, vSW, MHarea, MHdist, satPath
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
    isSat   = False
    obsFRstart, obsFRend, obsShstart = None, None, None
    mass = 5.
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
        print('Simulation starts at '+dObj.strftime('%Y %b %d %H:%M '))
                        
    global thisName
    if noDate:
        thisName = suffix
    else:
        thisName = date+suffix
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
            
    print( 'Running: ')
    print( 'ForeCAT: ', doFC)
    print( 'ANTEATR: ', doANT)
    print( 'FIDO:    ', doFIDO)
    print('')

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
    
    

    # Important parameters with defaults ----------------------------------------------------------------
    if 'CMEr' in input_values:
        CMEr = float(input_values['CMEr'])
        if startpoint == 'FC':
            if (CMEr < 1.05) or (CMEr > 2.5):
                sys.exit('Initial CME front distance (CMEr) must be within [1.05, 2.5] Rs for ForeCAT start') 
        elif startpoint == 'ANT':
            if (CMEr < 10) or (CMEr > 50):
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
        if (CMEAW < 5) or (CMEAW > 90) :
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
        CMEM = 0.005756 * float(input_values['CMEvr']) -0.84
        if CMEM > 0:
            input_values['CMEM'] = str(CMEM)
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
        if (Gamma < 0) or  (Gamma > 1):
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
        dtor = 3.14159 / 360.
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
            if (FRB < 500) or  (FRB > 10000):
                sys.exit('Flux rope B (FRB) must be in [500, 10000] nT') 
        else:
            KE = 0.5 * mass*1e15 * (v*1e5)**2 /1e31
            phiflux = np.power(10, np.log10(KE / 0.19) / 1.87)*1e21
            B0 = phiflux * Cnm * (delCS**2 + 1) / avgR / Ltot / delCS**2 *1e5
            Bcent = delCS * tau * B0
            if (Bcent < 500) or  (Bcent > 10000):
                sys.exit('Cannot calculate a reasonable default flux rope B using empirical scaling. Please provide FRB')
            else:
                print('Using ', FRB, ' nT for FRB')
                input_values['FRB'] = str(Bcent)
    
        if 'FRT' in input_values:
            FRT = float(input_values['FRT'])
            if (FRT < 5e4) or  (FRT > 1e6):
                sys.exit('Flux rope T (FRT) must be in [5e4, 1e6] nT') 
        else:
            vSheath = 0.129 * v + 376
            vIS = (vSheath + 51.73) / 1.175
            vExp = 0.175 * vIS -51.73
            logTIS = 3.07e-3 * vIS +3.65
            FRT = np.power(10, logTIS) * np.power(215*7e10/rFront, 0.7)
            if (FRT < 5e4) or  (FRT > 1e6):
                sys.exit('Cannot calculate a reasonable default flux rope T using empirical scaling. Please provide FR T')
            else:
                print('Using ', FRT, ' K for FRT')
                
                input_values['FRT'] = str(FRT)
        
        
    # Satellite Parameters ----------------------------------------------------------------
    hasSatPath = False
    if 'satPath' in input_values:
        hasSatPath = True
        
    if 'SatLon' in input_values:
        SatLon = float(input_values['SatLon'])
        if (SatLon < -180) or  (SatLon > 360):
            sys.exit('Satellite longitude (SatLon) must be in [-180, 180] or [0, 360] deg') 
    else:
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
            if (SWn < 0) or  (SWn > 10):
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
            if (SWB < 0.1) or  (SWB > 50):
                sys.exit('Solar wind magnetic field strength (SWB) must be in [0.1, 50] nT if satR is near 1 AU') 
        else:
            input_values['SWB'] = str(6)
        
        if 'SWT' in input_values:
            SWT = float(input_values['SWT'])
            if (SWT < 0.1) or  (SWT > 50):
                sys.exit('Solar wind temperature (SWT) must be in [1e4, 5e5] K if satR is near 1 AU') 
        else:
            input_values['SWT'] = str(75000)
        
    
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
        nRuns = float(input_values['nRuns'])
        if (nRuns < 0) or  (nRuns > 10):
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
    
    hasFRstart, hasFRend = False, False    
    if 'obsFRstart' in input_values:
        obsFRstart = float(input_values['obsFRstart'])
        if (obsFRstart < 0) or  (obsFRstart >= 360):
            sys.exit('Observed flux rope start (obsFRstart) must in [0, 360) fraction DoY') 
        else:
            hasFRstart = True

    if 'obsFRend' in input_values:
        obsFRend = float(input_values['obsFRend'])
        if (obsFRend < 0) or  (obsFRend >= 360):
            sys.exit('Observed flux rope end (obsFRend) must in [0, 360) fraction DoY') 
        else:
            hasFRend = True

    if hasFRstart and not hasFRend:
        sys.exit('Provided FRstart but not FRend. Need both or neither.')

    if hasFRend and not hasFRstart:
        sys.exit('Provided FRend but not FRstart. Need both or neither.')

    if 'obsShstart' in input_values:
        obsShstart = float(input_values['obsShstart'])
        if (obsShstart < 0) or  (obsShstart >= 360):
            sys.exit('Observed shock/sheath start (obsShstart) must in [0, 360) fraction DoY') 
    
        
    # MEOW-HiSS parameters ----------------------------------------------------------
    doMH = False
    if 'doMH' in input_values:
        doMH = input_values['doMH']
        if doMH in ['True', 'False']:
            if doMH == 'True':
                doMH = True
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
            if (MHdist < -0.3) or  (MHdist > 1.5):
                sys.exit('Initial HSS front distance (MHdist) must be in [-0.3, 1.5] AU') 
        else:
            sys.exit('Initial HSS front distance (MHdist) must be provided if MEOW-HiSS is included')
        
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
    possible_vars =  ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB', 'PFSSscale', 'IVDf1', 'IVDf2', 'IVDf', 'Gamma', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWT', 'FRB', 'FRtau', 'FRCnm', 'FRT', 'MHarea', 'MHdist']
    print( 'Determining parameters varied in ensemble...')
    EnsData = np.genfromtxt(FC.fprefix+'.ens', dtype=str, encoding='utf8')
    # Make a dictionary containing the variables and their uncertainty
    global EnsInputs
    EnsInputs = {}
    for i in range(len(EnsData)):
        #temp = EnsData[i]
        if len(EnsData) > 1:
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
    global ensembleFile
    ensembleFile = open(Dir+'/EnsembleParams'+thisName+'.dat', 'w')
    # Add a header and the seed values
    # Have to do ForeCAT values first then non ForeCAT based on how 
    # genEnsMem runs.  
    # nSW and vSW variations are only captured in ANTEATR output
    outstr1 = 'RunID '
    outstr2 = '0000 '
    for item in EnsInputs.keys():
        if item not in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWT', 'FRB', 'FRT', 'IVDf1', 'IVDf2', 'Gamma', 'MHarea', 'MHdist']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
    for item in EnsInputs.keys():
        if item in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWT',  'FRB', 'FRT', 'IVDf1', 'IVDf2', 'Gamma', 'MHarea', 'MHdist']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
    ensembleFile.write(outstr1+'\n')
    ensembleFile.write(outstr2+'\n')
            
def genEnsMem(runnum=0):
    # Vary parameters for all models at the same time
    outstr = str(runnum)
    outstr = outstr.zfill(4) + '   '
    flagAccel = False
    flagExp   = False
    new_pos = [float(input_values['CMElat']), float(input_values['CMElon']), float(input_values['CMEtilt'])]
    for item in EnsInputs.keys():
        # Sort out what variable we adjust for each param
        # The lambda functions will auto adjust to new global values in them
        if item == 'CMElat':
            new_pos[0] = np.random.normal(loc=float(input_values['CMElat']), scale=EnsInputs['CMElat'])
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
    
    # add changes to non ForeCAT things onto the CME object
    for item in EnsInputs.keys():
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
            outstr += '{:5.2f}'.format(CME.B0) + ' '
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
        thisdate = datetime.datetime.strptime(thisdatestr, '%Y-%m-%d %H:%M:%S')
        delts.append((thisdate - tzero).total_seconds())
    satrs   = np.array(data[:,2]).astype(float) * 1.49e13 / rsun # convert to cm then rsun
    satlats = np.array(data[:,3]).astype(float)
    satlons = np.array(data[:,4]).astype(float) % 360.
    idx = np.where(np.abs(delts) == np.min(np.abs(delts)))[0]
    lon0 = satlons[idx]
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
        CME, path = runForeCAT(CME, rmax, path=True)
        
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
            outstuff = [path[0,j], path[1,j], path[2,j], path[3,j], path[4,j], path[5,j], path[6,j], path[7,j][0],  path[7,j][1],  path[7,j][2],  path[7,j][3],  path[7,j][4],  path[7,j][5],  path[7,j][6], path[8,j], path[9,j], path[10,j], path[11,j]]
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
    if 'MHarea'    in input_values: CME.MHarea = float(input_values['MHarea'])
    if 'MHdist'    in input_values: CME.MHdist = float(input_values['MHdist'])

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
    

def goANTEATR(makeRestart=False, satPath=False):
    # ANTEATR portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|

    # Open a file to save the ANTEATR output
    ANTEATRfile = open(Dir+'/ANTEATRresults'+thisName+'.dat', 'w')
    if doPUP:
        PUPfile = open(Dir+'/PUPresults'+thisName+'.dat', 'w')
        SITfile = open(Dir+'/SITresults'+thisName+'.dat', 'w')
    if doFIDO:
        FIDOfile = open(Dir+'/FIDOresults'+thisName+'.dat', 'w')
        


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

    global ANTsatLons, impactIDs, SWv, SWn, ANTCMErs
    ANTsatLons = {}
    ANTCMErs = {}
    impactIDs = []
        
    for i in range(nRuns):
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr0, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        # reset path functions for t0 at start of ANTEATR
        if satPath:
            satLatf2 = lambda x: satLatf(x + CME.t*60)
            satLonf2 = lambda x: satLonf(x + CME.t*60)
            satRf2   = lambda x: satRf(x + CME.t*60)
        
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
        
        # get sat pos using simple orbit
        myParams = SatVars0
        myParams[1] = SatLons[i]
        # replace with the correct sat position for this time if using path
        if satPath:
            myParams = [satLatf(CMEarray[0].t*60), satLonf(CMEarray[0].t*60), satRf(CMEarray[0].t*60),0]
                                     
        # Add in ensemble variation if desired
        if (i > 0) and useFCSW:
            if 'SWn' in EnsInputs: CME.nSW = np.random.normal(loc=CME.nSW, scale=EnsInputs['SWn'])
            if 'SWv' in EnsInputs: CME.vSW = np.random.normal(loc=CME.vSW, scale=EnsInputs['SWv'])
            if 'SWB' in EnsInputs: CME.BSW = np.random.normal(loc=CME.BSW, scale=EnsInputs['SWB'])              
            if 'SWT' in EnsInputs: CME.TSW = np.random.normal(loc=CME.TSW, scale=EnsInputs['SWT'])              
        FRB, tau, cnm, FRT = CME.FRBtor, CME.tau, CME.cnm, CME.FRT
        # Package up invec, run ANTEATR        
        gamma = CME.gamma
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, np.abs(FRB), Cd, tau, cnm, FRT, gamma]
        SWvec = [CME.nSW, CME.vSW, np.abs(CME.BSW), CME.TSW]
        # check if given SW 1D profiles
        if flag1DSW:
            SWvec = SWfile
        
        # SW polarity in or out
        inorout = np.sign(CME.BSW) 
        # high fscales = more convective like
        isSilent = False
        if satPath:
            if doMH:
                ATresults, Elon, CME.vs, estDur, thetaT, thetaP, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, satfs=[satLatf2, satLonf2, satRf2], flagScales=flagScales, doPUP=doPUP, MEOWHiSS=[CME.MHarea, CME.MHdist], aFIDOinside=doFIDO, inorout=inorout, name='testt')
            else:
                ATresults, Elon, CME.vs, estDur, thetaT, thetaP, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, satfs=[satLatf2, satLonf2, satRf2], flagScales=flagScales, doPUP=doPUP, aFIDOinside=doFIDO, inorout=inorout)
        else:
            if doMH:
                ATresults, Elon, CME.vs, estDur, thetaT, thetaP, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, flagScales=flagScales, doPUP=doPUP, MEOWHiSS=[CME.MHarea, CME.MHdist], aFIDOinside=doFIDO, inorout=inorout)
            else:    
                ATresults, Elon, CME.vs, estDur, thetaT, thetaP, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, flagScales=flagScales, doPUP=doPUP, aFIDOinside=doFIDO, inorout=inorout)
        
        # update background SW params to current values
        # will do nothing if using given values but needed for
        # 1D profile to make FIDO-SIT happy
        CME.nSW, CME.vSW, CME.BSW, CME.BSWvec = SWparams[0], SWparams[1], SWparams[2], SWparams[3]        
        # Check if miss or hit  
        if ATresults[0][0] not in [9999, 8888]:
            impactIDs.append(i)
            # get improved v estimates using FIDO code
            vFvec, vExvec = FIDO.getvCMEframe(1., thetaT, thetaP, ATresults[5][-1], ATresults[6][-1], CME.vs)
            temp = rotx(vFvec, -(90.-tilt))
            temp2 = roty(temp, CMElat - myParams[0]) 
            vInSitu = rotz(temp2, CMElon - myParams[1])
            vF = vInSitu[0] / 1e5
            #print (sd)
            # this is vExp for axis and cs not the same as measured vexp....
            temp = rotx(vExvec, -(90.-tilt))
            temp2 = roty(temp, CMElat - myParams[0]) 
            vExVec = rotz(temp2, CMElon - myParams[1])
            vEx = vExVec[0] / 1e5
        
            # Can add in ForeCAT time to ANTEATR time
            FCATtime = CME.t/60./24
            # want time of first contact, which is not idx -1 if do fullContact/FIDO
            hitIdx = -1
            if doFIDO:
                regs = FIDOresults[7]
                hitTime = FIDOresults[0][np.min(np.where(regs == 1))]/24
                hitIdx = np.min(np.where(ATresults[0] >= hitTime))
            TotTime  = ATresults[0][hitIdx]+FCATtime
            rCME     = ATresults[1][hitIdx]
            CMEvs    = ATresults[2][hitIdx]
            CMEAW    = ATresults[3][hitIdx]
            CMEAWp   = ATresults[4][hitIdx]
            deltax   = ATresults[5][hitIdx]
            deltap   = ATresults[6][hitIdx]
            deltaCA  = ATresults[7][hitIdx]
            B0       = ATresults[8][hitIdx]/1e5
            cnm      = ATresults[9][hitIdx]
            CMEn     = ATresults[10][hitIdx]
            logT     = ATresults[11][hitIdx]
            # Store things to pass to FIDO ensembles
            ANTsatLons[i] = Elon # lon at time CME nose is at Earth/sat radius
            # update CME if has that variable
            CME.points[CC.idcent][1,0] = rCME
            CME.AW = CMEAW*dtor
            CME.AWp = CMEAWp*dtor
            CME.deltaAx = deltax
            CME.deltaCS = deltap
            CME.deltaCSAx = deltaCA
            # this B0 is actually Btor, which is what FIDO wants (Btor at center at time of impact)
            CME.B0 = B0 * 1e5 * np.sign(FRB) * deltap * CME.tau # in nT now
            CME.cnm   = cnm
            CME.impV = vF
            CME.impVE = vEx
            #CME.t = TotTime
            CME.FRT = logT
            
            # CME sheath parameters
            if doPUP:
                # check if it was able to form a sheath (might not if vCME < vSW)
                if (1 in PUPresults[11]):
                    CME.hasSheath = True
                    shIdx = np.min(np.where(PUPresults[11]==1))
                    CME.vShock = PUPresults[0][shIdx]
                    CME.comp   = PUPresults[1][shIdx]
                    CME.shDur  = PUPresults[4][shIdx]
                    CME.shDens = PUPresults[6][shIdx]
                    CME.shB    = PUPresults[7][shIdx]
                    CME.shTheta = PUPresults[8][shIdx]
                    CME.shvt   = PUPresults[10][shIdx]
                    CME.shT    = PUPresults[9][shIdx]
                    CME.shv    = ATresults[2][shIdx][0]
            
            print (str(i)+' Contact after '+"{:.2f}".format(TotTime)+' days with front velocity '+"{:.2f}".format(vF)+' km/s (expansion velocity ' +"{:.2f}".format(vEx)+' km/s) when nose reaches '+"{:.2f}".format(rCME) + ' Rsun and angular width '+"{:.0f}".format(CMEAW)+' deg and estimated duration '+"{:.0f}".format(estDur)+' hr')
            # prev would take comp of v's in radial direction, took out for now !!!!
            if not noDate:
                dImp = dObj + datetime.timedelta(days=TotTime)
                print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
            print ('   Density: ', CMEn, '  Temp:  ', np.power(10,logT))
            
                             
            # For ANTEATR, save CME id number (necessary? matches other file formats)
            # total time, velocity at impact, nose distance, Elon at impact, Elon at 213 Rs
            for j in range(len(ATresults[0])):
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = [ATresults[0,j], ATresults[1,j], ATresults[3,j], ATresults[4,j], ATresults[5,j], ATresults[6,j], ATresults[7,j], ATresults[2,j][0], ATresults[2,j][1], ATresults[2,j][2], ATresults[2,j][3], ATresults[2,j][4], ATresults[2,j][5], ATresults[2,j][6], ATresults[8,j], ATresults[9,j], tau, ATresults[10,j], ATresults[11,j], ATresults[12,j]]
                for iii in outstuff:
                    outprint = outprint +'{:6.3f}'.format(iii) + ' '
                ANTEATRfile.write(outprint+'\n')
            # save PUP results (if doing)
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
                    
            if doFIDO:
                # Save FIDO profiles
                for j in range(len(FIDOresults[0])):
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    Btot = np.sqrt(FIDOresults[1][j]**2 + FIDOresults[2][j]**2 + FIDOresults[3][j]**2)
                    if isSat:
                        outstuff = [FIDOresults[0][j]/24+FCATtime, Btot, FIDOresults[1][j], FIDOresults[2][j], FIDOresults[3][j], FIDOresults[4][j], FIDOresults[5][j], FIDOresults[6][j], FIDOresults[7][j]]
                    else:
                        # GSE coords, flip R,T to xy
                        outstuff = [FIDOresults[0][j]/24+FCATtime, Btot, -FIDOresults[1][j], -FIDOresults[2][j], FIDOresults[3][j], FIDOresults[4][j], FIDOresults[5][j], FIDOresults[6][j], FIDOresults[7][j]]    
                    for iii in outstuff:
                        outprint = outprint +'{:6.3f}'.format(iii) + ' '
                    FIDOfile.write(outprint+'\n')
                
            if doPUP:
                # Save sheath/shock properties at first contact
                #outstuff = id [dur, comp, Mach, n, vsheath, B, vshock] + 
                # PUP 0 vShock, 1 r, 2 Ma, 3 wid, 4 dur, 5 mass, 6 dens 7 temp 8 theta 9 B 10 vt 11 init
                shIdx = np.min(np.where(PUPresults[11,:] == 1))
                outstuff = [PUPresults[4,shIdx], PUPresults[1,shIdx], PUPresults[2,shIdx], PUPresults[6,shIdx], ATresults[2,shIdx][0], PUPresults[9,shIdx], PUPresults[0,shIdx], PUPresults[7,shIdx]]
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                for iii in outstuff:
                    outprint = outprint +'{:6.3f}'.format(iii) + ' '
                SITfile.write(outprint+'\n')
                
                
        elif ATresults[0][0] == 8888:
            print ('ANTEATR-PARADE forces unstable')
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = np.zeros(20)+8888
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
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = np.zeros(9)+8888
                for iii in outstuff:
                    outprint = outprint + '{:6.3f}'.format(iii) + ' '
                FIDOfile.write(outprint+'\n')
                
        else:
            print('Miss')
        
                        
        # write a file to restart ANTEATR/FIDO from current CME values
        # this is probably outdated...
        if makeRestart:      
            # update current inps with things that have changed in ForeCAT to pass to
            currentInps['CMEr'] = rCME
            currentInps['CMEvr'] = CME.vs[0] /1e5
            currentInps['CMEAW']  = CMEAW
            currentInps['CMEAWp'] = CMEAWp
            currentInps['CMEdelAx'] = deltax
            currentInps['CMEdelCS'] = deltap
            currentInps['FRB'] = CME.B0
            currentInps['FRCnm'] = CME.cnm
            currentInps['SatLon'] = Elon
            currentInps['date'] = dImp.strftime('%Y')+dImp.strftime('%m')+dImp.strftime('%d')
            currentInps['time'] = dImp.strftime("%H:%M")
            # SW params if using those...
            genNXTtxt(CME, num=str(i), tag='FIDO')
        
        
    ANTEATRfile.close()  
    if doPUP: 
        PUPfile.close()
        SITfile.close()
    if doFIDO:
        FIDOfile.close()
    
    
# This have been updated since FIDO has now been integrated with ANTEATR
# We now run ANTEATR with all the forces turned off if want only FIDO
# This allows all SW values to be left at defaults since they are not used
def goFIDO(satPath=False):
    # FIDO portion -----------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|
    # Open a file to save the FIDO output
    FIDOfile = open(Dir+'/FIDOresults'+thisName+'.dat', 'w')

    # Pull in ANTEATR values from the input file
    global SatVars0
    SatVars0, Cd, mySW = processANTinputs(input_values, hasPath=satPath)
    
    # No FC so don't need to adjust satlons but still need to make array 
    SatRotRate = SatVars0[3]
    SatLons = [SatVars0[1] for i in range(nRuns)]
    
    global ANTsatLons, impactIDs, SWv, SWn, ANTCMErs
    ANTsatLons = {}
    ANTCMErs = {}
    impactIDs = []
    
    for i in range(nRuns):
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr0, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        # reset path functions for t0 at start of ANTEATR
        if satPath:
            satLatf2 = lambda x: satLatf(x + CME.t*60)
            satLonf2 = lambda x: satLonf(x + CME.t*60)
            satRf2   = lambda x: satRf(x + CME.t*60)
    
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
        
        # get sat pos using simple orbit
        myParams = SatVars0
        myParams[1] = SatLons[i]
        # replace with the correct sat position for this time if using path
        if satPath:
            myParams = [satLatf(CMEarray[0].t*60), satLonf(CMEarray[0].t*60), satRf(CMEarray[0].t*60),0]
            
        # Add in ensemble variation if desired
        if (i > 0) and useFCSW:
            if 'SWn' in EnsInputs: CME.nSW = np.random.normal(loc=CME.nSW, scale=EnsInputs['SWn'])
            if 'SWv' in EnsInputs: CME.vSW = np.random.normal(loc=CME.vSW, scale=EnsInputs['SWv'])
            if 'SWB' in EnsInputs: CME.BSW = np.random.normal(loc=CME.BSW, scale=EnsInputs['SWB'])              
            if 'SWT' in EnsInputs: CME.TSW = np.random.normal(loc=CME.TSW, scale=EnsInputs['SWT'])              
        FRB, tau, cnm, FRT = CME.FRBtor, CME.tau, CME.cnm, CME.FRT
        
        # Package up invec, run ANTEATR        
        gamma = CME.gamma
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, np.abs(FRB), Cd, tau, cnm, FRT, gamma]
        SWvec = [CME.nSW, CME.vSW, np.abs(CME.BSW), CME.TSW]
        # SW polarity in or out
        inorout = np.sign(CME.BSW) 
        
        isSilent = True
        ATresults, Elon, CME.vs, estDur, thetaT, thetaP, SWparams, PUPresults, FIDOresults = getAT(invec, myParams, SWvec, fscales=IVDfs, silent=isSilent, flagScales=flagScales, doPUP=False, aFIDOinside=doFIDO, inorout=inorout, thermOff=True, csOff=True, axisOff=True, dragOff=True)
        
        # Check if miss or hit  
        if ATresults[0][0] not in [9999, 8888]:
            impactIDs.append(i)
            if doFIDO:
                regs = FIDOresults[7]
                hitTime = FIDOresults[0][np.min(np.where(regs == 1))]/24
                hitIdx = np.min(np.where(ATresults[0] >= hitTime))
            TotTime  = ATresults[0][hitIdx]
            rCME     = ATresults[1][hitIdx]
            CMEvs    = ATresults[2][hitIdx]
            CMEAW    = ATresults[3][hitIdx]
            CMEAWp   = ATresults[4][hitIdx]
            deltax   = ATresults[5][hitIdx]
            deltap   = ATresults[6][hitIdx]
            deltaCA  = ATresults[7][hitIdx]
            B0       = ATresults[8][hitIdx]/1e5
            cnm      = ATresults[9][hitIdx]
            CMEn     = ATresults[10][hitIdx]
            logT     = ATresults[11][hitIdx]
            if not noDate:
                dImp = dObj + datetime.timedelta(days=TotTime)
                print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
            
            if doFIDO:
                # Save FIDO profiles
                for j in range(len(FIDOresults[0])):
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    Btot = np.sqrt(FIDOresults[1][j]**2 + FIDOresults[2][j]**2 + FIDOresults[3][j]**2)
                    if isSat:
                        outstuff = [FIDOresults[0][j]/24, Btot, FIDOresults[1][j], FIDOresults[2][j], FIDOresults[3][j], FIDOresults[4][j], FIDOresults[5][j], FIDOresults[6][j], FIDOresults[7][j]]
                    else:
                        # GSE coords, flip R,T to xy
                        outstuff = [FIDOresults[0][j]/24, Btot, -FIDOresults[1][j], -FIDOresults[2][j], FIDOresults[3][j], FIDOresults[4][j], FIDOresults[5][j], FIDOresults[6][j], FIDOresults[7][j]]    
                    for iii in outstuff:
                        outprint = outprint +'{:6.3f}'.format(iii) + ' '
                    FIDOfile.write(outprint+'\n')
            
        else:
            print ('Miss')
            
    
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
    

def runOSPREI():
    setupOSPREI()
    
    if nRuns > 1: setupEns()

    global CMEarray
    CMEarray = []
    
    global currentInps
    currentInps = {}
    for key in input_values: currentInps[key] = input_values[key]
    
    global doSatPath 
    doSatPath = False
    global satRf, satLatf, satLonf
    if 'satPath' in input_values:
        doSatPath = True
        satPath = input_values['satPath']
        # functions give correct R/lat/lon if give t in sec from start time
        satRf, satLatf, satLonf = makeSatPaths(satPath, dObj, Clon0=satPos[1])
        # reset the initial CME distance using satRf if doing FIDO only 
        # and wasn't originally given a distance
        if input_values['SatR'] == '9999':
           satPos[2] = satRf[0]
    if doFC:
        goForeCAT(makeRestart=False)        
    else:
        # Fill in CME array for ANTEATR or FIDO
        makeCMEarray()
    
    #readMoreInputs()

    if doANT: goANTEATR(makeRestart=False, satPath=doSatPath)
    
    # Option for FIDO only when not running ANTEATR
    # (actually just run ANTEATR w/no forces near arrival)
    if doFIDO and not doANT: 
        goFIDO(satPath=doSatPath)

    if nRuns > 1: ensembleFile.close()

if __name__ == '__main__':
    runOSPREI()
