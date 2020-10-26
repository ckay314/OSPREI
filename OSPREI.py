import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
import datetime

# Import all the OSPREI files, make this match your system
mainpath = '/Users/ckay/OSPREI/' #MTMYS
# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
sys.path.append(os.path.abspath(mainpath+'codes/')) #MTMYS
from ForeCAT import *
import CME_class as CC
from ForeCAT_functions import readinputfile, calc_dist, calc_SW
import ForceFields 
#from funcANTEATR import *
from PARADE import *
import FIDO as FIDO

# Useful constant
global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

np.random.seed(120150)

def setupOSPREI():
    # Initial OSPREI setup ---------------------------------------------|
    # Make use of ForeCAT function to read in vars----------------------|
    # ------------------------------------------------------------------|
    global input_values, allinputs, date
    input_values, allinputs = FC.readinputfile()
    try:
        date = input_values['date']
    except:
        sys.exit('Need name of magnetogram/date to run!!!')    
            
    # Pull in other values from allinputs
    # possible_vars = ['suffix', 'nRuns']   
    # Set defaults for these values
    global suffix, nRuns
    # these are values its convenient to read early for processOSPREI
    global time, satPos, shape, Sat_rot, ObsDataFile, includeSIT
    suffix = ''
    nRuns  = 1
    models = 'ALL'
    satPos = [0,0]
    shape  = [1,0.15] 
    Sat_rot = 360./365.25
    ObsDataFile = None
    includeSIT = False
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
        elif temp[0][:-1] == 'Sat_lat':
            satPos[0] = float(temp[1])
        elif temp[0][:-1] == 'Sat_lon':
            satPos[1] = float(temp[1])
        elif temp[0][:-1] == 'shapeA':
            shape[0] = float(temp[1])
        elif temp[0][:-1] == 'shapeB':
            shape[1] = float(temp[1])
        elif temp[0][:-1] == 'Sat_rot':
            Sat_rot = float(temp[1])
        elif temp[0][:-1] == 'ObsDataFile':
            ObsDataFile = temp[1]
        elif temp[0][:-1] == 'includeSIT':
            if temp[1] == 'True':
                includeSIT = True
            
            
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
    thisName = date+suffix
    print('Running '+str(nRuns)+' OSPREI simulation(s) for '+thisName)

    # See if we have a directory for output, create if not
    # I save things in individual folders, but you can dump it whereever
    # by changing Dir
    global Dir
    Dir = mainpath+date
    if not os.path.exists(Dir):
        os.mkdir(Dir)
    
    # Check to see if we are running all components or just part
    # flags are ALL = all three, IP = ANTEATR+FIDO, ANT = ANTEATR, FIDO = FIDO    
    # FC = ForeCAT only, noB = ForeCAT+ANTEATR
    # Default is ALL
    global doFC, doANT, doFIDO
    doFC, doANT, doFIDO = True, True, True
    if models in ['IP', 'FIDO', 'ANT']: doFC = False
    if models in ['ANT', 'noB']: doFIDO = False
    if models == 'FIDO': doANT = False
    if models == 'FC': doANT, doFIDO, = False, False

    print( 'Running: ')
    print( 'ForeCAT: ', doFC)
    print( 'ANTEATR: ', doANT)
    print( 'FIDO:    ', doFIDO)

def setupEns():
    # All the possible parameters one could in theory want to vary
    possible_vars =  ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB', 'CMEvExp', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWcs', 'SWvA', 'FRBscale', 'FRtau', 'FRCnm', 'CMEvTrans', 'SWBx', 'SWBy', 'SWBz']
    print( 'Determining parameters varied in ensemble...')
    EnsData = np.genfromtxt(FC.fprefix+'.ens', dtype=str, encoding='utf8')
    # Make a dictionary containing the variables and their uncertainty
    global EnsInputs
    EnsInputs = {}
    for i in range(len(EnsData)):
        #temp = EnsData[i]
        if len(EnsData) > 2:
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
        if item not in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWcs', 'SWvA', 'FRB', 'FRBscale', 'CMEvExp', 'SWn', 'SWv']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
    for item in EnsInputs.keys():
        if item in ['SWCd', 'SWn', 'SWv', 'SWB', 'SWcs', 'SWvA', 'FRB', 'FRBscale', 'CMEvExp', 'SWn', 'SWv']:
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
    if 'SWcs' in input_values: CME.cs = float(input_values['SWcs'])
    if 'SWvA' in input_values: CME.vA = float(input_values['SWvA'])
    if 'FRB' in input_values: CME.B0 = float(input_values['FRB'])
    if 'FRBscale' in input_values: CME.Bscale = float(input_values['FRBscale'])
    if 'CMEvExp' in input_values: CME.vExp = float(input_values['CMEvExp'])
    # add changes to non ForeCAT things onto the CME object
    for item in EnsInputs.keys():
        if item == 'SWCd':
            CME.Cd = np.random.normal(loc=float(input_values['SWCd']), scale=EnsInputs['SWCd'])
            if CME.Cd <0: CME.Cd = 0
            outstr += '{:5.3f}'.format(CME.Cd) + ' '
        if item == 'SWn':
            CME.nSW = np.random.normal(loc=float(input_values['SWn']), scale=EnsInputs['SWn'])
            outstr += '{:6.2f}'.format(CME.nSW) + ' '
        if item == 'SWv':
            CME.vSW = np.random.normal(loc=float(input_values['SWv']), scale=EnsInputs['SWv'])
            outstr += '{:6.2f}'.format(CME.vSW) + ' '
        if item == 'SWB':
            CME.BSW = np.random.normal(loc=float(input_values['SWB']), scale=EnsInputs['SWB'])
            outstr += '{:6.2f}'.format(CME.BSW) + ' '
        if item == 'SWcs':
            CME.cs = np.random.normal(loc=float(input_values['SWcs']), scale=EnsInputs['SWcs'])
            outstr += '{:6.2f}'.format(CME.cs) + ' '
        if item == 'SWvA':
            CME.vA = np.random.normal(loc=float(input_values['SWvA']), scale=EnsInputs['SWvA'])
            outstr += '{:6.2f}'.format(CME.vA) + ' '
        if item == 'FRB':
            CME.B0 = np.random.normal(loc=float(input_values['FRB']), scale=EnsInputs['FRB'])
            outstr += '{:5.2f}'.format(CME.B0) + ' '
        if item == 'FRBscale':
            CME.Bscale = np.random.normal(loc=float(input_values['FRBscale']), scale=EnsInputs['FRBscale'])
            outstr += '{:5.2f}'.format(CME.Bscale) + ' '
        if item == 'FRtau':
            CME.tau = np.random.normal(loc=float(input_values['FRtau']), scale=EnsInputs['FRtau'])
            outstr += '{:5.2f}'.format(CME.tau) + ' '
        if item == 'FRCnm':
            CME.cnm = np.random.normal(loc=float(input_values['FRCnm']), scale=EnsInputs['FRCnm'])
            outstr += '{:5.2f}'.format(CME.cnm) + ' '
        if item == 'CMEvExp':
            CME.vExp = np.random.normal(loc=float(input_values['CMEvExp']), scale=EnsInputs['CMEvExp'])
            outstr += '{:6.2f}'.format(CME.vExp) + ' '
    ensembleFile.write(outstr+'\n')  
    CME.vs[3] = CME.vExp    
    CME.vs[2] = CME.vs[0] - CME.vs[3]
    CME.vs[1] = CME.vs[2]*np.tan(CME.AW)
    CME.vs[4] = CME.vs[3]/CME.deltaCS
    CME.vs[6] = CME.vs[1] - CME.vs[4]
        
    return CME
                        
def goForeCAT():
    # ForeCAT portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|

    # Open a file to save the ForeCAT output
    ForeCATfile = open(Dir+'/ForeCATresults'+thisName+'.dat', 'w')

    # init ForeCAT gives input params
    # CME_params = [shapeA, shapeB, rstart]
    # init_pos = [ilat, ilon, tilt]    
    global iparams, ipos
    ipos, rmax = initForeCAT(input_values)

    CME = initCME([FC.deltaAx, FC.deltaCS, FC.rstart], ipos)
    # add any ANTEATR/FIDO params to the seed case (genEnsMem will add for other cases)
    if 'SWCd' in input_values: CME.Cd = float(input_values['SWCd'])
    if 'FRB' in input_values: CME.B0 = float(input_values['FRB'])
    if 'CMEvExp' in input_values: CME.vExp = float(input_values['CMEvExp'])
    if 'FRBscale'  in input_values: CME.Bscale = float(input_values['FRBscale'])
    if 'FRtau'  in input_values: CME.tau = float(input_values['FRtau'])
    if 'FRCnm'  in input_values: CME.cnm = float(input_values['FRCnm'])
    
    for i in range(nRuns):
        
        print('Running ForeCAT simulation '+str(i+1)+' of '+str(nRuns))
    
        # Make a new ensemble member
        if i > 0:
            CME = genEnsMem(runnum = i)

        # Run ForeCAT
        CME, path = runForeCAT(CME, rmax, path=True)
        
        # add a few useful 1 AU SW parameters to the CME object
        HCSdist = calc_dist(CME.cone[1,1],CME.cone[1,2])
        SWnv = calc_SW(213,HCSdist)
        nSW, vSW = SWnv[0]/1.6727e-24, SWnv[1]/1e5
        CME.nSW = nSW 
        CME.vSW = vSW 
        # get B
        latID = int(CME.cone[1,1]*2 + 179)
        lonID = int(CME.cone[1,2]*2)%720
        fullBvec = ForceFields.B_high[-1,latID,lonID]
        inorout = np.sign(np.dot(CME.rhat,fullBvec[:3])/fullBvec[3])
        CME.BSW = inorout*fullBvec[3] * (rmax/213)**2 *1e5
        # calc alfven speed
        CME.vA = np.abs(CME.BSW/1e5) / np.sqrt(4*3.14159 * CME.nSW * 1.67e-24) / 1e5
        # Save the CME in the array
        CMEarray.append(CME)
        # Write the simulation results to a file
        for j in range(len(path[0,:])):
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = [path[0,j], path[1,j], path[2,j], path[3,j], path[4,j], path[5,j], path[6,j], path[7,j], path[8,j], path[9,j], path[10,j]]
            for iii in outstuff:
                outprint = outprint +'{:4.2f}'.format(iii) + ' '
            ForeCATfile.write(outprint+'\n')
        
    ForeCATfile.close()
    

def makeCMEarray():
    # Called if FC not ran
    global ipos
    ipos, rmax = initForeCAT(input_values)
    # wipe the FC functions to constants, assume we're above height of variations
    FC.user_mass = lambda x: float(input_values['CMEM']) * 1e15
    FC.user_exp  = lambda x: float(input_values['CMEAW'])
    FC.AWratio   = lambda x: float(input_values['CMEAWp']) / float(input_values['CMEAW'])
    # initiate the CME
    CME = initCME([FC.deltaAx, FC.deltaCS, FC.rstart], ipos)
    # add non ForeCAT vars
    if 'SWCd' in input_values: CME.Cd = float(input_values['SWCd'])
    if 'FRB' in input_values: CME.B0 = float(input_values['FRB'])
    if 'CMEvExp' in input_values: 
        CME.vExp = float(input_values['CMEvExp'])
        CME.vs[3] = CME.vExp*1e5
    if 'FRBscale'  in input_values: CME.Bscale = float(input_values['FRBscale'])
    if 'FRtau'  in input_values: CME.tau = float(input_values['FRtau'])
    if 'FRCnm'  in input_values: CME.cnm = float(input_values['FRCnm'])
    # Move to end of ForeCAT distance    
    CME = move2corona(CME, rmax)
    # modify rest of vs = [CMEnose, rEdge, d, br, bp, a, c]
    # CME.vs[0] = CME.v1AU*1e5
    CME.vs[3] = CME.vExp    
    CME.vs[2] = CME.vs[0] - CME.vs[3]
    CME.vs[1] = CME.vs[2]*np.tan(CME.AW)
    CME.vs[4] = CME.vs[3]/CME.deltaCS
    CME.vs[6] = CME.vs[1] - CME.vs[4]
        
    global CMEarray
    CMEarray = [CME]
    
    # Repeat process with ensemble variation
    for i in range(nRuns-1):
        CME = genEnsMem()
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
    
def goANTEATR():
    # ANTEATR portion --------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|

    # Open a file to save the ANTEATR output
    ANTEATRfile = open(Dir+'/ANTEATRresults'+thisName+'.dat', 'w')


    # ANTEATR takes inputs
    # invec = [CMElat, CMElon, CMEtilt, CMEvel0, CMEmass, CMEAW, CMEA, CMEB, vSW, SWrho0, Cd]
    # SatVars0 = [Satlat, Satlon, Satradius] -> technically doesn't have to be Earth!
    # which we can generate from the ForeCAT data

    # Pull in ANTEATR values from the input file
    global SatVars0
    SatVars0, Cd, mySW = processANTinputs(input_values)

    # Assume that the given lon is the time of eruption, need to add in
    # the orbit during the ForeCAT run
    SatRotRate = SatVars0[3]
    SatLons = [SatVars0[1]+SatRotRate*CMEarray[i].t for i in range(nRuns)]

    global ANTsatLons, impactIDs, SWv, SWn, ANTCMErs
    ANTsatLons = {}
    ANTCMErs = {}
    impactIDs = []
        
    for i in range(nRuns):
        myParams = SatVars0
        myParams[1] = SatLons[i]
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr0, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
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
 
        # Check if passed SW variables
        if -9999 in mySW:
            nSW, vSW, BSW = CME.nSW, CME.vSW, np.abs(CME.BSW)
            Bscale, tau, cnm = CME.Bscale, CME.tau, CME.cnm
        else:
            # reset back to defaults before ensembling again
            nSW, vSW, BSW = mySW[0], mySW[1], mySW[2]
            # Add in ensemble variation if desired
            if i > 0:
                if 'SWn' in EnsInputs: nSW = np.random.normal(loc=nSW, scale=EnsInputs['SWn'])
                if 'SWv' in EnsInputs: vSW = np.random.normal(loc=vSW, scale=EnsInputs['SWv'])
                if 'SWB' in EnsInputs: BSW = np.random.normal(loc=BSW, scale=EnsInputs['SWB'])
                if 'CMEBscale' in EnsInputs: Bscale = np.random.normal(loc=Bscale, scale=EnsInputs['CMEBscale'])
                if 'FRtau' in EnsInputs: tau = np.random.normal(loc=tau, scale=EnsInputs['FRtau'])
                if 'FRCnm' in EnsInputs: cnm = np.random.normal(loc=cnm, scale=EnsInputs['FRCnm'])                
                CME.nSW, CME.vSW = nSW, vSW
                CME.Bscale, CME.BSW, CME.tau, CME.cnm = Bscale, BSW, tau, cnm
        
        # Package up invec, run ANTEATR        
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, np.abs(Bscale), nSW, vSW, BSW, Cd, tau, cnm]
        ATresults, Elon, CME.vs, estDur = getAT(invec, myParams, fscales=[0.5,0.5], silent=True)
        
        # Check if miss or hit  
        if ATresults[0][0] != 9999:
            impactIDs.append(i)
        
            # Can add in ForeCAT time to ANTEATR time
            # but usually don't because assume have time stamp of
            # when in outer corona=start of ANTEATR
            TotTime  = ATresults[0][-1]#+CME.t/60./24
            rCME     = ATresults[1][-1]
            CMEvtot  = ATresults[2][-1]
            CMEvbulk = ATresults[3][-1]
            CMEvexp  = ATresults[4][-1]
            CMEAW    = ATresults[5][-1]
            CMEAWp   = ATresults[6][-1]
            deltax   = ATresults[7][-1]
            deltap   = ATresults[8][-1]
            B0       = ATresults[9][-1]
            cnm      = ATresults[10][-1]
            # Store things to pass to FIDO ensembles
            ANTsatLons[i] = Elon # lon at time CME nose is at Earth/sat radius
            # update CME if has that variable
            CME.points[CC.idcent][1,0] = rCME
            CME.AW = CMEAW*dtor
            CME.AWp = CMEAWp*dtor
            CME.deltaAx = deltax
            CME.deltaCS = deltap
            # this B0 is actually Btor, which is what FIDO wants (Btor at center at time of impact)
            CME.B0 = B0 * 1e5 * np.sign(Bscale) * deltap * CME.tau # in nT now
            CME.cnm   = cnm
            CME.vTrans = (rCME-CMEr0)*7e5/(TotTime*24*3600.)
            CME.t = TotTime
            print (str(i)+' Contact after '+"{:.2f}".format(TotTime)+' days with front velocity '+"{:.2f}".format(CMEvtot)+' km/s (expansion velocity ' +"{:.2f}".format(CMEvexp)+' km/s) when nose reaches '+"{:.2f}".format(rCME) + ' Rsun and angular width '+"{:.0f}".format(CMEAW)+' deg and estimated duration '+"{:.0f}".format(estDur)+' hr')
            # prev would take comp of v's in radial direction, took out for now !!!!
            dImp = dObj + datetime.timedelta(days=TotTime)
            print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
            
    
            # For ANTEATR, save CME id number (necessary? matches other file formats)
            # total time, velocity at impact, nose distance, Elon at impact, Elon at 213 Rs
            for j in range(len(ATresults[0])):
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = [ATresults[0,j], ATresults[1,j], ATresults[2,j], ATresults[3,j], ATresults[4,j], ATresults[5,j], ATresults[6,j], ATresults[7,j], ATresults[8,j], ATresults[9,j], ATresults[10,j]]
                for iii in outstuff:
                    outprint = outprint +'{:4.2f}'.format(iii) + ' '
                ANTEATRfile.write(outprint+'\n')
        else:
            print('miss')
    ANTEATRfile.close()    
    
def goFIDO():
    # FIDO portion -----------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|
    # Open a file to save the FIDO output
    FIDOfile = open(Dir+'/FIDOresults'+thisName+'.dat', 'w')

    # Check if adding a sheath or not
    if includeSIT:
        input_values['Add_Sheath'] = 'True'
        FIDO.hasSheath = True
    # Flux rope properties
    CMEB0 = 15. # completely arbitrary number in case not given one
    CMEH  = 1.
    if 'FRB' in input_values: CMEB0 = float(input_values['FRB'])
    if 'CMEH' in input_values: CMEH  = float(input_values['FR_pol'])

    # Check if ANT ran, if not take input from file
    global SatVars0
    CMEstart = 0.   
    if not doANT:
        SatVars0, Cd, swnv = processANTinputs(input_values)
        # For FIDO only, if CMEstart = DOY at arrival then need to not add to date when
        # plotting but in most ensemble cases want to keep wrt time since ForeCAT start
        try:
            CMEstart = float(input_values['CME_start'])
        except:
            CMEstart = 0.           
        
    # Figure out which cases to run - either all or non-misses from ANTEATR
    toRun = range(nRuns)
    if doANT: toRun = impactIDs
    
    # Loop through the CMEs
    for i in toRun:   
        CME = CMEarray[i]
        
        # order is Sat_lon [1], CMElat [2], CMElon [3], CMEtilt [4], CMEAW [5]
        # CMEAWp[6], CMEdeltaAx [7], CMEdeltaCS [8], CMEB0 [9], CMEH [10], tshift [11], 
        # tstart [12], Sat_rad [13], Sat_rot [14], CMEr[15], cnm [16], tau [17]
        inps = np.zeros(18)
        
        # CME parameters from CME object
        inps[2], inps[3] =  CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        inps[4] = CME.tilt
        inps[5], inps[6] = CME.AW*radeg, CME.AWp*radeg
        inps[7], inps[8] = CME.deltaAx, CME.deltaCS
        vs = CME.vs /1e5
        inps[9], inps[10] = CME.B0, CMEH
        # inps[11] is tshift which we keep at zero (only used in GUI)
        if doANT: CMEstart = CME.t
        inps[12] = CMEstart
        inps[15] = CME.points[CC.idcent][1,0]
        inps[16] = CME.cnm
        inps[17] = CME.tau
        
        # Sat parameters
        inps[0], inps[1], = SatVars0[0], SatVars0[1] # lat/lon
        inps[13], inps[14] =  SatVars0[2],  SatVars0[3]  # R/rot
        
         
        if doANT: 
            inps[1] = ANTsatLons[i] 
            vtrans = CME.vTrans
        else:
            # check if is trying to load FIDO CME at 10 Rs because rmax not change in input
            if (inps[13] > 200) & (inps[15] < 25.): inps[15] = 0.95*inps[13]
            if includeSIT:
                vtrans = float(input_values['CMEvTrans'])
        
        # Sheath stuff
        if includeSIT:
            # check if front velocity greater than SW
            if vs[0] > CME.vSW:
                vels = [vs[0]-vs[3], vs[3], vtrans, CME.vSW]
                sheathParams = FIDO.calcSheathInps(CMEstart, vels, CME.nSW, CME.BSW, SatVars0[2], cs=CME.cs, vA=CME.vA)
                actuallySIT = True
            # If not tell FIDO not to do sheath this time
            else:
                actuallySIT = False
                sheathParams = []
                FIDO.hasSheath = False
        else:
            sheathParams = []
            actuallySIT = False
        
            
        flagit = False
        try:
            Bout, tARR, Bsheath, tsheath, radfrac, isHit = FIDO.run_case(inps, sheathParams, vs)
            vProf = FIDO.radfrac2vprofile(radfrac, vs[0]-vs[3], vs[3])
            # turn the sheath back on if we turned it off for a low case
            if includeSIT: FIDO.hasSheath = True
            
        except:
            # sometimes get a miss even though ANTEATR says hit?
            # for now just flag and skip
            flagit = True
        
        if not flagit:    
            # Down sample B resolution
            t_res = 1 # resolution = 60 mins/ t_res
            tARRDS = FIDO.hourify(t_res*tARR, tARR)
            BvecDS = [FIDO.hourify(t_res*tARR,Bout[0][:]), FIDO.hourify(t_res*tARR,Bout[1][:]), FIDO.hourify(t_res*tARR,Bout[2][:]), FIDO.hourify(t_res*tARR,Bout[3][:])]
            vProfDS = FIDO.hourify(t_res*tARR, vProf)
        
            # Write sheath stuff first if needed
            if actuallySIT:
                tsheathDS = FIDO.hourify(t_res*tsheath, tsheath)
                BsheathDS = [FIDO.hourify(t_res*tsheath,Bsheath[0][:]), FIDO.hourify(t_res*tsheath,Bsheath[1][:]), FIDO.hourify(t_res*tsheath,Bsheath[2][:]), FIDO.hourify(t_res*tsheath,Bsheath[3][:])]
                for j in range(len(BsheathDS[0])):
                    outprint = str(i)
                    outprint = outprint.zfill(4) + '   '
                    outstuff = [tsheathDS[j], BsheathDS[3][j], BsheathDS[0][j], BsheathDS[1][j], BsheathDS[2][j], sheathParams[3]]
                    for iii in outstuff:
                        outprint = outprint +'{:6.3f}'.format(iii) + ' '
                    FIDOfile.write(outprint+'\n')
            # Print the flux rope field        
            for j in range(len(BvecDS[0])):
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = [tARRDS[j], BvecDS[3][j], BvecDS[0][j], BvecDS[1][j], BvecDS[2][j], vProfDS[j]]
                for iii in outstuff:
                    outprint = outprint +'{:6.3f}'.format(iii) + ' '
                FIDOfile.write(outprint+'\n')  
        # quick plotting script to check things for ~single case
        # will plot each run individually
        if True:
            cols = ['r', 'b','g', 'k']  
            fig = plt.figure()
            for i2 in range(len(Bout)):
                if actuallySIT: plt.plot(tsheath, Bsheath[i2], linewidth=3, color=cols[i2])
                plt.plot(tARRDS, BvecDS[i2], linewidth=3, color=cols[i2])
            plt.show() 
        print (i, 'min Bz ', np.min(BvecDS[2]), ' (nT)')
    FIDOfile.close()
    
def runOSPREI():
    setupOSPREI()

    if nRuns > 1: setupEns()

    global CMEarray
    CMEarray = []

    if doFC:
        goForeCAT()        
    else:
        # Fill in CME array for ANTEATR or FIDO
        makeCMEarray()
    
    #readMoreInputs()
        
    if doANT: goANTEATR()
    
    if doFIDO: goFIDO()

    if nRuns > 1: ensembleFile.close()

if __name__ == '__main__':
    runOSPREI()
