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
from funcANTEATR import *
from FIDO import *

# Useful constant
global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

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
    global time, satPos, shape, Sat_rot
    suffix = ''
    nRuns  = 1
    models = 'ALL'
    satPos = [0,0]
    shape  = [1,0.15] 
    Sat_rot = 360./365.25
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
    possible_vars = ['ilat', 'ilon', 'tilt', 'Cdperp', 'rstart', 'shapeA', 'shapeB', 'raccel1', 'raccel2', 'vrmin', 'vrmax', 'AWmin', 'AWmax', 'AWr', 'maxM', 'rmaxM', 'shapeB0', 'Cd', 'SSscale','FR_B0', 'CME_vExp', 'CME_v1AU', 'nSW', 'vSW']
    print( 'Determining parameters varied in ensemble...')
    EnsData = np.genfromtxt(FC.fprefix+'.ens', dtype=str)
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
        if item not in ['Cd', 'SSscale', 'FR_B0', 'CME_vExp', 'CME_v1AU', 'nSW', 'vSW']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
    for item in EnsInputs.keys():
        if item in ['Cd', 'SSscale', 'FR_B0', 'CME_vExp', 'CME_v1AU']:
            outstr1 += item + ' '
            outstr2 += str(input_values[item]) + ' '
    ensembleFile.write(outstr1+'\n')
    ensembleFile.write(outstr2+'\n')
            
def genEnsMem(runnum=0):
    # Pass it CME params? ipos?
    # Vary parameters for all models at the same time
    outstr = str(runnum)
    outstr = outstr.zfill(4) + '   '
    flagAccel = False
    new_pos = np.empty(3)
    for item in EnsInputs.keys():
        # Sort out what variable we adjust for each param
        # The lambda functions will auto adjust to new global values in them
        if item == 'ilat':
            new_pos[0] = np.random.normal(loc=float(input_values['ilat']), scale=EnsInputs['ilat'])
            outstr += '{:4.2f}'.format(new_pos[0]) + ' '
        if item == 'ilon':
            new_pos[1] = np.random.normal(loc=float(input_values['ilon']), scale=EnsInputs['ilon'])
            outstr += '{:4.2f}'.format(new_pos[1]) + ' '
        if item == 'tilt':
            new_pos[2] = np.random.normal(loc=float(input_values['tilt']), scale=EnsInputs['tilt'])
            outstr += '{:4.2f}'.format(new_pos[2]) + ' '
        if item == 'Cdperp':
            FC.Cd = np.random.normal(loc=float(input_values['Cdperp']), scale=EnsInputs['Cdperp'])
            # Make sure drag coeff is positive if inappropriate uncertainty is used
            if FC.Cd < 0: FC.Cd = 0
            outstr += '{:4.3f}'.format(FC.Cd) + ' '           
        if item == 'shapeA':
            FC.shapeA = np.random.normal(loc=float(input_values['shapeA']), scale=EnsInputs['shapeA'])
            outstr += '{:5.4f}'.format(FC.shapeA) + ' '
        if item == 'shapeB':
            FC.shapeB = np.random.normal(loc=float(input_values['shapeB']), scale=EnsInputs['shapeB'])
            outstr += '{:5.4f}'.format(FC.shapeB) + ' '
        if item == 'shapeB0':
            FC.shapeB0 = np.random.normal(loc=float(input_values['shapeB0']), scale=EnsInputs['shapeB0'])
            outstr += '{:5.4f}'.format(FC.shapeB0) + ' '
        if item == 'rstart':
            FC.rstart = np.random.normal(loc=float(input_values['rstart']), scale=EnsInputs['rstart'])
            outstr += '{:5.4f}'.format(FC.rstart) + ' '
        if item == 'raccel1':
            FC.rga = np.random.normal(loc=float(input_values['raccel1']), scale=EnsInputs['raccel1'])
            outstr += '{:5.4f}'.format(FC.rga) + ' '
            flagAccel = True
        if item == 'raccel2':
            FC.rap = np.random.normal(loc=float(input_values['raccel2']), scale=EnsInputs['raccel2'])
            outstr += '{:5.4f}'.format(FC.rap) + ' '
            flagAccel = True
        if item == 'vrmin':
            FC.vmin = np.random.normal(loc=float(input_values['vrmin']), scale=EnsInputs['vrmin'])*1e5
            outstr += '{:5.2f}'.format(FC.vmin/1e5) + ' '
            flagAccel = True
        if item == 'vrmax':
            FC.vmax = np.random.normal(loc=float(input_values['vrmax']), scale=EnsInputs['vrmax'])*1e5
            outstr += '{:6.2f}'.format(FC.vmax/1e5) + ' '
            flagAccel = True
        if item == 'AWr':
            FC.awR = np.random.normal(loc=float(input_values['AWr']), scale=EnsInputs['AWr'])
            outstr += '{:4.3f}'.format(FC.awR) + ' '
        if item == 'AWmin':
            FC.aw0 = np.random.normal(loc=float(input_values['AWmin']), scale=EnsInputs['AWmin'])
            outstr += '{:5.2f}'.format(newaw0) + ' '
        if item == 'AWmax':
            FC.awM = np.random.normal(loc=float(input_values['AWmax']), scale=EnsInputs['AWmax'])
            outstr += '{:5.2f}'.format(FC.awM) + ' '
        if item == 'rmaxM':
            FC.rmaxM = np.random.normal(loc=float(input_values['rmaxM']), scale=EnsInputs['rmaxM'])
            outstr += '{:5.2f}'.format(FC.rmaxM) + ' '
        if item == 'maxM':
            FC.max_M = np.random.normal(loc=float(input_values['maxM']), scale=EnsInputs['maxM'])* 1e15
            # Set somewhat arbitrary lower bound, definitely don't want negative mass
            if FC.max_M < 1e14: FC.max_M = 1e14
            outstr += '{:5.2f}'.format(FC.max_M/1e15) + ' '        
    
    if flagAccel:
        FC.a_prop = (FC.vmax**2 - FC.vmin **2) / 2. / (FC.rap - FC.rga)
            
    CME = initCME([FC.shapeA, FC.shapeB, FC.rstart], new_pos)
    # add non ForeCAT vars
    if 'Cd' in input_values: CME.Cd = float(input_values['Cd'])
    if 'FR_B0' in input_values: CME.FR_B0 = float(input_values['FR_B0'])
    if 'CME_vExp' in input_values: CME.vExp = float(input_values['CME_vExp'])
    if 'CME_v1AU' in input_values: CME.v1AU = float(input_values['CME_v1AU'])
    # add non ForeCAT things onto the CME object
    for item in EnsInputs.keys():
        if item == 'Cd':
            CME.Cd = np.random.normal(loc=float(input_values['Cd']), scale=EnsInputs['Cd'])
            if CME.Cd <0: CME.Cd = 0
            outstr += '{:5.3f}'.format(CME.Cd) + ' '
        if item == 'FR_B0':
            CME.FR_B0 = np.random.normal(loc=float(input_values['FR_B0']), scale=EnsInputs['FR_B0'])
            outstr += '{:5.2f}'.format(CME.FR_B0) + ' '
        if item == 'CME_vExp':
            CME.vExp = np.random.normal(loc=float(input_values['CME_vExp']), scale=EnsInputs['CME_vExp'])
            outstr += '{:6.2f}'.format(CME.vExp) + ' '
        if item == 'CME_v1AU':
            CME.v1AU = np.random.normal(loc=float(input_values['CME_v1AU']), scale=EnsInputs['CME_v1AU'])
            outstr += '{:6.2f}'.format(CME.v1AU) + ' '
        if item == 'SSscale':
            newScale = np.random.normal(loc=float(input_values['SSscale']), scale=EnsInputs['SSscale'])
            if newScale > 1.: newScale = 1.
            CME.SSscale = newScale
            outstr += '{:6.3f}'.format(CME.SSscale) + ' '

    ensembleFile.write(outstr+'\n')       
        
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

    CME = initCME([FC.shapeA, FC.shapeB, FC.rstart], ipos)
    # add any ANTEATR/FIDO params to the seed case
    if 'Cd' in input_values: CME.Cd = float(input_values['Cd'])
    if 'FR_B0' in input_values: CME.FR_B0 = float(input_values['FR_B0'])
    if 'CME_vExp' in input_values: CME.vExp = float(input_values['CME_vExp'])
    if 'CME_v1AU' in input_values: CME.v1AU = float(input_values['CME_v1AU'])
    if 'SSscale'  in input_values: CME.SSscale = float(input_values['SSscale'])
    
    for i in range(nRuns):
        print('Running ForeCAT simulation '+str(i+1)+' of '+str(nRuns))
    
        # Make a new ensemble member
        if i > 0:
            CME = genEnsMem(runnum = i)

        # Run ForeCAT
        CME, path = runForeCAT(CME, rmax, path=True)
        
        # Save the CME in the array
        CMEarray.append(CME)

        # Write the simulation results to a file
        # Only saving run number, nose dist, lat, lon, tilt
        # since vr, AW can be reconstructed from empirical equation
        for j in range(len(path[0,:])):
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = [path[0,j], path[1,j], path[2,j], path[3,j], path[4,j], path[5,j], path[6,j], path[7,j]]
            for iii in outstuff:
                outprint = outprint +'{:4.2f}'.format(iii) + ' '
            ForeCATfile.write(outprint+'\n')
        
    ForeCATfile.close()

def makeCMEarray():
    global ipos
    ipos, rmax = initForeCAT(input_values)
    # initiate the CME
    CME = initCME([FC.shapeA, FC.shapeB, FC.rstart], ipos)
    # add non ForeCAT vars
    if 'Cd' in input_values: CME.Cd = float(input_values['Cd'])
    if 'FR_B0' in input_values: CME.FR_B0 = float(input_values['FR_B0'])
    if 'CME_vExp' in input_values: CME.vExp = float(input_values['CME_vExp'])
    if 'CME_v1AU' in input_values: CME.v1AU = float(input_values['CME_v1AU'])
    if 'SSscale'  in input_values: CME.SSscale = float(input_values['SSscale'])
    # Move to end of ForeCAT distance    
    CME = move2corona(CME, rmax)
    
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
    CME.ang_width = FC.awM * dtor
    CME.shape_ratios[0] = FC.shapeA
    CME.shape_ratios[1] = FC.shapeB
    # Calculate shape
    CME.shape[2] = rmax * np.tan(CME.ang_width) / (1 + CME.shape_ratios[1] 	+ (CME.shape_ratios[0]+ CME.shape_ratios[1]) * np.tan(CME.ang_width))
    CME.shape[0] = CME.shape_ratios[0] * CME.shape[2]
    CME.shape[1] = CME.shape_ratios[1] * CME.shape[2]
	# calc new center/cone pos using new nose position and shape
    CME.cone[1,0] = rmax
    CME.cone[1,0] += - CME.shape[0] - CME.shape[1] # remove a+b from radial distance
    CME.cone[0,:] = FC.SPH2CART(CME.cone[1,:]) 
    # set vr, don't really care about correct XYZ orientation, ANTEATR just needs magnitude
    CME.vels[0,:] = CME.vels[0,:]*0
    CME.vels[0,0] = float(input_values['vrmax']) * 1e5
	# determine new mass
    CME.M = float(input_values['maxM'])*1e15

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
    global ANTinputs, SatVars0
    ANTinputs = getANTinputs(allinputs)
    SatVars0, Cd, mySW = processANTinputs(ANTinputs)

    # Assume that the given lon is the time of eruption, need to add in
    # the orbit during the ForeCAT run
    SatRotRate = SatVars0[3]
    SatLons = [SatVars0[1]+SatRotRate*CMEarray[i].t for i in range(nRuns)]

    global ANTvBulks, ANTvExps, ANTAWs, ANTts, ANTsatLons, impactIDs
    ANTvBulks = {}
    ANTvExps = {}
    ANTts  = {}
    ANTsatLons = {}
    ANTAWs  = {}
    impactIDs = []
    
    # Check if we were give SW values or if using background from ForeCAT
    if -9999 not in mySW:  nSW, vSW = mySW[0], mySW[1]
    
    for i in range(nRuns):
        myParams = SatVars0
        myParams[1] = SatLons[i]
        # CME parameters from CME object
        CME = CMEarray[i]
        # CME position
        CMEr, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        # Tilt 
        tilt = CME.tilt
        # Calculate vr
        vr = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
        # Mass
        mass = CME.M/1e15
        # CME shape
        cmeAW = CME.ang_width*radeg
        cmeA, cmeB = CME.shape_ratios[0], CME.shape_ratios[1]

        # Check if passed SW variables
        if -9999 in mySW:
            # Get solar wind values using the ForeCAT model
            HCSdist = calc_dist(CME.cone[1,1],CME.cone[1,2])
            SWnv = calc_SW(myParams[2],HCSdist)
            nSW, vSW = SWnv[0]/1.6727e-24, SWnv[1]/1e5
        else:
            # reset back to defaults before ensembling again
            nSW, vSW = mySW[0], mySW[1]
        # Add in ensemble variation if desired
        if i > 0:
            if 'nSW' in EnsInputs: nSW = np.random.normal(loc=nSW, scale=EnsInputs['nSW'])
            if 'vSW' in EnsInputs: vSW = np.random.normal(loc=vSW, scale=EnsInputs['vSW'])
        
        # Package up invec, run ANTEATR
        invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeA, cmeB, vSW, nSW, Cd]
        ATresults, Elon, ElonEr = getAT(invec,CMEr,myParams, silent=True, SSscale=CME.SSscale)
        
        # Check if miss or hit  
        if ATresults[0][0] != 9999:
            impactIDs.append(i)
        
            # Can add in ForeCAT time to ANTEATR time
            # but usually don't because assume have time stamp of
            # when in outer corona=start of ANTEATR
            TotTime = ATresults[0][-1]#+CME.t/60./24
            rCME = ATresults[1][-1]
            CMEvtot = ATresults[2][-1]
            CMEvbulk = ATresults[3][-1]
            CMEvexp  = ATresults[4][-1]
            CMEAW = ATresults[5][-1]
            
            # Store things to pass to FIDO ensembles
            ANTsatLons[i] = ElonEr # lon at time CME nose is at Earth/sat radius
            ANTvBulks[i] = CMEvbulk#ATresults[1] 
            ANTvExps[i]  = CMEvexp
            ANTAWs[i] = CMEAW
            ANTts[i] = TotTime
                        
            print (str(i)+' Contact after '+"{:.2f}".format(TotTime)+' days with front velocity '+"{:.2f}".format(CMEvtot)+' km/s (expansion velocity ' +"{:.2f}".format(CMEvexp)+' km/s) when nose reaches '+"{:.2f}".format(rCME) + ' Rsun and angular width '+"{:.0f}".format(CMEAW)+' deg')
            dImp = dObj + datetime.timedelta(days=TotTime)
            print ('   Impact at '+dImp.strftime('%Y %b %d %H:%M '))
            
    
            # For ANTEATR, save CME id number (necessary? matches other file formats)
            # total time, velocity at impact, nose distance, Elon at impact, Elon at 213 Rs
            for j in range(len(ATresults[0])):
                outprint = str(i)
                outprint = outprint.zfill(4) + '   '
                outstuff = [ATresults[0,j], ATresults[1,j], ATresults[2,j], ATresults[3,j], ATresults[4,j], ATresults[5,j]]
                for iii in outstuff:
                    outprint = outprint +'{:4.2f}'.format(iii) + ' '
                ANTEATRfile.write(outprint+'\n')
            #outprint = str(i)
            #outprint = outprint.zfill(4) + '   '
            #for iii in ATresults:
            #        outprint = outprint +'{:4.2f}'.format(iii) + ' '
            
            #outprint = outprint + '{:4.2f}'.format(nSW) + ' ' + '{:4.2f}'.format(vSW)
            #ANTEATRfile.write(outprint+'\n')
    ANTEATRfile.close()    
    
def goFIDO():
    # FIDO portion -----------------------------------------------------|
    # ------------------------------------------------------------------|
    # ------------------------------------------------------------------|
    # Open a file to save the ANTEATR output
    FIDOfile = open(Dir+'/FIDOresults'+thisName+'.dat', 'w')

    # Essentially just copying the relevant parts of runFIDO()
    # First set up additional parameters and set to not launch GUI
    # take allinputs and add relevant to input_values
    input_valuesFIDO = read_more_inputs(allinputs, input_values)
    input_valuesFIDO['Launch_GUI'] = 'False'
    input_valuesFIDO['No_Plot'] = 'True' # will do our own
    setupOptions(input_valuesFIDO, silent=True)
    #CMEB0 = float(input_valuesFIDO['FR_B0'])
    CMEH  = float(input_valuesFIDO['FR_pol'])
    # Need to sort out switching between ANTEATR t or given t
    # check if ant ran, if not def take input from file
    global SatVars0
    if not doANT:
        ANTinputs = getANTinputs(allinputs)
        SatVars0, Cd, swnv = processANTinputs(ANTinputs)
        try:
            CMEstart = float(input_values['CME_start'])
        except:
            CMEstart = 0.            
    try:
        CMEstop = float(input_valuesFIDO['CME_stop'])
    except:
        CMEstop = 0. # it will ignore this value, but needs to be passed something
    #try:
    #    vexp = float(input_valuesFIDO['CME_vExp'])
    #except:
    #    vexp = 0.  # set to zero if not given, will ignore if not using vexp expansion
        
    # Figure out which cases to run - either all or non-misses from ANTEATR
    toRun = range(nRuns)
    if doANT: toRun = impactIDs

    for i in toRun:    
        # set up input array   
        # start with parameters that either come from ANTEATR
        # or the input file
        if doANT: 
            SatLon = ANTsatLons[i]
            CMEstart = ANTts[i]
            CME_v1AU = ANTvBulks[i]
            vexp = ANTvExps[i]
            cmeAW = ANTAWs[i]
        else:
            SatLon = SatVars0[1]
            CME_v1AU = float(input_valuesFIDO['CME_v1AU'])
            vexp  = CME.vExp      
            cmeAW = CME.ang_width*radeg

        # other CME parameters from CME object
        CME = CMEarray[i]
        CMEr, CMElat, CMElon = CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], CME.points[CC.idcent][1,2]
        tilt = CME.tilt
        # Mass
        mass = CME.M/1e15
        # CME shape
        cmeA, cmeB = CME.shape_ratios[0], CME.shape_ratios[1]
        CMEB0 = CME.FR_B0
       
        # order is Sat_lat [0], Sat_lon0 [1], CMElat [2], CMElon [3], CMEtilt [4], CMEAW [5]
        # CMESRA [6], CMESRB [7], CMEvr [8], CMEB0 [9], CMEH [10], tshift [11], start [12], end [13], vexp[14], 
        # Sat_rad [15], Sat_rot [16]
        inps =[SatVars0[0], SatLon, CMElat, CMElon, tilt,  cmeAW, cmeA, cmeB, CME_v1AU, CMEB0, CMEH, 0., CMEstart, 0., vexp, SatVars0[2], SatVars0[3]/60.]
        # sheath params are empty array, not using for now
        Bout, tARR, radfrac = run_case(inps, [])
        vProf = radfrac2vprofile(radfrac, CME_v1AU, vexp)
        
        # Down sample B resolution
        t_res = 1 # resolution = 60 mins/ t_res
        tARRDS = hourify(t_res*tARR, tARR)
        BvecDS = [hourify(t_res*tARR,Bout[0][:]), hourify(t_res*tARR,Bout[1][:]), hourify(t_res*tARR,Bout[2][:]), hourify(t_res*tARR,Bout[3][:])]
        vProfDS = hourify(t_res*tARR, vProf)
        for j in range(len(BvecDS[0])):
            outprint = str(i)
            outprint = outprint.zfill(4) + '   '
            outstuff = [tARRDS[j], BvecDS[3][j], BvecDS[0][j], BvecDS[1][j], BvecDS[2][j], vProfDS[j]]
            for iii in outstuff:
                outprint = outprint +'{:6.3f}'.format(iii) + ' '
            FIDOfile.write(outprint+'\n')    
            
        # quick plotting script to check things for ~single case
        # will plot each run individually
        #cols = ['k', 'b','r', 'k']  
        #fig = plt.figure()
        #for i2 in range(len(Bout)):
        #    plt.plot(tARRDS, BvecDS[i2], linewidth=3, color=cols[i2])
        #plt.show()
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
        
    if doANT: goANTEATR()
    
    if doFIDO: goFIDO()

    if nRuns > 1: ensembleFile.close()

runOSPREI()
