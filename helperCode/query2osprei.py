import numpy as np
import os, sys
myPaths = np.genfromtxt('myPaths.txt', dtype=str)
mainpath = myPaths[0,1]
magpath  = myPaths[2,1]
sys.path.append(os.path.abspath(mainpath+'helperCode')) 
from getSatLoc import getSatLoc, getEarthClon 
import datetime
import mag2PFSS as m2P


    
def preProcessIt(inputs):    
    # ------------------------------------------------
    # ----- Make satellite path and get init pos -----
    # ------------------------------------------------
    # Need to do this early on bc of inputs that depend on satellite distance
    sat = inputs['satellite'] # can be PSP, STA, STB, Earth, SolO
    trajDir = myPaths[3,1]
    if sat == 'Earth':
        satfile = trajDir + 'planets/earth.lst'
        satName = 'earth'
    elif sat == 'PSP':
        satfile = trajDir + 'spacecraft/psp.lst'
        satName = 'psp'
    elif sat == 'SolO':
        satfile = trajDir + 'spacecraft/solo.lst'
        satName = 'solo'
    elif sat == 'STA':
        satfile = trajDir + 'spacecraft/stereoa.lst'
        satName = 'stereoa'
    elif sat == 'STB':
        satfile = trajDir + 'spacecraft/stereob.lst'
        satName = 'stereob'
    # Could add more locations here as long as in helioweb files
    
    # Need to get correct start time
    hasTime = True
    dateSTR =  inputs['date']
    timeSTR = inputs['time']
    fmt = '%Y%m%d %H:%M'
    startDT = datetime.datetime.strptime(dateSTR+' '+timeSTR, fmt)

    # Pad it by 30 days which should be good for any normal case
    # to have smooth satellite profile
    filest = startDT - datetime.timedelta(days=30)
    fileend = startDT + datetime.timedelta(days=30)
    stSTR = filest.strftime('%Y/%m/%d %H:%M')
    endSTR = fileend.strftime('%Y/%m/%d %H:%M')
    
    # Create .sat file -> satName_temp.sat
    pathName = getSatLoc(satName, stSTR, endSTR, outName='temp', fracYr=False, satpath0=trajDir)
    # Get Carrington longitude of Earth, needed if running ForeCAT as ref point to convert sat StonyLon 
    eCLon =  getEarthClon(startDT)
    
    # Get the initial sat values
    satData = np.genfromtxt(satName+'_temp.sat', dtype=str, skip_header=1)
    dayidx = np.where(satData[:,0] == startDT.strftime('%Y-%m-%d'))[0][0]
    fracDay = (startDT -  datetime.datetime.strptime(dateSTR, '%Y%m%d')).total_seconds() / (24.*3600)
    # Linearly interpret between days to precise start time
    rA, latA, lonA = float(satData[dayidx,2]), float(satData[dayidx,3]), float(satData[dayidx,4])
    rB, latB, lonB = float(satData[dayidx+1,2]), float(satData[dayidx+1,3]), float(satData[dayidx+1,4])

    # Get initial sat location to use for inputs
    satr = rA * (1-fracDay) + rB * fracDay
    satLat = latA * (1-fracDay) + latB * fracDay
    satLon = lonA * (1-fracDay) + lonB * fracDay
    satLonCar = satLon + eCLon
      


    # ------------------------------------------------
    # -------- Process Query file to Inputs ----------
    # ------------------------------------------------
    # Check that all inputs are explicity listed whether defaults or not
    # Took out FCmagname, satPath/satLon, bc not query output, will be derived below
    allInNames = ['date', 'time', 'suffix', 'nRuns', 'models', 'CMElat', 'CMElon', 'CMEtilt', 'CMEyaw', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEr',  'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'SunRss', 'PFSSscale', 'IVDf', 'SWCd', 'SWCdp', 'SWR', 'SWn', 'SWv', 'SWB', 'SWT', 'FRtau', 'FRCnm', 'FRpol', 'Gamma', 'doPUP',  'doMH', 'MHarea', 'MHdist', 'obsFRstart', 'obsFRend', 'obsShstart', 'simYaw', 'FRB', 'FRT']

    # Dictionary with defaults for the simple parameters
    # Some of these aren't accesible by query form, but might want
    # available in special custom cases?
    defaults = {'nRuns':'1', 'CMEyaw':'0', 'CMEdelAx':'0.75', 'CMEdelCS':'1.', 'FCrmax':'21.5', 'FCraccel1':'1.3', 'FCraccel2':'10', 'FCvrmin':'50', 'FCAWmin':'5', 'FCAWr':'1',  'SunRss':'2.5', 'PFSSscale':'1.0', 'IVDf':'0.5',  'SWCd':'1.', 'SWCdp':'1.', 'FRtau':'1.', 'FRCnm':'1.927', 'Gamma':'1.33', 'doPUP':'False', 'doMH':'False','simYaw':'False'}
    SW1AUdefaults = {'SWn':'7.5', 'SWv':'350', 'SWB':'6.9', 'SWT':'75000'}

    # List of parameters that have no default - must be specified in query
    noDef = ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'date', 'time', 'suffix', 'nRuns', 'models']

    # List of parameters that have more complicated defaults that can't 
    # just be pulled from a dictionary
    funDef = ['CMEAWp', 'CMEr', 'CMEM', 'FCrmaxM', 'FRB', 'FRpol', 'FRT', 'SatLat', 'SatLon', 'SatR', 'SWn', 'SWv', 'SWB', 'SWT', 'SWR' , 'MHarea', 'MHdist', 'obsFRstart', 'obsFRend', 'obsShstart']

    # Record of inputs that OSPREI technically accepts but not avail through query
    # and don't expect needed in even custom cases. 
    # And things like uploading obs data for comparison
    # 'FCtprint', 'L0', 'saveData', 'printData', 'useFCSW', 'IVDf1', 'IVDf2', 'ObsDataFile', 'SWfile',  'SatRot', 'SunRotRate','SunR'
    # 'FCNtor', 'FCNpol', 'FCRotCME', 'FCtprint',  'flagScales',

    # Set up dictionary to hold all the input values
    allIns = {}

    # Set up satellite things that we have already calculated
    allIns['satPath'] = pathName # Need to add directory to this?
    allIns['SatR'] = str(satr * 215.) # convert to Rs
    allIns['SatLat']  = str(satLat)
    # set SatLon to Carrington if using ForeCAT, otherwise stony
    if inputs['models'] in ['All', 'FC']:
        allIns['SatLon']  = str(satLonCar)
    else:
        allIns['SatLon']  = str(satLon)
    
    # Check if doing Earth or a satellite and set isSat accordingly
    if sat == 'earth':
        allIns['isSat'] = 'False'
    else:
        allIns['isSat'] = 'True'
    
    # Run through full list of names and check if value given or if need 
    # to add in the default value
    for name in allInNames:
        #print (name)
        # Use the given value if it exists
        if name in inputs.keys():
            allIns[name] = inputs[name]
        
        # Use the default value for simple cases
        elif name in defaults.keys():
            allIns[name] = defaults[name]    
        
        # Quit if missing critical input (shouldn't be possible from query requirements)
        elif name in noDef:
            sys.exit('Error in inputs, missing critical value for '+name)
        
        # Set up the more complicated defaults     
        elif name in funDef:
            # Angular Width
            if name == 'CMEAWp':
                allIns[name] = '{:.2f}'.format(float(allIns['CMEAW']) / 3)
            
            # Starting radial distance
            elif name == 'CMEr':
                if allIns['models'] in ['All', 'FC']:
                    allIns[name] = '1.1'
                elif allIns['models'] in ['ANT', 'IP']:
                    allIns[name] = '21.5'
                elif allIns['models'] == 'FIDO':
                    # Need to set to satellite R
                    allIns[name] = allIns['SatR']
                
            # CME mass:
            elif name == 'CMEM':
                # LLAMACoRe v1.0 relation
                allIns[name] = 0.010 * float(allIns['CMEvr']) + 0.16
        
            # Distance of maximum mass
            elif name == 'FCrmaxM':
                allIns[name] = allIns['FCrmax']
                       
            # Flux rope B
            elif name == 'FRB':   
                dtor = 3.14159 / 180
                if allIns['models'] in ['All', 'FC']:
                    rFront = float(allIns['FCrmax']) * 7e10
                else:
                    rFront = float(allIns['CMEr']) * 7e10
                AW = float(allIns['CMEAW']) * dtor
                AWp = float(allIns['CMEAWp']) * dtor
                delAx = float(allIns['CMEdelAx'])
                delCS = float(allIns['CMEdelCS'])
                v = float(allIns['CMEvr'])
                Cnm = float(allIns['FRCnm'])
                tau = float(allIns['FRtau'])
                mass = float(allIns['CMEM'])
            
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
            
                KE = 0.5 * mass*1e15 * (v*1e5)**2 /1e31
                phiflux = np.power(10, np.log10(KE / 0.19) / 1.87)*1e21
                B0 = phiflux * Cnm * (delCS**2 + 1) / avgR / Ltot / delCS**2 *1e5
                Bcent = delCS * tau * B0
                allIns[name] = str(Bcent)
            
            # Flux rope polarity
            elif name == 'FRpol':
                if float(allIns['CMElat']) > 0:
                    allIns[name] = '-1'
                else:
                    allIns[name] = '1'
                
            # Flux rope temperature - calculate from T
            elif name == 'FRT':
                v = float(allIns['CMEvr'])
                if allIns['models'] in ['All', 'FC']:
                    rFront = float(allIns['FCrmax']) * 7e10
                else:
                    rFront = float(allIns['CMEr']) * 7e10
                vSheath = 0.129 * v + 376
                vIS = (vSheath + 51.73) / 1.175
                vExp = 0.175 * vIS -51.73
                logTIS = 3.07e-3 * vIS +3.65
                FRT = np.power(10, logTIS) * np.power(215*7e10/rFront, 0.7)
                allIns['FRT'] = str(FRT)
        
            # Solar wind reference distance
            elif name == 'SWR':
                # if not given assume equal to sat distance
                allIns[name] = allIns['SatR']
        
            # Solar wind parameters - scale based on 1 AU defaults
            elif name in ['SWn', 'SWv', 'SWB', 'SWT']:
                # if within 5Rs of 1 AU then just set to Earth/1 AU defaults
                if np.abs(float(allIns['SWR'])- 215.) < 5.: 
                    allIns[name] = SW1AUdefaults[name]
                else:
                    # else scale to actual r
                    SWr = float(allIns['SWR'])
                    RoverR = SWr / 215.
                    if name == 'SWn':
                        allIns[name] = str(float(SW1AUdefaults[name]) * RoverR**2)
                    elif name == 'SWv':
                        allIns[name] = SW1AUdefaults[name]
                    elif name == 'SWB':
                        # assume normal solar rotation rate
                        BphiBr1AU =  2.7e-6 * 215 * 7e5  / float(allIns['SWv'])
                        Br1AU = float(SW1AUdefaults[name]) / np.sqrt(1 + BphiBr1AU**2)
                        Br = Br1AU / RoverR**2
                        Bphi = Br * (RoverR)**2 * 2.7e-6 * SWr * 7e5 / float(allIns['SWv']) 
                        B = np.sqrt(Br**2 + Bphi**2)
                        allIns[name] = str(B) 
                    
                    elif name == 'SWT':
                        allIns[name] = str(float(SW1AUdefaults[name]) / RoverR)
        
            # MEOW-HiSS parameters
            # Don't have default for these parameters, but check that doMH
            # is False if they aren't included.
            elif name in [ 'MHarea', 'MHdist']:
                if allIns['doMH'] == 'True':
                    sys.exit('Error in inputs, missing critical value for '+name+' or turn off doMH')
            
            # Things it's ok to run without (obs in situ times) 
            elif name in ['obsFRstart', 'obsFRend', 'obsShstart']:
                pass
            
        #if name in allIns.keys():    
        #    print(name + ': ' + allIns[name])




    # ------------------------------------------------
    # ---- Setting up ForeCAT PFSS Magnetic Field ----
    # ------------------------------------------------
    # Check if we are running ForeCAT need the PFSS
    runPFSS = False
    if inputs['models'] in ['All', 'FC']:
        runPFSS = True

    #runPFSS = False # rm when no longer testing!!!     
    if runPFSS:
        # Grab the appropriate magnetogram - this needs to be changed to CCMC specifics
        magObs = inputs['Magnetogram']
        magFile = inputs['MagFile']
        rSS = float(allIns['SunRss'])
        magName = magFile.replace('.fits','') + '_Rss' + allIns['SunRss']
    
        #magName = 'HMI2253synop' # for testing, comment out
        # check if the file already exists before running
        if not os.path.isfile(magpath+ 'PFSS_'+magName+'b3.pkl'):
            # set isSinLat based on magnetogram source
            # Gong is not sinLat, don't know about WSA setting
            isSinLat = False
            if magObs in ['HMI']:
                isSinLat = True
        
            # Adjust magnetogram to Carrington 0-360 longitude (most have newest data on left)
    
            # calculate the harmonic coeffs
            # make it ignore div by zero in logs, complains about unused part of array
            np.seterr(divide = 'ignore', invalid = 'ignore') 
            coeff_file = m2P.harmonics(magObs, '', 90, nameIn=magFile, nameOut='tempHarmonics.dat')   
    
            # make the PFSS pickles
            pickle_file = m2P.makedapickle(coeff_file, magObs, '', 90, rSS, nameOut=magName)
    
            # get the distance from the HCS
            m2P.calcHCSdist(pickle_file)   
        else:
            pickle_file = 'PFSS_'+magName  
    
    #print (Sd)
    # For testing
    #pickle_file = 'PFSS_temp'
    #runPFSS = True

    # Determine if ensembling
    if int(allIns['nRuns']) > 1:
        f2 = open('runScript_'+allIns['suffix']+'.ens', 'w')
        if 'delta_CMElat' in inputs.keys():
            f2.write('CMElat: '+inputs['delta_CMElat'] + '\n')
        if 'delta_CMElon' in inputs.keys():
            f2.write('CMElon: '+inputs['delta_CMElon'] + '\n')
        if 'delta_CMEtilt' in inputs.keys():
            f2.write('CMEtilt: '+inputs['delta_CMEtilt'] + '\n')
        if 'delta_CMEvr' in inputs.keys():
            f2.write('CMEvr: '+inputs['delta_CMEvr'] + '\n')
        if 'delta_CMEAW' in inputs.keys():
            f2.write('CMEAW: '+inputs['delta_CMEAW'] + '\n')
        if 'delta_CMEAWp' in inputs.keys():
            f2.write('CMEAWp: '+inputs['delta_CMEAWp'] + '\n')
        if 'delta_CMEdelAx' in inputs.keys():
            f2.write('CMEdelAx: '+inputs['delta_CMEdelAx'] + '\n')
        if 'delta_CMEdelCS' in inputs.keys():
            f2.write('CMEdelCS: '+inputs['delta_CMEdelCS'] + '\n')
        if 'delta_CMEM' in inputs.keys():
            f2.write('CMEM: '+inputs['delta_CMEM'] + '\n')   
        f2.close()

    # Reprint input file with everything explicitly listed
    f1 = open('runScript_'+allIns['suffix']+'.txt', 'w')
    # need to add PFSS pickle to inputfile
    if runPFSS:
        if pickle_file[:5] == 'PFSS_':
            pickle_file = pickle_file[5:]
        f1.write('FCmagname:   '+ pickle_file + '\n')
        f1.write('Magnetogram:   '+ inputs['Magnetogram'] + '\n')
    # write everything else to file    
    for key in allIns.keys():
        f1.write(key+':   '+allIns[key] + '\n')
    # add in satellite
    f1.write('satellite:   '+ inputs['satellite'] + '\n')
    f1.close()




if __name__ == '__main__':
    # Takes the output from the CCMC query form and sets up everything
    # needed to run OSPREI
    input_file = sys.argv[1]

    # ------------------------------------------------
    # ---------- Read in the query files -------------
    # ------------------------------------------------
    # Set to skip top line "Control File"
    data = np.genfromtxt(input_file, dtype=str, skip_header=1)
    # Turn into a dictionary and rm : from names
    inputs = {}
    for i in range(len(data[:,0])):
        inputs[data[i,0][:-1]] = data[i,1]
    
    preProcessIt(inputs)

