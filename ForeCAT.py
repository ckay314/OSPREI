import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import os
# give it the path wherever the ForeCAT files are stored
sys.path.append(os.path.abspath('/Users/ckay/OSPREI/code'))
import ForeCAT_functions as FC
import CME_class as CC
import ForceFields as FF

def initForeCAT(input_values, skipPkl=False):
    #---------------------------------------------------------------------------------------|
    # Read in the filename from the command line and load the parameters -------------------|
    #---------------------------------------------------------------------------------------|
    global tprint, Ntor, Npol
    ipos, rmax, tprint, Ntor, Npol = FC.getInps(input_values)

    #---------------------------------------------------------------------------------------|
    # Simulation set up --------------------------------------------------------------------|
    #---------------------------------------------------------------------------------------|
    if not skipPkl: # option to skip this part if just using getInps for FIDO only
        # Initialize magnetic field data
        FF.init_CPU(FC.CR, Ntor, Npol)

        # Initialize magnetic field and distance pickles
        FC.initdefpickle(FC.CR)
    return ipos, rmax

def initCME(CME_params, ipos):
    # Initialize CME
    CME = CC.CME(ipos, CME_params, Ntor, Npol, FC.user_vr, FC.user_exp, FC.user_mass, FC.AWratio, FC.rsun)
    return CME


def runForeCAT(CME, rmax, silent=False, path=False):
    #---------------------------------------------------------------------------------------|
    # Simulation Main Loop -----------------------------------------------------------------|
    #---------------------------------------------------------------------------------------|

    dtprint = CME.dt # initiate print counter
    
    # Set up empty arrays if output path
    if path:
        outts, outRs, outlats, outlons, outtilts, outvs, outAWs, outAWps, outvdefs, outdeltaAxs, outdeltaCSs, outdeltaCAs = [], [], [], [], [], [], [], [], [], [], [], []
    
    # Run until nose hits rmax
    while CME.points[CC.idcent][1,0] <= rmax:
        if CME.points[CC.idcent][1,0] < 2.:
            critT = tprint/2.
        else:
            critT = tprint
        # Check if time to print to screen and/or path
        if (dtprint > critT) or (CME.t==0):
            if not silent:           
                FC.printstep(CME)
            if path:
                outts.append(CME.t)
                outRs.append(CME.points[CC.idcent][1,0]) 
                outlats.append(CME.points[CC.idcent][1,1]) 
                outtilts.append(CME.tilt)                 
                # Need to apply lon and tilt corrections done in printstep
                thislon = CME.points[CC.idcent][1,2]
                # FC.lon0 <-998 flags that we want to keep effects of solar rotation in
                # ie. the CME lon would slip toward lower Carrington lons
                # This is probably not going to be used in many cases so the if part is
                # mostly unnecessary since will always want to adjust
                # lon0 = 0 means run in Carrington coords without including solar rot
                # lon0 = initial CME lon will give relative change (without solar rot)
                if FC.lon0 > -998:
                    newlon = thislon - FC.lon0 + FC.rotrate * 60. * FC.radeg * CME.t
                outlons.append(newlon) 
                #vCME = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
                vdef = np.sqrt(np.sum((CME.vdefLL+CME.vdragLL)**2))/1e5
                vs = np.copy(CME.vs)/1e5
                outvs.append(vs)
                outvdefs.append(vdef)
                outAWs.append(CME.AW*FC.radeg)
                outAWps.append(CME.AWp*FC.radeg)
                outdeltaAxs.append(CME.deltaAx)
                outdeltaCSs.append(CME.deltaCS)
                outdeltaCAs.append(CME.deltaCSAx)
            # Reset print counter        
            dtprint = CME.dt	

        # Advance a step
        CME.update_CME(FC.user_vr, FC.user_exp, FC.user_mass)
        dtprint += CME.dt
            
    # Print final step if tprint not equal to time resolution so that we
    # include the last point that actually crosses rmax       
    if not silent:
        if tprint != CME.dt:
            FC.printstep(CME)
    if path:
        if tprint != CME.dt:
            outts.append(CME.t)
            outRs.append(CME.points[CC.idcent][1,0]) 
            outlats.append(CME.points[CC.idcent][1,1]) 
            outtilts.append(CME.tilt)
            thislon = CME.points[CC.idcent][1,2]
            if FC.lon0 > -998:
                newlon = thislon - FC.lon0 + FC.rotrate * 60. * FC.radeg * CME.t
            outlons.append(newlon)
            #vCME = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
            vdef = np.sqrt(np.sum((CME.vdefLL+CME.vdragLL)**2))/1e5
            vs = CME.vs/1e5
            outvs.append(vs)
            outvdefs.append(vdef)
            outAWs.append(CME.AW*FC.radeg)
            outAWps.append(CME.AWp*FC.radeg)
            outdeltaAxs.append(CME.deltaAx)
            outdeltaCSs.append(CME.deltaCS)
            outdeltaCAs.append(CME.deltaCSAx)
            
            
    # Clean up things        
    if FC.saveData: FC.outfile.close()    # close the output files
    
    # Correct the nose lon of the CME (take out solar rotation)
    # and recalculate the CME points
    if FC.lon0 > -998:
        newlon = CME.cone[1,2] - FC.lon0 + FC.rotrate * 60. * FC.radeg * CME.t
        CME.cone[1,2] = newlon
        CME.calc_points()
    
    # Return the path data if needed, else just return final CME 
    if path:
        return CME, np.array([outts, outRs, outlats, outlons, outtilts, outAWs, outAWps, outvs, outvdefs, outdeltaAxs, outdeltaCSs, outdeltaCAs])
    else:
        return CME



if __name__ == '__main__':
    input_values, allinputs = FC.readinputfile()
    ipos, rmax = initForeCAT(input_values)
    CME = initCME([FC.deltaAx, FC.deltaCS, FC.rstart], ipos)
    CME = runForeCAT(CME, rmax, path=True)
