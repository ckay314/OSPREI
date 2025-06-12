import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm, pearsonr
from scipy import ndimage
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import warnings
warnings.filterwarnings("ignore")

global dtor
dtor = math.pi / 180.


# make label text size bigger
plt.rcParams.update({'font.size':14})

# Set up the path variable
# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
import OSPREI as OSP
mainpath = OSP.mainpath
sys.path.append(os.path.abspath(OSP.codepath)) #MTMYS
sys.path.append(os.path.abspath(OSP.propath)) #MTMYS

import setupPro as SP
import proMetrics as met
import proForeCAT as proFore
import proANTEATR as proANT
import proFIDO

# |----------------------------------------------------------|
# |------ Flags to control if plot all or a single one ------|
# |----------------------------------------------------------|
plotAll = True
# e.g. switch above to false then specific 'if plotAll' to just 'if True'
doMovie = False # option to include enlilesque movie

# |------------------------------------------------|
# |------ Set tag for type of figures to save -----|
# |------------------------------------------------|
global figtag
figtag = 'png' # can only be 'png' or 'pdf'

# |-------------------------------------------------------|
# |------ Switch for giving the BF for each sat ----------|
# |------  it's own special color in each fig ------------|
# |-------------------------------------------------------|
colorEns = True
# Set up a list of nice colors to use to highlight best fit for each satellite
#BFcols = ['#882255', '#88CCEE', '#CC6677', '#44AA99', '#DDCC77', '#AA4499', '#117733', '#332288', ]
BFcols = ['#03135B', '#A50026', '#364B9A', '#E34D34', '#5385BC', '#F99858', '#83B8D7', '#FDD081', '#B7DDEB', '#F3Fd81']
# Color for metric combining all sats
comboCol = 'red'


# |-----------------------------------------------------------|
# |------ The main processing code, just set as func to ------|
# |------  allow for external calls --------------------------|
# |-----------------------------------------------------------|
def runproOSP(inputPassed='noFile', onlyIS=False):
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |------------------------------------- Basic setup -------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    
    # |-----------------------------------------------------------|
    # |------ Use OSPREI to pull in the simulation parameters ----|
    # |-----------------------------------------------------------|
    OSP.setupOSPREI(inputPassed=inputPassed)
    
    
    # |---------------------------------------------|
    # |------ Read in the results from OSPREI ------|    
    # |---------------------------------------------|
    global nSat, nEns, moreSilence, hitsSat, nFails, DoY, dObj, satColors
    try:
        ResArr, nSat, hitsSat, nFails, DoY, dObj = SP.txt2obj()
        nEns = len(ResArr.keys())
        # Stop it from printing too much if have a large ensemble
        if nEns > 10:
            moreSilence = True
        # Set up satellite colors
        satColors = [BFcols[i] for i in range(nSat)]
        satColors = np.append(satColors, comboCol)        
    except:
        sys.exit('Error reading in OSPREI results. Exiting without plotting anything.')
        
    
    # |------------------------------------------------------------|
    # |------ Check if we have observations and pull in if so -----|    
    # |------------------------------------------------------------|    
    global ObsData, hasObs
    hasObs = False
    ObsData = [None]
    satNames = [''] # default for single sat case with no data
    satLoc0 = [[] for i in range(nSat)]  # array of sat locations at start of simulation
    satLocI = [[] for i in range(nSat)]  # array of sat locations at its own time of impact (for each one)
    satLocAllI = [[] for i in range(nSat)]  # array of sat locations at impact for each one
    # e.g. if have PSP, STA then would be [[PSP at PSP time, STA at PSP time], [PSP at STA time, STA at STA time]]

    if (OSP.ObsDataFile is not None) or ('satPath' in OSP.input_values):
        hasObs = True
        #print (OSP.ObsDataFile)
        if OSP.ObsDataFile == None:
            hasObs = False
        try:
            ObsData, satNames, satLoc0, satLocI, satLocAllI, hasObs = SP.processObs(ResArr, nSat, hasObs=hasObs)  
        except:
            print('Error in reading in observations for comparison. Proceeding without them.')
            hasObs = False
            ObsData = [None]
    else:
        hasObs = False
        ObsData, satNames, satLoc0, satLocI, satLocAllI, hasObs = SP.processObs(ResArr, nSat, hasObs=hasObs)

    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |----------------------------------- Ensemble Figures ----------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|

    # |------------------------------------------------------------|
    # |------ Calculate metrics first if we have ensemble ---------|    
    # |------------------------------------------------------------|  
    # setupMetrics will pull out the best fit for each satellite which
    # we can use to color those ensemble members in all the other figs 
    
    BFbySat, BFall = None, None # holders for ens best fits 
    satNamesL = np.append(satNames, ['All'])
    metFail   = False
    try:
        if hasObs:
            allScores, BFbySat, BFall, friends = met.setupMetrics(ResArr, ObsData, nEns, nSat, hasObs, hitsSat, satNames, DoY, silent=False)    
        if nEns > 1:
            comboBFs = np.append(BFbySat, BFall)
        else:
            comboBFs = [None]
        # reset to Nones if don't want to plot colors
        if not colorEns:
            BFbySat, BFall = None, None
    except:
        print('Error in calculating metrics. Proceeding without score analysis')
        metFail = True
        
    # |------------------------------------------------------------|
    # |--------- Make the ensemble score scatter plots ------------|    
    # |------------------------------------------------------------|  
    # These plots are pretty slow for larger ensembles
    if plotAll and (nEns > 1) and not metFail and hasObs: 
        for i in range(nSat):
            # friends at -1 will plot values from combined score
            # switch -1 to i (satID) to use friends for that satellite
            try:
                met.plotEnsScoreScatter(ResArr, allScores, nEns, satID=i, BFs=comboBFs, satNames=satNames, BFcols=satColors, friends=friends[-1])
            except:
                print('Error in making ensemble score scatter for ', satNames[i])
        try:
            # Convert allScores to oneScores
            oneScores = np.sum(allScores, axis=0)
            # Reflag the bad cases after summing messes up
            oneScores[np.where(oneScores < 0)] == -9998
            met.plotEnsScoreScatter(ResArr, oneScores, nEns, satID=-1, BFs=comboBFs, satNames=satNames, BFcols=satColors, friends=friends[-1])
        except:
            print('Error in making combined ensemble score scatter')

  
    # |------------------------------------------------------------|
    # |----- Make the ensemble input/output scatter plots ---------|    
    # |------------------------------------------------------------|  
    if plotAll and (nEns > 1):
        for i in range(nSat):
            # friends at -1 will plot values from combined score
            # switch -1 to i (satID) to use friends for that satellite
            try:
                if hasObs:
                    met.makeEnsplot(ResArr, nEns, critCorr=0.5, satID=i, satNames=satNamesL, BFs=comboBFs, BFcols=satColors, friends=friends[-1])
                else:
                    met.makeEnsplot(ResArr, nEns, critCorr=0.5, satID=i, satNames=satNamesL)
            except:
                print('Error in making ensemble input/output scatter plot')
                

    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |--------------------------------- ForeCAT figure(s) -----------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    
    if OSP.doFC:
    # |------------------------------------------------------------|
    # |------------------------ CPA plot --------------------------|    
    # |------------------------------------------------------------|  
        if plotAll:
            try:
                proFore.makeCPAplot(ResArr, nEns, BFs=comboBFs, satCols=satColors, satNames=satNamesL)
            except:
                print('Error in making CPA plot')
    


    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |------------------------------ Interplanetary figures ---------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    
    if OSP.doANT: 
    # |------------------------------------------------------------|
    # |----------------- Make the J map-like plot -----------------|    
    # |------------------------------------------------------------| 
        if plotAll:
            for i in range(nSat):
                # This has some weird packaging in the call bc the code will do multi sat
                # but not something used in OSPREI for now
                # Get time at start of ANTEATR portion
                try:
                    if not OSP.noDate:
                        datestr = OSP.input_values['date'] + 'T'+ OSP.input_values['time']
                        start = (datetime.datetime.strptime(datestr, "%Y%m%dT%H:%M" ))
                    else:
                        start = None
                    proANT.makeJmap([ResArr[0]], [satLoc0[i]], [start], satName=satNames[i])
                except:
                    print('Error in making J map plot for satellite ', satNames[i])
                
    # |------------------------------------------------------------|
    # |------------ Make the profiles with sheath data ------------|    
    # |------------------------------------------------------------| 
        # The sheath profiles don't always look the best if it is a borderline case
        # where some fraction of the cases stop forming a shock halfway through
        if plotAll and OSP.doPUP:
            try:
                proANT.makePUPplot(ResArr, nEns, BFs=comboBFs, satCols=satColors, satNames=satNamesL)
            except:
                print('Error in making PUP histogram')
            
    # |------------------------------------------------------------|
    # |---------- Make the profiles without sheath data -----------|    
    # |------------------------------------------------------------| 
        if plotAll:
            try:
                proANT.makeDragless(ResArr, nEns, BFs=comboBFs, satCols=satColors, satNames=satNamesL)
            except:
                print('Error in making drag plot')
                
    # |------------------------------------------------------------|
    # |---------------- Make the ANTEATR histogram ----------------|    
    # |------------------------------------------------------------| 
        if plotAll and (nEns>1):
            for i in range(nSat):
                try:
                    proANT.makeAThisto(ResArr, dObj=dObj, DoY=DoY, satID=i, BFs=comboBFs, satCols=satColors, satNames=satNamesL)
                except:
                    print('Error in making ANTEATR histogram for satellite ', satNamesL[i])
            
    # |----------------------------------------------------------------|
    # |------------ Make impact and /or other contours plot -----------|    
    # |----------------------------------------------------------------| 
        if plotAll and (nEns>1):
            for i in range(nSat):
                try:
                    # Option for the multi panel plot
                    #proANT.makeContours(ResArr, nEns, nFails, satID=3,  satLocs=satLocI, satNames=satNames, allSats=True, satCols=satColors, calcwid=95, plotwid=50 )
                    # Option for just single panel with percentage of impacts
                    proANT.makePercMap(ResArr, nEns, nFails, satID=i, satLocs=satLocAllI[i], satNames=satNames, allSats=True, satCols=satColors, calcwid=95, plotwid=50 )
                except:
                    print('Error in making impact contours')

    # |------------------------------------------------------------|
    # |------------- Make tne Enlil-like movie figures ------------|    
    # |------------------------------------------------------------| 
        if doMovie:
            try:
                # vel0 = 300 -> sets min of contours at 300 (defaults to 300)
                # vel1 = 750 -> sets max of contours at 750 (defaults to nearest 50 over vCME)
                proANT.enlilesque(ResArr, bonusTime=0, doSat=True, planes='both', satNames=satNames, satCols=satColors)
            except:
                print('Error in making Enlilesque frames')
    



    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |--------------------------------- In Situ figures -------------------------------------|
    # |---------------------------------------------------------------------------------------|
    # |---------------------------------------------------------------------------------------|
    
    if OSP.doFIDO: 
    # |------------------------------------------------------------|
    # |------------- Make the in situ spaghetti plots -------------|    
    # |------------------------------------------------------------| 
        if plotAll:
            for i in range(nSat):
                if hitsSat[i]:
                    if nEns == 1:
                        comboBFs = [None]
                    try:
                        proFIDO.makeISplot(ResArr, dObj, DoY, satID=i, SWpadF=12, SWpadB=12, BFs=comboBFs, satCols=satColors, satNames=satNamesL, hasObs=hasObs, ObsData=ObsData)
                    except:
                        print('Error in making in situ plot for satellite ', satNames[i])
    # |------------------------------------------------------------|
    # |------------- Make histograms with sheath data -------------|    
    # |------------------------------------------------------------| 
        if OSP.doPUP:
            if plotAll and (nEns>1):
                for i in range(nSat):
                    try:
                        proFIDO.makeallIShistos(ResArr, dObj, DoY, satID=i, satNames=satNamesL, BFs=comboBFs, satCols=satColors)
                    except:
                        print('Error in making FIDO histogram for satellite ', satNamesL[i])
    
    # |------------------------------------------------------------|
    # |------------ Make histograms without sheath data -----------|    
    # |------------------------------------------------------------|
        else:
             if plotAll and (nEns>1):
                 for i in range(nSat):
                     try:
                         proFIDO.makeFIDOhistos(ResArr, dObj, DoY, satID=i, satNames=satNamesL, BFs=comboBFs, satCols=satColors)
                     except:
                        print('Error in making FIDO histogram for satellite ', satNamesL[i])
                    
                 
    
    # |------------------------------------------------------------|
    # |------------------ Make heat map timeline ------------------|    
    # |------------------------------------------------------------| 
        if plotAll and (nEns>1):
            for i in range(nSat):
                if hitsSat[i]:
                    # Adding BFs is possible but does make it pretty messy 
                    #proFIDO.makeAllprob(ResArr, dObj, DoY, satID=i, BFs=comboBFs, satCols=satColors, satNames=satNamesL, hasObs=hasObs, ObsData=ObsData)
                    # Typically better to run without BFs
                    try:
                        proFIDO.makeAllprob(ResArr, dObj, DoY, satID=i, hasObs=hasObs, ObsData=ObsData, satNames=satNames)
                    except:
                        print('Error in making FIDO heat map for satellite ', satNamesL[i])
                    
    
    
    
if __name__ == '__main__':
    runproOSP()