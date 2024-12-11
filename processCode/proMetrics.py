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

from ForeCAT_functions import rotx, roty, rotz, SPH2CART, CART2SPH
import processOSPREI as pO
import setupPro as sP


# |--------------- Helper function to convert to hourly resolution -----------------|        
def hourify(tARR, vecin):
    # Assume input is in days, will spit out results in hourly averages
    # can conveniently change resolution by multiplying input tARR
    # ex. 10*tARR -> 1/10 hr resolution
    sthr = int(tARR[0]*24) 
    endhr = int(tARR[-1]*24)
    nhrs = endhr - sthr
    thrs = np.arange(sthr,endhr,1)
    if np.array_equal(tARR, vecin):
        return thrs/24.
    vecout = np.zeros(len(thrs))
    for i in range(len(thrs)):
        t = thrs[i]
        idx = np.where((tARR*24 >= (t - 0.5)) & (tARR*24 < (t+0.5)))
        if len(idx) != 0:
            vecout[i] = np.mean(vecin[idx])
    # keep the first point, important for precise timing
    vecout[0] = vecin[0]
    return vecout


# |---------------------------------------------------------------------------------|        
# |----------------- Pull OSPREI data and compare to observations ------------------|        
# |---------------------------------------------------------------------------------|        
def getISmetrics(ResArr, ObsData, DoY, satNames, satID=0, ignoreSheath=False, silent=False):
    # |----- Check if we have an observed sheath and if we actually want to ----|
    # |----- include it when calculating the metrics ---------------------------|
    global hasSheath
    hasSheath = False
    if isinstance(OSP.obsShstart[satID], float) and not ignoreSheath: 
        hasSheath = True
        obstSh = OSP.obsShstart[satID]
    obstFR1, obstFR2 = OSP.obsFRstart[satID], OSP.obsFRend[satID]
    
    # |----- Reformat observed time into float day of year -----|
    day1 = datetime.datetime(year=ObsData[satID][0,0].year,month=1,day=1,hour=0)
    deltat = ObsData[satID][0,:]-day1
    # need the days since jan 1 = 0 days
    obst = np.array([1+deltat[i].days + deltat[i].seconds/(3600*24)for i in range(len(deltat))])
 
    # |----- Pull out keys for ensemble members that impact -----|
    goodKeys = []
    for key in ResArr.keys():
        if not ResArr[key].fail:
            goodKeys.append(key)
    
    # |----- Set up the plot boundaries -------------|        
    # |----- Find earliest simulated front/back -----|
    mindate, maxdate = None, None
    for key in goodKeys:
        if ResArr[key].FIDOtimes[satID] is not None:
            transientidx = ResArr[key].FIDO_FRidx[satID]
            if hasSheath:
                transientidx = np.append(ResArr[key].FIDO_shidx[satID], ResArr[key].FIDO_FRidx[satID]).astype(int)
            dates = ResArr[key].FIDOtimes[satID][transientidx]
            # save the extreme times to know plot range
            if mindate is None: 
                mindate = dates[0]
                maxdate = dates[-1]
            if dates[0] < mindate: mindate = dates[0]
            if dates[-1] > maxdate: maxdate = dates[-1]
            
    # |----- Grab obs data that extends slightly past transient time -----|
    mintsim, maxtsim = mindate+DoY+1, maxdate+DoY+1
    pad = 48/24.
    mintobs = np.max([mintsim - pad, obst[0]])
    maxtobs = np.min([maxtsim + pad, obst[-1]])
    
    obshrt = hourify(obst, obst)
    goodidx = np.where((obshrt >= mintobs) & (obshrt <= maxtobs))[0]
    obshr = np.zeros([len(goodidx), 8])
    obshr[:,0] = obshrt[goodidx]
    
    # |----- Check for missing data in individual parameters -----|
    haveObsidx = []
    for i in range(7):
        thisParam = ObsData[satID][i+1,:]
        isNans = []
        for j in range(len(thisParam)):
            if math.isnan(thisParam[j]): isNans.append(j)
        if len(isNans) < len(thisParam) * 0.5:
            haveObsidx.append(i)            
    for i in haveObsidx:
        hrvalsObs = hourify(obst, ObsData[satID][i+1,:])
        obshr[:,i+1] = hrvalsObs[goodidx]
    
    # |----- Count the number of impacting events -----|
    nHits = 0
    for key in goodKeys:
        if ResArr[key].FIDOtimes[satID] is not None:
            nHits += 1
    
    # |----- Set up arrays to hold scores depending on -----|
    # |----- whether using the sheath or not ---------------|        
    if hasSheath:
        allScores = np.zeros([nHits,21]) 
        meanVals  = np.zeros(21)
    else:
        allScores = np.zeros([nHits,7])   
        meanVals = np.zeros(7)
    timingErr = np.zeros([nHits,3])
    
    # |----- Set up a file to save output -----|    
    f1 = open(OSP.Dir+'/metrics'+str(ResArr[0].name)+''+satNames[satID]+'.dat', 'w')    
    
    # |----- Start looping through the results for metrics -----|    
    hitCounter = 0
    betterKeys = []
    for key in ResArr.keys():
        if (ResArr[key].FIDOtimes[satID] is not None) & (key in goodKeys):
            # |----- Double check not ANT fail but is FIDO miss -----|
            betterKeys.append(key) 
            # |----- Get the sim start/stop indices -----|
            thisRes = ResArr[key]
            thistime = thisRes.FIDOtimes[satID]+DoY+1
            shidx = thisRes.FIDO_shidx[satID]
            if not thisRes.sheathOnly[satID]:
                FRidx = thisRes.FIDO_FRidx[satID]
            transientidx = np.append(thisRes.FIDO_shidx[satID], thisRes.FIDO_FRidx[satID])
            # |----- Pull the sim results corresponding to event -----|
            compVals = [thisRes.FIDOBs[satID], thisRes.FIDOBxs[satID], thisRes.FIDOBys[satID], thisRes.FIDOBzs[satID], thisRes.FIDOns[satID], thisRes.FIDOvs[satID], thisRes.FIDOtems[satID]]
            # |----- Check if actually made a sheath when it should -----|
            if hasSheath:
                if len(shidx)==0:
                    shErr = 1 # penalty for not making a sheath
                else:
                    shErr = np.abs(obstSh - thistime[shidx[0]])
                if not thisRes.sheathOnly[satID]:    
                    timingErr[hitCounter,:] = [np.abs(obstFR1 -thistime[FRidx[0]]), np.abs(obstFR2 - thistime[FRidx[-1]]), shErr]
                else:
                    # Big timing penalties for no FR only sheath
                    timingErr[hitCounter,:] = [99, 99, shErr]
            else:
                timingErr[hitCounter,:] = [np.abs(obstFR1 -thistime[FRidx[0]]), np.abs(obstFR2 - thistime[FRidx[-1]]), 0]
            # |----- Convert the sim data into hourly resolution ----|
            simhrts = hourify(thistime, thistime)
            obshrts = obshr[:,0]            
            
            # |----- Calculate metrics for 3 diff ranges ----|
            # |----- (1 full transient, 2 sheath, 3 FR) -----|
            if hasSheath:
                rngs = [[obstSh, obstFR2], [obstSh, obstFR1], [obstFR1, obstFR2]]
                haveObsidxFull = [haveObsidx[i] for i in range(len(haveObsidx))] + [haveObsidx[i] +7 for i in range(len(haveObsidx))] +  [haveObsidx[i] +14 for i in range(len(haveObsidx))]
                if thisRes.sheathOnly[satID]:
                    # should have set FR1 bound at back of sheath previously for sheath only case
                    rngs = [[obstSh, obstFR1]]
                    haveObsidxFull = [haveObsidx[i] for i in range(len(haveObsidx))]
                    
            else:
                rngs =[[obstFR1, obstFR2]]
                haveObsidxFull = [haveObsidx[i] for i in range(len(haveObsidx))] + [haveObsidx[i] +7 for i in range(len(haveObsidx))]
            
            outprint = ''
            rngcounter = 0
            for rng in rngs:    
                # |----- Make sure actually have obs data for each param -----|
                for i in haveObsidx:
                    # |----- Make obs hourly -----|
                    hrvals = hourify(thistime, compVals[i])
                    
                    # |----- Extract appropriate time ranges -----|
                    subSidx = np.where((simhrts >= rng[0]) & (simhrts <= rng[1]))[0]
                    subOidx = np.where((obshrts >= rng[0]) & (obshrts <= rng[1]))[0]
                    
                    # |----- Check that no missing data -----|
                    # |----- Take subset if needed ----------| 
                    if simhrts[subSidx][-1] != obshrts[subOidx][-1]:
                        if simhrts[subSidx][-1] > obshrts[subOidx][-1]:
                            maxidx = np.where(simhrts[subSidx] == obshrts[subOidx][-1])[0]
                            subSidx = subSidx[:maxidx[0]+1]
                        else:
                            maxidx = np.where(obshrts[subOidx] == simhrts[subSidx][-1])[0]
                            subOidx = subOidx[:maxidx[0]+1]
                    
                    # |---- Get the time and hourly values for comparison -----|
                    subt = simhrts[subSidx]
                    simvals = hrvals[subSidx]
                    obsvals = obshr[subOidx,i+1]
                    
                    # |----- Clean out any bad values -----|
                    cleanidx = np.where((np.isfinite(simvals) & np.isfinite(obsvals)))[0]
                    simvalsC = simvals[cleanidx]
                    obsvalsC = obsvals[cleanidx]
                    
                    # |----- Calculate the mean absolute hourly error for every param ----|
                    thisErr = np.mean(np.abs(obsvalsC - simvalsC))
                    allScores[hitCounter, 7*rngcounter + i] = thisErr
                    outprint += '{:.3f}'.format(thisErr)+' '
                    
                    # |----- Get the mean obs value during seed run -----|
                    # |----- Need this to do normalized metrics ---------|
                    if hitCounter == 0:
                        meanVals[7*rngcounter + i] = np.mean(np.abs(obsvalsC))
                    
                outprint += '  '
                rngcounter += 1
            hitCounter += 1
            
            # |----- Print and save results -----|
            if not silent:
                print (key, outprint)  
            f1.write(str(key).rjust(4)+' '+outprint+'\n')          
    f1.close()
    betterKeys = np.array(betterKeys)
    return allScores, timingErr, meanVals, betterKeys, haveObsidxFull, haveObsidx


# |---------------------------------------------------------------------------------|        
# |----------------- Take individual errors and convert to scores ------------------|        
# |---------------------------------------------------------------------------------|        
def analyzeMetrics(ResArr, allScores, timingErr, meanVals, betterKeys, haveObsidxFull, haveObsidx, fIn=None, silent=False):
    # |----- Set up labels on screen -----|
    if hasSheath:
        if not silent:
            print ('')            
            print ('')
            print ('Average Errors (Full/Sheath/Flux Rope, B/Bx/By/Bz/n/v/T):')
    else:
        if not silent:
            print ('')            
            print ('')
            print ('Average Errors (B/Bx/By/Bz/n/v/T):')
    if fIn:
        fIn.write('Average Errors (B/Bx/By/Bz/n/v/T): \n')
            
            
    # |----- Sum the average error over all members        
    avgErr = np.mean(allScores, axis=0)
    for i in range(len(avgErr)):
        if i not in haveObsidxFull:
            avgErr[i] = 9999.
            meanVals[i] = 9999.
    outprint = ''
    for val in avgErr:
        outprint += '{:.3f}'.format(val)+' '
    if not silent:
        print (outprint)
        print (' ')
    if fIn:
        labs = ['       Full: ', '     Sheath: ', '         FR: ']
        for i in range(3):
            outprint = labs[i]
            for j in range(7):
                outprint += '{:.3f}'.format(avgErr[7*i + j])+' '
            fIn.write(outprint + '\n')
                
    
    # |----- Calculate weighted errors -----|
    weightedScores = allScores/meanVals
    if not silent: print ('Average Weighted Errors:')
    if fIn:
        fIn.write('Average Weighted Errors: \n')
        
    avgwErr = np.mean(weightedScores, axis=0)
    for i in range(len(avgErr)):
        if i not in haveObsidxFull:
            avgwErr[i] = 9999.
    outprint = ''
    for val in avgwErr:
        outprint += '{:.3f}'.format(val)+' '
    if not silent:
        print (outprint)    
    if fIn: 
        for i in range(3):
            outprint = labs[i]
            for j in range(7):
                outprint += '{:.3f}'.format(avgwErr[7*i + j])+' '
            fIn.write(outprint + '\n')
         
    
    # |----- Calculate overall metrics combining parameters ----|
    totTimeErr = np.sum(timingErr, axis=1)
    # |----- Pick whichever you want to use in the total score -----|
    # |----- (order is [Btot, Bx, By, Bz, v, T, n]) ----------------|
    #want2use = [0,4,5,6] # no B vec
    want2use = [0,1,2,3,4,5,6] # all
    #want2use = [1,2,3] # b vec only
    
    # |----- Check that data isn't missing then sum those params -----|
    canuse = []
    for i in range(7):
        if (i in want2use) and (i in haveObsidx):
            canuse.append(i)
    oneScores = np.sum(weightedScores[:,canuse], axis=1) + totTimeErr

    # |----- Find the best overall score (for this sat) -----|
    bestScore = np.min(oneScores)
    bestidx = np.where(oneScores == bestScore)[0]
    if not silent:
        print ('')
        print ('Seed has total score of ', oneScores[0])
        print ('Best total score of ', bestScore, 'for ensemble member ', betterKeys[bestidx][0])
        print ('')
    if fIn:
        fIn.write('Seed has total score of ' + str(oneScores[0]) + '\n')
        fIn.write('Best total score of ' + str(bestScore) + ' for ensemble member ' + str(betterKeys[bestidx][0])+'\n')
        fIn.write('\n')

    return oneScores, betterKeys, betterKeys[bestidx][0]


# |---------------------------------------------------------------------------------|        
# |------------------- Make figure showing scatter in scores  ----------------------|        
# |---------------------------------------------------------------------------------|        
def plotEnsScoreScatter(ResArr, allScores, nEns, satID=0, BFs=[None], satNames=[None], BFcols=None, friends=[None]):
    #|---------- Pull out the scores for the chosen satellite ----------|
    if satID != -1:
        oneScores = allScores[satID,:]
    else:
        oneScores = allScores
    goodkeys = []
    for idx in range(nEns):
        if oneScores[idx] != -9998:
            goodkeys.append(idx)
    if satNames[0]:
        satName = satNames[satID]
    else:
        satName = '' 

    nVaried = len(ResArr[0].EnsVal.keys())
    ensKeys = ResArr[0].EnsVal.keys()

    
    #|---------- Set up the figure ----------|
    nx = int(nVaried/4)
    if nx*4 < nVaried: nx += 1
    fig, axes = plt.subplots(nx, 4, sharey=True, figsize=(8,12))
    axes = axes.reshape([-1])
    counter = 0
    
    #|---------- Dictionary to convert keys to label names ----------|
    deg = '('+'$^\\circ$'+')'
    myLabs = {'CMElat':'Lat '+deg, 'CMElon':'Lon '+deg, 'CMEtilt':'Tilt ' +deg, 'CMEvr':'v$_F$ (km/s)', 'CMEAW':'AW '+deg, 'CMEAWp':'AW$_{\\perp}$ '+deg, 'CMEdelAx':'$\\delta_{Ax}$', 'CMEdelCS':'$\\delta_{CS}$', 'CMEdelCSAx':'$\\delta_{CA}$', 'CMEr':'R$_{F0}$ (R$_S$)', 'FCrmax':'FC end R$_{F0}$ (R$_S$)', 'FCraccel1':'FC R$_{v1}$ (km/s)', 'FCraccel2':'FC R$_{v2}$ (km/s)', 'FCvrmin':'FC v$_{0}$ (km/s)', 'FCAWmin':'FC AW$_{0}$ '+deg, 'FCAWr':'FC R$_{AW}$ (R$_S$)', 'CMEM':'M$_{CME}$ (10$^{15}$ g)', 'FCrmaxM':'FC R$_{M}$ (R$_S$)', 'FRB':'B$_0$ (nT)', 'CMEvExp':'v$_{Exp}$ (km/s)', 'SWCd': 'C$_d$', 'SWCdp':'C$_{d,\\perp}$', 'SWn':'n$_{SW}$ (cm$^{-3}$)', 'SWv':'v$_{SW}$ (km/s)', 'SWB':'B$_{SW}$ (nT)', 'SWT':'log(T$_{SW}$) (K)', 'SWcs':'c$_s$ (km/s)', 'SWvA':'v$_A$ (km/s)', 'FRB':'B (nT)', 'FRtau':'$\\tau', 'FRCnm':'C$_{nm}$', 'FRT':'log(T) (K)',  'Gamma':'$\\gamma$', 'IVDf1':'$f_{Exp}$', 'IVDf':'$f_{Exp}$', 'IVDf2':'$f_2$', 'CMEvTrans':'v$_{Trans}$ (km/s)', 'SWBx':'SW B$_x$ (nT)', 'SWBy':'SW B$_y$ (nT)', 'SWBz':'SW B$_z$ (nT)', 'MHarea':'CH Area (10$^{10}$ km$^2$)', 'MHdist':'HSS Dist. (au)'}
    
    #|---------- Clean up the range for the non/sheath impact cases ----------|
    goodHits = np.where((oneScores < 100) & (oneScores > 0))# think this criteria will always be good?
    newMax = np.max(oneScores[goodHits])
    newMin = np.max(oneScores[goodHits])
    outliers = np.where((oneScores > 100) | (oneScores <= 0))
    badval = newMin + 0.25 * (newMax-newMin)
    cleanScores = oneScores*0
    cleanScores[goodHits] = oneScores[goodHits]
    cleanScores[outliers] = badval

    #|---------- Make the scatter plot from ensemble values and scores ----------|
    for key in goodkeys:
        axCounter=0
        for keyV in ensKeys:
            if oneScores[key] > 0: # negative is only for fails
                if keyV in ['FRT', 'SWT']:
                    axes[axCounter].scatter(np.log10(ResArr[key].EnsVal[keyV]), cleanScores[key], c='lightgray')
                else:
                    axes[axCounter].scatter(ResArr[key].EnsVal[keyV], cleanScores[key], c='lightgray')
            axCounter +=1
    
    #|---------- Add labels ----------|        
    axCounter = 0
    for key in ensKeys:            
        axes[axCounter].set_xlabel(myLabs[key])
        if axCounter % 4 == 0:
            axes[axCounter].set_ylabel('Score')
        axCounter +=1

    #|---------- Highlight near best fit cases ----------|
    if friends[0] or (friends[0]==0):
        for idx in friends:
            axCounter = 0
            for keyV in ensKeys:
                if keyV in ['FRT', 'SWT']:
                    axes[axCounter].scatter(np.log10(ResArr[idx].EnsVal[keyV]), cleanScores[idx], c='dimgray')
                else:
                    axes[axCounter].scatter(ResArr[idx].EnsVal[keyV], cleanScores[idx], c='dimgray')
                axCounter += 1
            
    #|---------- Highlight best fit cases ----------|
    if BFs[0] or (BFs[0] == 0):
        for i in range(len(BFs)):
            key = int(BFs[i])
            myCol = BFcols[i]
            moreidxs = np.where(BFs == key)[0]
            # If have multiple sats with same BF set all to first color
            if len(moreidxs) > 1:
                myCol = BFcols[moreidxs[0]]
            axCounter=0
            for keyV in ensKeys:
                if axCounter == 0:
                    if i != len(BFs) -1:
                        myName = satNames[i]
                    else:
                        myName = 'All'
                    if keyV in ['FRT', 'SWT']:    
                        axes[axCounter].plot(np.log10(ResArr[key].EnsVal[keyV]), cleanScores[key], 'o', ms=11, mfc=myCol, mec='k', label=myName)
                    else:
                        axes[axCounter].plot(ResArr[key].EnsVal[keyV], cleanScores[key], 'o', ms=11, mfc=myCol, mec='k', label=myName)
                else:
                    if keyV in ['FRT', 'SWT']:
                        axes[axCounter].plot(np.log10(ResArr[key].EnsVal[keyV]), cleanScores[key], 'o', ms=11, mfc=myCol, mec='k')
                    else:
                        axes[axCounter].plot(ResArr[key].EnsVal[keyV], cleanScores[key], 'o', ms=11, mfc=myCol, mec='k')
                axCounter +=1
    # Add the legend
    fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(BFs))
    
    #|---------- Add a line under the outlier cases ----------| 
    axCounter = 0
    if len(outliers):
        for key in ensKeys:
            xr = axes[axCounter].get_xlim()
            yr = axes[axCounter].get_ylim()
            axes[axCounter].plot(xr, [badval, badval], 'lightgray', ls='--', zorder=0)
            axes[axCounter].set_xlim(xr)
            axes[axCounter].set_ylim(yr)
            axCounter +=1
    
    
    #|---------- Turn off empty panels ----------|    
    tooMany = len(axes) - nVaried 
    for i in range(tooMany):
        axes[-(i+1)].axis('off')
        
    #|---------- Make pretty and save ----------|    
    plt.subplots_adjust(wspace=0.2, hspace=0.5,left=0.1,right=0.95,top=0.95,bottom=0.05)   
    if satID != -1: 
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_EnsScores_'+satNames[satID]+'.'+pO.figtag)
    else:
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_EnsScores_All.'+pO.figtag)


# |---------------------------------------------------------------------------------|        
# |----------------- Make figure showing scatter in input/output  ------------------|        
# |---------------------------------------------------------------------------------|        
def makeEnsplot(ResArr, nEns, critCorr=0.5, satID=0, satNames='', BFs=[None], BFcols=None, silent=False, friends=[None]):
    # |---------- Pull the sat name for saving ----------| 
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
    
    # |---------- Number of params varied in the ensemble ----------| 
    nVaried = len(sP.varied)
    
    # |---------- Set up plot label dictionary  ----------| 
    deg = '('+'$^\\circ$'+')'

    out2outLab = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\\perp}$\n'+deg, 'CMEdelAx':'$\\delta_{Ax}$', 'CMEdelCS':'$\\delta_{CS}$', 'CMEvF':'v$_{F}$\n(km/s)', 'CMEvExp':'v$_{Exp}$\n(km/s)', 'TT':'Transit\nTime\n(days)', 'Dur':'Dur\n(hours)', 'n':'n\n(cm$^{-3}$)',  'B':'max B (nT)', 'Bz':'min Bz\n(nT)', 'Kp':'max Kp', 'logT':'log$_{10}$T\n(K)'}
    
    myLabs = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEvr':'v$_F$\n(km/s)', 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\\perp}$\n'+deg, 'CMEdelAx':'$\\delta_{Ax}$', 'CMEdelCS':'$\\delta_{CS}$', 'CMEdelCSAx':'$\\delta_{CA}$', 'CMEr':'R$_{F0}$ (R$_S$)', 'FCrmax':'FC end R$_{F0}$\n (R$_S$)', 'FCraccel1':'FC R$_{v1}$\n (km/s)', 'FCraccel2':'FC R$_{v2}$\n (km/s)', 'FCvrmin':'FC v$_{0}$\n (km/s)', 'FCAWmin':'FC AW$_{0}$\n'+deg, 'FCAWr':'FC R$_{AW}$\n (R$_S$)', 'CMEM':'M$_{CME}$\n(10$^{15}$ g)', 'FCrmaxM':'FC R$_{M}$\n(R$_S$)', 'FRB':'B$_0$ (nT)', 'CMEvExp':'v$_{Exp}$\n (km/s)', 'SWCd': 'C$_d$', 'SWCdp':'C$_{d,\\perp}$', 'SWn':'n$_{SW}$\n(cm$^{-3}$)', 'SWv':'v$_{SW}$\n(km/s)', 'SWB':'B$_{SW}$\n(nT)', 'SWT':'T$_{SW}$\n(K)', 'SWcs':'c$_s$\n(km/s)', 'SWvA':'v$_A$\n(km/s)', 'FRB':'B (nT)', 'FRtau':'$\\tau', 'FRCnm':'C$_{nm}$', 'FRT':'T [K]',  'Gamma':'$\\gamma$', 'IVDf1':'$f_{Exp}$', 'IVDf':'$f_{Exp}$', 'IVDf2':'$f_2$', 'CMEvTrans':'v$_{Trans}$\n(km/s)', 'SWBx':'SW B$_x$\n(nT)', 'SWBy':'SW B$_y$\n(nT)', 'SWBz':'SW B$_z$\n(nT)', 'MHarea':'CH Area (10$^{10}$ km$^2$)', 'MHdist':'HSS Dist. (au)'}
    
    # |---------- Figure out which models were run  ----------| 
    # |----------  use to pick appropriate outputs  ----------| 
    
    configID = 0
    if OSP.doFC: configID += 100
    if OSP.doANT: configID += 10
    if OSP.doFIDO: configID += 1
    nVertDict = {100:9, 110:14, 111:16, 11:13, 10:11, 1:4}
    nVert = nVertDict[configID]
    outDict = {100:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp'], 110:['CMElat', 'CMElon',  'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT','Kp'], 111:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT', 'B', 'Bz', 'Kp'], 11:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n',  'logT', 'B', 'Bz', 'Kp'], 10:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT', 'Kp'], 1:['Dur',  'B', 'Bz',  'Kp']}
    
    # |---------- Pull out only impacts  ----------| 
    hits = []
    for i in range(nEns):
        if (not ResArr[i].FIDOmiss[satID]) and (not ResArr[i].fail):
            hits.append(i)

    # |---------- Determine grid dimension ----------| 
    nHoriz = len(sP.varied)
            
    # |---------- Regroup ensemble values to make nicer for loops below  ----------| 
    EnsVal = np.zeros([nHoriz, nEns]) 
    i = 0
    for key in ResArr.keys():
        j = 0
        for item in sP.varied:
            if item != 'CMElon':
                EnsVal[j,key] = ResArr[key].EnsVal[item]
            else:
                EnsVal[j,key] = ResArr[key].EnsVal[item] - OSP.satPos[1] 
            j += 1
    
    # |---------- More regrouping but for OSPREI results   ----------| 
    OSPres = {}
    for item in outDict[configID]: OSPres[item] = []
    counter = 0
    i = 0
    goodIDs = []
    failIDs = []
    for key in ResArr.keys():
        if (not ResArr[key].FIDOmiss[satID]):
            if not ResArr[key].fail:
                FIDOidx = ResArr[key].FIDO_FRidx[satID]
                if len(FIDOidx) > 0:
                    goodIDs.append(key)
                    ANTidx  = np.min(np.where(ResArr[key].ANTtimes >= ResArr[key].FIDOtimes[satID][FIDOidx][0]))
            else:
                failIDs.append(key)
        elif configID == 100: goodIDs.append(key)
        # |---------- Loop through the potential outputs for the setup ----------| 
        if (not ResArr[key].FIDOmiss[satID]):
            for item in outDict[configID]:
                if item == 'CMElat':
                    OSPres[item].append(ResArr[key].FClats[-1])
                if item == 'CMElon':
                    OSPres[item].append(ResArr[key].FClonsS[-1])
                if item == 'CMEtilt':
                    OSPres[item].append(ResArr[key].FCtilts[-1])
                if item == 'CMEAW':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTAWs[ANTidx])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTAWs[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCAWs[-1])
                if item == 'CMEAWp':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTAWps[ANTidx])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTAWps[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCAWps[-1])
                if item == 'CMEdelAx':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTdelAxs[ANTidx])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTdelAxs[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCdelAxs[-1])
                if item == 'CMEdelCS':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTdelCSs[ANTidx])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTdelCSs[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCdelCSs[-1])
                if item == 'CMEvF':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].FIDOvs[satID][FIDOidx][0])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTvFs[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCvFs[-1])
                if item == 'CMEvExp':
                    if OSP.doFIDO and (key in goodIDs):
                        OSPres[item].append(ResArr[key].FIDOvs[satID][FIDOidx][-1] - ResArr[key].FIDOvs[satID][FIDOidx][0])
                    elif OSP.doANT and (key in goodIDs):
                        OSPres[item].append(ResArr[key].ANTvCSrs[-1])
                    else:
                        OSPres[item].append(ResArr[key].FCvCSrs[-1])
            
                if (key in goodIDs) and (key not in failIDs):
                    if item == 'TT':
                        if OSP.doFIDO:
                            if ResArr[key].FIDOtimes[satID] is not None:
                                OSPres[item].append(ResArr[key].FIDOtimes[satID][0])    
                        else:
                            OSPres[item].append(ResArr[key].ANTtimes[-1]+ResArr[key].FCtimes[-1]/60/24.)                    
                    if item == 'Dur':
                        if OSP.doFIDO:
                            if ResArr[key].FIDOtimes[satID] is not None:
                                OSPres[item].append(ResArr[key].FIDO_FRdur[satID])
                        else:
                            OSPres[item].append(ResArr[key].ANTdur[satID])
                    if item == 'n':
                        if OSP.doFIDO and (key in goodIDs):
                            OSPres[item].append(ResArr[key].ANTns[ANTidx])
                        elif OSP.doANT and (key in goodIDs):  
                            OSPres[item].append(ResArr[key].ANTns[-1])   
                    if item == 'logT':
                        if OSP.doFIDO and (key in goodIDs):
                            OSPres[item].append(ResArr[key].ANTlogTs[ANTidx])    
                        elif OSP.doANT and (key in goodIDs):  
                            OSPres[item].append(ResArr[key].ANTlogTs[-1])                 
                    if item == 'Kp':
                        if OSP.doFIDO:
                            if ResArr[key].FIDOtimes[satID] is not None:
                                OSPres[item].append(np.max(ResArr[key].FIDOKps[satID]))
                        else:
                            OSPres[item].append(ResArr[key].ANTKp0[satID])
                    if item == 'B':
                        if ResArr[key].FIDOtimes[satID] is not None:
                            OSPres[item].append(np.max(ResArr[key].FIDOBs[satID]))                                
                    if item == 'Bz':
                        if ResArr[key].FIDOtimes[satID] is not None:
                            OSPres[item].append(np.min(ResArr[key].FIDOBzs[satID]))
    goodIDs = np.array(goodIDs)
    
    # |---------- Save the overall values characterizing ensemble outputs ----------| 
    f1 = open(OSP.Dir+'/EnsembleOutputs'+str(ResArr[0].name)+'_'+satName+'.dat', 'w')
    if not silent:
        print ('')
        print ('|------------------------', satName[1:], '------------------------|')
        print ('Number of hits: ', len(goodIDs)) 
        print ('                Mean        STD        Min        Max')
    for item in outDict[configID]:
        OSPres[item] = np.array(OSPres[item])
        ensMean, ensSTD, ensMin, ensMax = np.mean(OSPres[item]), np.std(OSPres[item]), np.min(OSPres[item]), np.max(OSPres[item])
        if not silent:
            print (item.ljust(10), '{:10.3f}'.format(ensMean), '{:10.3f}'.format(ensSTD), '{:10.3f}'.format(ensMin), '{:10.3f}'.format(ensMax))  
        f1.write(item.ljust(10) + '{:10.3f}'.format(ensMean) + '{:10.3f}'.format(ensSTD) + '{:10.3f}'.format(ensMin)+  '{:10.3f}'.format(ensMax) +'\n')
    f1.close()
    
    # |---------- Calculate correlations between inputs/outputs ----------| 
    corrs = np.zeros([nVert, nHoriz])
    for i in range(nHoriz):
        for j in range(nVert):
            if len(OSPres[outDict[configID][j]]) == nEns:
                try:
                    col = np.abs(pearsonr(EnsVal[i,:], OSPres[outDict[configID][j]])[0])#*np.ones(nEns)
                except:
                    col = 0 # add catch in case no variation and can't make correlation
            else:
                try:
                    col = np.abs(pearsonr(EnsVal[i,goodIDs], OSPres[outDict[configID][j]])[0])#*np.ones(len(goodIDs))
                except:
                    col = 0.

            corrs[j,i] = col
    # clean out any NaNs (might have for zero variation params)
    corrs[~np.isfinite(corrs)] = 0
    
    # |---------- Figure out who has meaningful correlations ----------| 
    goodVidx = []
    goodHidx = []
    for i in range(nVert):
        maxCorr =  np.max(corrs[i,:])
        if maxCorr >= critCorr: goodVidx.append(i)
    for i in range(nHoriz):
        maxCorr =  np.max(corrs[:,i])
        if maxCorr >= critCorr: goodHidx.append(i)
    newnVert = len(goodVidx)
    newnHoriz = len(goodHidx)
    
    newCorr = np.zeros([newnVert, newnHoriz])
    for i in range(newnVert):
        vidx = goodVidx[i]
        newCorr[i] = corrs[vidx,goodHidx]
    
    newOuts = np.array(outDict[configID])[goodVidx]
    newIns  = np.array(sP.varied)[goodHidx]
        
    # |---------- Repackage the results with meaningful correlations ----------| 
    newEnsVal = np.zeros([newnHoriz, nEns]) 
    i = 0
    for key in ResArr.keys():
        j = 0
        for item in newIns:
            if item != 'CMElon':
                newEnsVal[j,key] = ResArr[key].EnsVal[item]
            else:
                newEnsVal[j,key] = ResArr[key].EnsVal[item] - OSP.satPos[1]
            j += 1
            
    # |---------- Set up the figure and plot ----------| 
    f, a = plt.subplots(1, 1)
    img = a.imshow(np.array([[0,1]]), cmap="turbo")
    
    if newnHoriz > 1:
        fig, axes = plt.subplots(newnVert, newnHoriz, figsize=(1.5*newnHoriz,1.5*(newnVert+0.5)))
        # Turn off tick labels to declutter
        for i in range(newnVert-1):
            for j in range(newnHoriz):
                axes[i,j].set_xticklabels([])
        for j in range(newnHoriz-1):
            for i in range(newnVert):
                axes[i,j+1].set_yticklabels([])
                
        # Actually plot things 
        for i in range(newnHoriz):
            for j in range(newnVert):
                if len(OSPres[newOuts[j]]) == nEns:
                    axes[j,i].scatter(newEnsVal[i,:], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,i]*np.ones(len(newEnsVal[i,:]))))   
                else:
                    axes[j,i].scatter(newEnsVal[i,goodIDs], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,i]*np.ones(len(newEnsVal[i,goodIDs])))) 
                
                # Add the close to BF friends    
                if friends[0] or (friends[0] == 0):
                    for idx in friends:
                        k = np.where(goodIDs == idx)[0]
                        axes[j,i].scatter(newEnsVal[i,k], OSPres[newOuts[j]][k], color='dimgray')    
                        
                # Add the BF    
                if BFs[0]  or (BFs[0] == 0):    
                    for idx in range(len(BFs)):
                        k = np.where(goodIDs == BFs[idx])[0]
                        allidx = np.where(BFs == BFs[idx])[0]
                        if len(allidx) > 1:
                            myCol = BFcols[allidx[0]]
                        else:
                            myCol = BFcols[idx]
                            
                        if (i+j)==0:
                            axes[j,i].plot(newEnsVal[i,k], OSPres[newOuts[j]][k], 'o', ms=11, mfc=myCol, mec='k', label=satNames[idx])
                        else:
                            axes[j,i].plot(newEnsVal[i,k], OSPres[newOuts[j]][k], 'o', ms=11, mfc=myCol, mec='k')
                
        # Rotate bottom axes labels
        for i in range(newnHoriz):
            plt.setp(axes[-1,i].xaxis.get_majorticklabels(), rotation=70 )
    
        # Add labels
        for i in range(newnHoriz): axes[-1,i].set_xlabel(myLabs[newIns[i]])  
        for j in range(newnVert):  axes[j,0].set_ylabel(out2outLab[newOuts[j]])  
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.15, bottom=0.2, top=0.97, right=0.99)
        cbar_ax = fig.add_axes([0.15, 0.09, 0.79, 0.02])    
        cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')   
        cb.set_label('Correlation') 
        
        if BFs[0] or (BFs[0] == 0):
            fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(BFs))
        
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_ENS'+satName+'.'+pO.figtag)
        
        
    # |---------- Different plot algorithm if only a single column ----------| 
    elif newnVert*newnHoriz >0:
        fig, axes = plt.subplots(newnVert, newnHoriz, figsize=(6,1.5*(newnVert+0.5)))
        for j in range(newnVert-1):
            axes[j].set_xticklabels([])
        for j in range(newnVert):
            if len(OSPres[newOuts[j]]) == nEns:
                axes[j].scatter(newEnsVal[0,:], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,0]*np.ones(len(newEnsVal[0,:]))))
            else:
                axes[j].scatter(newEnsVal[0,goodIDs], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,0]*np.ones(len(newEnsVal[0,goodIDs]))))
            
            # Add the close to BF friends    
            if friends[0] or (friends[0] == 0):
                for idx in friends:
                    k = np.where(goodIDs == idx)[0]
                    axes[j,i].scatter(newEnsVal[0,k], OSPres[newOuts[j]][k], color='dimgray')    
                    
            # Add the BF    
            if BFs[0] or (BFs[0] == 0):    
                for idx in range(len(BFs)):
                    k = np.where(goodIDs == idx)[0]
                    if (i+j)==0:
                        axes[j,i].plot(newEnsVal[0,k], OSPres[newOuts[j]][k], 'o', ms=11, mfc=BFcols[idx], mec='k', label=satNames[idx])
                    else:
                        axes[j,i].plot(newEnsVal[0,k], OSPres[newOuts[j]][k], 'o', ms=11, mfc=BFcols[idx], mec='k')
                
        
        axes[-1].set_xlabel(myLabs[newIns[0]])  
        for j in range(newnVert): 
            axes[j].set_ylabel(out2outLab[newOuts[j]])
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.25, bottom=0.2, top=0.97, right=0.98)
        
       
        cbar_ax = fig.add_axes([0.15, 0.09, 0.79, 0.02])    
        cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')   
        cb.set_label('Correlation') 
        
        if BFs[0] or (BFs[0] == 0):
            fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(BFs))
            
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_ENS.'+pO.figtag)
        plt.close() 
    else:
        print('No significant correlations, not making ENS plot')


# |---------------------------------------------------------------------------------|        
# |------------------- Overall metrics wrapper called by proOSP --------------------|        
# |---------------------------------------------------------------------------------|        
def setupMetrics(ResArr, ObsData, nEns, nSat, hasObs, hitsSat, satNames, DoY, silent=False):
    # Set up variable used in combining metrics from different satellites
    sumScore = 0
    winners  = np.zeros(nSat)
    allFriends = [[] for i in range(nSat+1)]
    # Option to unevenly weight sats if desired (e.g. satScoreWeights = [1,1,2,1])
    # The sat order is the same as in satNames/.sat file
    satScoreWeights = np.ones(nSat)
    # |---- Make a file to hold combined metrics ----|
    if nEns > 1:
        fAll = open(OSP.Dir+'/metricsCombined'+str(ResArr[0].name)+'.dat', 'w')  
    else:
        fAll = None
        
    
    # |---- Loop through each satellite computing metrics ----|
    storeScores = np.ones([nSat, nEns]) -9999
    for i in range(nSat):
        if hasObs and (hitsSat[i]):
            if OSP.obsFRstart[i] not in [None, 9999.]:
                if not silent:
                    print ('----------------------- Sat:' , satNames[i], '-----------------------')
                if isinstance(OSP.obsFRstart[i], float) and isinstance(OSP.obsFRend[i], float) and OSP.doFIDO:
                    # |----- Calculate the metrics for this satellite -----|
                    #try:
                        # can include ignoreSheath=True if have simulated sheath but don't want to use in metric
                        allScores, timingErr, meanVals, betterKeys, haveObsidxFull, haveObsidx = getISmetrics(ResArr, ObsData, DoY, satNames, satID=i, ignoreSheath=False, silent=silent)
                        if fAll:
                            fAll.write('|----------------------------'+satNames[i]+'----------------------------|\n')
                        oneScore, goodKeys, bestID = analyzeMetrics(ResArr, allScores, timingErr, meanVals, betterKeys, haveObsidxFull, haveObsidx, silent=silent, fIn=fAll)
                        winners[i] = bestID
                        # get the nearby BFs
                        bestS, worstS = np.min(oneScore), np.max(oneScore)
                        rngS = worstS - bestS
                        allFriends[i] = goodKeys[np.where((oneScore - bestS) <= 0.1 * rngS)[0]]
                        '''if not silent: 
                            print ('Best fit for ensemble member: ', bestID)
                            print ('')
                            print('')'''
                        # Pull out the good scores    
                        newScores = np.ones(len(ResArr.keys())) * 9999
                        newScores[goodKeys] = oneScore
                        storeScores[i,goodKeys] = oneScore
                        sumScore += newScores * satScoreWeights[i]
                        if fAll:
                            fAll.write('\n') 
                    #except:
                    #    sumScore += 9999.
                    #    print ('Error in calculating metrics for sat '+satNames[i])               
            else:
                print ('Missing FR start for ', satNames[i], ' so no metrics')
    
    # |----- Option to combine metrics from multiple satellites -----|            
    if nSat > 1:
        # |---------- Write some headers ----------|
        if not silent: 
            print ('Total score over all satellites: ')
        if fAll:
            fAll.write('|------------------------ Combined Results ------------------------|\n')

        # |---------- Pull the number for each case and make pretty to print ----------|
        for i in range(len(sumScore)):
            score = sumScore[i]
            if score > 9999:
                score = 'Fail '.rjust(8)
            else:
                score = '{:8.2f}'.format(score)
            outline = score + ' '
            for j in range(nSat):
                myScore = storeScores[j,i]
                if myScore == -9998:
                    myScore = ' Fail'.rjust(8)
                else:
                    myScore = '{:8.2f}'.format(myScore)
                outline += myScore
            # Print to screen    
            if not silent:
                 print('  ', i, outline) 
            # Print to file
            if fAll:
                
                fAll.write(str(i).rjust(4) + ' '+outline +'\n')
             
        # |---------- Sum over all satellites ----------|   
        if nEns > 1:
            minSumScore = np.min(sumScore)
            bestID = np.where(sumScore == minSumScore)[0]    
            maxSumScore = np.max(sumScore[np.where(sumScore < np.percentile(sumScore,95))])      
            worstID = np.where(sumScore == maxSumScore)[0]
            scoreRng = maxSumScore - minSumScore
        
            sumFriends = np.where(sumScore <= minSumScore + 0.1*scoreRng)[0]
            nClose = len(sumFriends)
            allFriends[-1] = sumFriends
            # Print to screen
            if not silent: 
                print ('Best total score over all satellites: ', '{:5.2f}'.format(minSumScore), 'for run number ', str(bestID[0]))
                print (str(nClose)+' members with a score within 10 percent of range (' '{:3.2f}'.format(0.1*scoreRng)+') of min score')
                outstr = ''
                for idx in np.where(sumScore <= minSumScore + 0.1*scoreRng)[0]:
                    outstr += str(idx) + ' '
                print('   members: ' + outstr + '\n')
            # Printe to file
            if fAll:
                fAll.write('\n')
                fAll.write('Best total score over all satellites: '+ '{:5.2f}'.format(minSumScore) + ' for run number ' + str(bestID[0])+'\n')
                fAll.write('Highest score for a non-fail case: ' + '{:5.2f}'.format(maxSumScore) + ' for run number ' + str(worstID[0])+'\n')
                fAll.write(str(nClose)+' members with a score within 10 percent of range (' '{:3.2f}'.format(0.1*scoreRng)+') of min score'+'\n')
                outstr = ''
                for idx in sumFriends:
                    outstr += str(idx) + ' '
                fAll.write('   members: ' + outstr + '\n')
            
    if fAll:
        fAll.close()    
    return storeScores, winners, bestID, allFriends