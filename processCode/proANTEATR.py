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
from matplotlib.ticker import FuncFormatter, PercentFormatter

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
from CME_class import cart2cart
from ANT_PUP import lenFun, getvCMEframe, whereAmI

import processOSPREI as pO
import setupPro as sP

# |------------------------------------------------------------|
# |---------------------- J map plot --------------------------|    
# |------------------------------------------------------------| 
def makeJmap(CMEarr, satLoc_unswap, startTimes, arrivals=None, satName=''):
    # Do not pass this a full ResArr unless you want ~100 overlapping
    # J maps of the essentially same CME. This is set up to take in multi
    # different CMEs and put on same figure but this not used by proOSP
    
    # |----------- set up holders for multi CME case -----------|
    allts = []
    allSheaths = []
    allFRstarts = []
    allFRends   = []
    satLoc = [[sl[2], sl[0], sl[1]] for sl in satLoc_unswap]
    
    # |----------- Loop through the CMEs -----------|
    for j in range(len(CMEarr)):        
        myCME    = CMEarr[j]
        mySatLoc = satLoc[j]
        startt   = startTimes[j]
        nANT     = len(myCME.ANTtimes)
        CMEpos   = [myCME.FClats[-1], myCME.FClons[-1], myCME.FCtilts[-1], 0]

        mySheath  = []
        myFRstart = []
        myFRend   = []
        myDTs     = []
        # |----------- Loop through the time steps -----------|
        for i in range(nANT):
            # CME properties at this time step
            thisR, thisAW, thisAWp, thisDelAx, thisDelCS = myCME.ANTrs[i], myCME.ANTAWs[i]*dtor, myCME.ANTAWps[i]*dtor, myCME.ANTdelAxs[i], myCME.ANTdelCSs[i]
            
            # |----------- Get the sat position relative to CME -----------|
            Epos1 = SPH2CART(mySatLoc)
            temp = rotz(Epos1, -CMEpos[1])
            temp2 = roty(temp, CMEpos[0])
            temp3 = rotx(temp2, 90.-CMEpos[2])
            # this is assuming constant yaw for now, to fix replace CMEpos[3] with yaw(t)
            temp3[0] -= thisR*7e10
            Epos2 = roty(temp3, CMEpos[3])
            Epos2[0] += thisR*7e10
            SatPos = np.array(Epos2)/7e10
        
            # |----------- Calculate the CME lengths -----------|
            CMElens = np.zeros(7)
            CMElens[0] = thisR
            CMElens[4] = np.tan(thisAWp) / (1 + thisDelCS * np.tan(thisAWp)) * CMElens[0]
            CMElens[3] = thisDelCS * CMElens[4]
            CMElens[6] = (np.tan(thisAW) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + thisDelAx * np.tan(thisAW))  
            CMElens[5] = thisDelAx * CMElens[6]
            CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
            CMElens[1] = CMElens[2] * np.tan(thisAW)
    
            # |----------- Make a guess at what psi the sat intersects -----------|
            satR = np.sqrt(np.sum(SatPos**2))
            guessPsi = np.abs(np.arccos(np.dot(SatPos, [1,0,0])/satR))
            # don't trust the sign on this so check if should be +/-
            # get axis pos for both and check which closer
            thisAxA = np.array([CMElens[2] + thisDelAx * CMElens[6] * np.cos(guessPsi), 0.,  0.5 * CMElens[6] * (np.sin(guessPsi) + np.sqrt(1 - np.cos(guessPsi)))])
            thisAxB = np.array([CMElens[2] + thisDelAx * CMElens[6] * np.cos(-guessPsi), 0.,  -0.5 * CMElens[6] * (np.sin(guessPsi) + np.sqrt(1 - np.cos(guessPsi)))])
            distA = np.sum((thisAxA-SatPos)**2)
            distB = np.sum((thisAxB-SatPos)**2)
            if distB < distA:
                guessPsi = - guessPsi
    
            # |----------- Refine the guess by brute force -----------|
            delPsi = 30 * dtor
            psis = np.linspace(guessPsi-delPsi, guessPsi+delPsi, 101)
            sns = np.sign(psis)
            xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(psis)
            zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(psis)) + np.sqrt(1 - np.cos(np.abs(psis))))   
            dists2 = ((SatPos[0] - xFR)**2 + SatPos[1]**2 + (SatPos[2] - zFR)**2) / (CMElens[4])**2
            thismin = np.min(dists2)
            thisPsi = psis[np.where(dists2 == np.min(dists2))][0]
            sn = np.sign(thisPsi)
            if sn == 0: sn = 1
            thisPsi = np.abs(thisPsi)
            
            # |----------- Get distance at the axis -----------|
            xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(thisPsi)
            zFR = 0.5 * sn * CMElens[6] * (np.sin(np.abs(thisPsi)) + np.sqrt(1 - np.cos(np.abs(thisPsi))))   
            rAx = np.sqrt(xFR**2 + zFR**2)
    
            # |----------- Approx width using angles and local cylinder approx -----------|
            # Get the normal vector
            if thisPsi == 0: thisPsi = 0.00001
            Nvec = [0.5 * sn*(np.cos(thisPsi) + 0.5*np.sin(thisPsi)/np.sqrt(1-np.cos(thisPsi))), 0., thisDelAx * np.sin(thisPsi)] 
            Nmag = np.sqrt(Nvec[0]**2+Nvec[2]**2)
            norm = np.array(Nvec) / Nmag
            # Determine angle from radial dir at sat
            rhat = SatPos / satR
            cosAng = np.abs(np.dot(rhat, norm))
            newWid = CMElens[3] / cosAng
            if OSP.doPUP:
                mySheath.append(rAx+newWid + myCME.PUPwids[i])
            myFRstart.append(rAx+newWid)
            myFRend.append(rAx-newWid)
            myDTs.append(startt + datetime.timedelta(days= myCME.ANTtimes[i]))
            
        # |----------- Add to holders -----------|
        allSheaths.append(np.array(mySheath))
        allFRstarts.append(np.array(myFRstart))
        allFRends.append(np.array(myFRend))
        allts.append(np.array(myDTs))
    
    # |----------- Set up figure -----------|
    fig, ax = plt.subplots(1,1, figsize=(10,5))
    for j in range(len(allts)):
        # |----------- Add sheath if included -----------|
        if OSP.doPUP:
            plt.fill_between(allts[j], allSheaths[j], allFRstarts[j], color='#882255', alpha=0.25 )
        # |----------- Add flux rope -----------|    
        plt.fill_between(allts[j], allFRstarts[j], allFRends[j], color='k', alpha=0.25 )
        if OSP.doPUP:
            plt.plot(allts[j], allSheaths[j], '#882255', lw=2)
        plt.plot(allts[j], allFRstarts[j], 'k',lw=2)
        plt.plot(allts[j], allFRends[j], 'k',lw=2)
        
    # |----------- Plot vertical time of arrival lines -----------|
    if arrivals != None:
        yl = ax.get_ylim()
        for item in arrivals:
            plt.plot([item, item], yl, '--', color='lightgray', zorder=0)
        ax.set_ylim(yl)
            
    # |----------- Plot horizontal line at sat distance -----------|
    xl = ax.get_xlim()
    plt.plot(xl, [mySatLoc[0], mySatLoc[0]], 'k--')    
    
    # |----------- Prettify and save -----------|
    ax.set_xlim(xl)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M"))
    ax.set_ylabel('Distance (R$_S$)')
    plt.savefig(OSP.Dir+'/fig'+str(CMEarr[0].name)+'_Jmap_'+satName+'.'+pO.figtag)



# |------------------------------------------------------------|
# |------------ Profiles including PUP sheath -----------------|    
# |------------------------------------------------------------|
def makexy(ResArr, key):
    thisRes = ResArr[key]
    xs = [thisRes.ANTrs+thisRes.PUPwids,thisRes.ANTrs,  thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, ]
    ys = [thisRes.ANTtimes*24, thisRes.ANTtimes*24, thisRes.ANTAWs, thisRes.ANTAWps, thisRes.ANTdelAxs, thisRes.ANTdelCSs, thisRes.PUPBs, thisRes.ANTBtors,  thisRes.PUPvshocks, thisRes.ANTvFs, thisRes.ANTvCSrs, thisRes.PUPwids, thisRes.ANTCMEwids, thisRes.PUPns, thisRes.ANTns, np.power(10,thisRes.PUPlogTs), np.power(10,thisRes.ANTlogTs)]       
    return xs, ys
     
def makePUPplot(ResArr, nEns, satID=0, BFs=[None], satCols=None, satNames=None):
    # |----------- Set up figure and color system -----------|
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(16,8))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[0,2], axes[0,2], axes[0,3], axes[0,3], axes[1,0], axes[1,0], axes[1,0], axes[1,1], axes[1,1], axes[1,2],  axes[1,2], axes[1,3], axes[1,3]]
    c1 = '#882255'
    c2 = '#332288'
    
    # |----------- Get r range -----------|
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # |----------- Get number of impacts -----------|
    nImp = 0
    hits = []
    for i in range(nEns):
        if (ResArr[i].ANTshidx[satID]):
            nImp += 1
            hits.append(i)

    # |----------- Set up arrays to hold spline results -----------|
    nParams = 17
    fakers = np.linspace(rStart,rEnd,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, nParams])
    means = np.zeros([100, nParams])
    stds  = np.zeros([100, nParams])
    lims  = np.zeros([100, 2, nParams])
    
    # |----------- Fill in spline results -----------| 
    i = 0     
    for key in hits:
        thexs, theParams = makexy(ResArr, key)
        thisRes = ResArr[key]

        if nEns > 1:
            # Fit a spline to data since may be different lengths since take different times
            # Sometimes the spline blows up if the member has an early impact and shorter r 
            # array so spline is unstable at farther distances
            for k in range(nParams):
                thefit = CubicSpline(thexs[k], theParams[k],bc_type='natural')
                splineVals[i,:, k] = thefit(fakers)
                tooFar = np.where(fakers > thisRes.ANTrs[-1])[0]
                if len(tooFar) != 0:
                    splineVals[i,tooFar,k] = splineVals[i,tooFar[0]-1,k]
                if k == 8:
                    mySV = splineVals[i,:, k]
                    mySV[np.where(mySV <1)] = 0
                    splineVals[i,:, k] = mySV
            i += 1
    
    # |----------- Determine which color a param is -----------|
    for i in range(nParams)[::-1]:
        col = c1
        if i == 10: 
            col = 'k'
        else:
            calci = i
            if i > 10:
                calci = i - 1
            if calci % 2 == 0: col = c2
            
        # |----------- Calculate metrics -----------|
        if nEns > 1:
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            
            # |----------- Remove the 0 points from the vShock spline fit -----------|
            if i==8:
                has0s = []
                for j in range(100):
                    # Check if vSh is above vCME, if not toss it
                    above0 = np.where(splineVals[:,j,8] > means[j,9])[0]
                    if len(above0) > 0:
                        has0s.append(j)
                        means[j,8] = np.mean(splineVals[above0,j,8])
                        stds[j,8] = np.std(splineVals[above0,j,8])
                        lims[j,0,8] = np.max(splineVals[above0,j,8])
                        lims[j,1,8] = np.min(splineVals[above0,j,8])
                has0s = np.array(has0s)

            # |----------- Color in the ranges -----------|
            rmidx = 1
            axes[i].fill_between(fakers[:-rmidx], lims[:-rmidx,0,i], lims[:-rmidx,1,i], color=col, alpha=0.25) 
            axes[i].fill_between(fakers[:-rmidx], means[:-rmidx,i]+stds[:-rmidx,i], means[:-rmidx,i]-stds[:-rmidx,i], color=col, alpha=0.25)
        
        if BFs[0] or (BFs[0] == 0):
            for idx in range(len(BFs)):
                j = BFs[idx]
                myCol = satCols[idx]
                allidxs = np.where(BFs == j)[0]
                if len(allidxs) > 1:
                    myCol = satCols[allidxs[0]]
                thisRes = ResArr[j]
                thexs, theParams = makexy(ResArr, j)
                if i != 10:
                    if i == 0:
                        axes[i].plot(thexs[i],theParams[i], linewidth=2, color=myCol, zorder=3, label=satNames[idx])
                    else:
                        axes[i].plot(thexs[i],theParams[i], linewidth=2, color=myCol, zorder=3)
            fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satCols))
        # |-------------------- Plot the main line if not vCS --------------------|
        # |----------- (use to plot vCS but easier to ignore than rm)  -----------|
        thisRes = ResArr[0]
        thexs, theParams = makexy(ResArr, 0)
        if i != 10:
            axes[i].plot(thexs[i],theParams[i], linewidth=4, color=col, zorder=4)
    
    # |----------- Add text with final result -----------|
    # |-------- Currently only for ens not single -------|
    labels = ['AT$_{Sh}$', 'AT$_{CME}$', 'AW', 'AW$_{\\perp}$', '$\\delta_{Ax}$', '$\\delta_{CS}$', 'B$_{sh}$', 'B$_{CME}$', 'v$_{Sh}$', 'v$_{CME}$', 'v$_{Exp}$', 'Wid$_{sh}$', 'Wid$_{CME}$', 'n$_{sh}$', 'n$_{CME}$', 'log(T$_{Sh}$)', 'log(T$_{CME}$)']
    deg = '$^\\circ$'
    units = ['hr', 'hr', deg, deg, '', '', 'nT', 'nT', 'km/s', 'km/s', 'km/s', 'R$_S$', 'R$_S$', 'cm$^{-3}$', 'cm$^{-3}$', 'K', 'K']
        
    if nEns > 1:
        all_Params = [[] for i in range(nParams)]
        for key in hits:
            thisRes = ResArr[key]
            theParams = [thisRes.ANTtimes*24, thisRes.ANTtimes*24, thisRes.ANTAWs, thisRes.ANTAWps, thisRes.ANTdelAxs, thisRes.ANTdelCSs, thisRes.PUPBs, thisRes.ANTBtors,  thisRes.PUPvshocks, thisRes.ANTvFs, thisRes.ANTvCSrs, thisRes.PUPwids, thisRes.ANTCMEwids, thisRes.PUPns, thisRes.ANTns, thisRes.PUPlogTs, thisRes.ANTlogTs]        
            for i in range(nParams):
                shidx, fridx = thisRes.ANTshidx[satID], thisRes.ANTFRidx[satID]
                if i == 0:
                    if shidx:
                        all_Params[i].append(theParams[i][thisRes.ANTshidx[satID]]) 
                elif i == 1:
                    if fridx:
                        all_Params[i].append(theParams[i][thisRes.ANTFRidx[satID]]) 
                else:
                    all_Params[i].append(theParams[i][-1])   
        for i in range(nParams):
            thefit = norm.fit(all_Params[i])
            endMean = thefit[0]
            endSTD  = thefit[1]    
            ytext = 0.96
            col = c1
            if i == 10: 
                col = 'k'
                ytext = 0.82
            else:
                calci = i
                if i > 10:
                    calci = i - 1
                if calci % 2 == 1: ytext = 0.89
                if calci % 2 == 0: col = c2
            axes[i].text(0.97, ytext, labels[i]+': '+'{:4.1f}'.format(endMean)+'$\\pm$'+'{:.2f}'.format(endSTD)+' '+units[i], transform=axes[i].transAxes, color=col, horizontalalignment='right', verticalalignment='center', zorder=5)
    
    # |----------- Prettify and save -----------|
    # Expand AW axis to add room for text
    yl = axes[3].get_ylim()
    axes[3].set_ylim([yl[0], 1.1*yl[1]])
    # Add labels
    ylabels = ['Time (hr)', 'AW ('+deg+')', '$\\delta$', 'B (nT)', 'v (km/s)', 'Width (R$_S$)', 'n (cm$^{-3}$)', 'log(T) (T)']
    for i in range(8):
        axes[2*i+1].set_ylabel(ylabels[i])
        if i > 3:
            axes[2*i+1].set_xlabel('R (au)')   
    for i in [7,13,15]:
        axes[i].set_yscale('log')
    axes[0].set_xlim([rStart, rEnd])
    plt.subplots_adjust(wspace=0.3, hspace=0.01,left=0.06,right=0.99,top=0.94,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANTPUP.'+pO.figtag)
    plt.close() 



# |------------------------------------------------------------|
# |------------- Profiles without PUP sheath ------------------|    
# |------------------------------------------------------------|
def makeDragless(ResArr, nEns, BFs=[None], satCols=None, satNames=None):
    # |----------- Set up figure and color system -----------|
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[1,0], axes[1,0], axes[1,1], axes[1,1], axes[0,1]]    
    col1 = '#882255'
    col2 = '#332288'
    col3 = '#44AA99'    
    c1   = [col1, col2, col1, col2, col1, col2, col1, col2, col3]
    
    # |----------- Get domain of runs -----------|
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # |----------- Get number of members that don't fail -----------|
    nImp = 0
    hits = []
    for i in range(nEns):
        if (not ResArr[i].fail):
            nImp += 1
            hits.append(i)
            
    # |----------- Add extra axis if including yaw in plot-----------|
    npts = 8
    if OSP.simYaw:
        npts = 9
        axes[8] = axes[2].twinx()
    
    # |----------- Arrays to hold spline results -----------|
    fakers = np.linspace(rStart,rEnd,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, npts])
    means = np.zeros([100, npts])
    stds  = np.zeros([100, npts])
    lims  = np.zeros([100, 2, npts])
    
    axes[7] = axes[6].twinx()
    
    # |----------- Loop through ensemble results repackaging  -----------|
    if nEns > 1:
        i = 0
        # |----------- Fit spline to data and get results at uniform rs  -----------|
        for key in hits:
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTAWs,bc_type='natural')
            splineVals[i,:, 0] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTAWps,bc_type='natural')
            splineVals[i,:, 1] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTdelAxs,bc_type='natural')
            splineVals[i,:, 2] = thefit(fakers)   
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTdelCSs,bc_type='natural')
            splineVals[i,:, 3] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvFs,bc_type='natural')
            splineVals[i,:, 4] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvCSrs,bc_type='natural')
            splineVals[i,:, 5] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTBtors,bc_type='natural')
            splineVals[i,:, 6] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTlogTs,bc_type='natural')
            splineVals[i,:, 7] = thefit(fakers)   
            if OSP.simYaw:
                thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTyaws,bc_type='natural')
                splineVals[i,:, 8] = thefit(fakers)
            # correct for profiles that are too short    
            tooFar = np.where(fakers > ResArr[key].ANTrs[-1])[0]
            if len(tooFar) != 0:
                for k in range(npts):
                    splineVals[i,tooFar,k] = splineVals[i,tooFar[0]-1,k]                    
            i += 1
        
        # |----------- Calculate overall properties  -----------|
        for i in range(npts):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            # |----------- Shade in the plot -----------|
            zorder = 2
            if i == 8: zorder = 0
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color=c1[i], alpha=0.25, zorder=zorder) 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color=c1[i], zorder=zorder, alpha=0.25) 
    
    # |----------- Add in the individual best fits -----------|
    lw = 2
    if BFs[0] or (BFs[0] == 0):            
            for i in range(len(BFs)):
                myCol = satCols[i]
                allidxs = np.where(BFs == BFs[i])[0]
                if len(allidxs) > 1:
                    myCol = satCols[allidxs[0]]
                    
                idx = BFs[i]
                axes[0].plot(ResArr[idx].ANTrs, ResArr[idx].ANTAWs, linewidth=lw, color=myCol, zorder=3, label=satNames[i])
                axes[1].plot(ResArr[idx].ANTrs, ResArr[idx].ANTAWps, linewidth=lw, color=myCol, zorder=3)
                axes[2].plot(ResArr[idx].ANTrs, ResArr[idx].ANTdelAxs, linewidth=lw, color=myCol, zorder=3)
                axes[3].plot(ResArr[idx].ANTrs, ResArr[idx].ANTdelCSs, linewidth=lw, color=myCol, zorder=3)
                axes[4].plot(ResArr[idx].ANTrs, ResArr[idx].ANTvFs, linewidth=lw, color=myCol, zorder=3)
                axes[5].plot(ResArr[idx].ANTrs, ResArr[idx].ANTvCSrs, linewidth=lw, color=myCol, zorder=3)
                axes[6].plot(ResArr[idx].ANTrs, ResArr[idx].ANTBtors, linewidth=lw, color=myCol, zorder=3)
                axes[7].plot(ResArr[idx].ANTrs, ResArr[idx].ANTlogTs, linewidth=lw, color=myCol, zorder=3)    
                if OSP.simYaw:
                    axes[8].plot(ResArr[idx].ANTrs, ResArr[0].ANTyaws, linewidth=4, color=col3, zorder=3) 
            fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satCols))
                    
    # |----------- Add the seed profile -----------|
    axes[0].plot(ResArr[0].ANTrs, ResArr[0].ANTAWs, linewidth=4, color=col1, zorder=4)
    axes[1].plot(ResArr[0].ANTrs, ResArr[0].ANTAWps, linewidth=4, color=col2, zorder=4)
    axes[2].plot(ResArr[0].ANTrs, ResArr[0].ANTdelAxs, linewidth=4, color=col1, zorder=4)
    axes[3].plot(ResArr[0].ANTrs, ResArr[0].ANTdelCSs, linewidth=4, color=col2, zorder=4)
    axes[4].plot(ResArr[0].ANTrs, ResArr[0].ANTvFs, linewidth=4, color=col1, zorder=4)
    axes[5].plot(ResArr[0].ANTrs, ResArr[0].ANTvCSrs, linewidth=4, color=col2, zorder=4)
    axes[6].plot(ResArr[0].ANTrs, ResArr[0].ANTBtors, linewidth=4, color=col1, zorder=4)
    axes[7].plot(ResArr[0].ANTrs, ResArr[0].ANTlogTs, linewidth=4, color=col2, zorder=4)    
    if OSP.simYaw:
        axes[8].plot(ResArr[0].ANTrs, ResArr[0].ANTyaws, linewidth=4, color=col3, zorder=4)    
        
    # |----------- Add in text with the final values -----------|
    degree = '$^\\circ$'
    # Ensemble case with uncertainty
    if nEns > 1:
        all_AWs, all_AWps, all_delAxs, all_delCSs, all_vFs, all_vCSrs, all_Btors, all_Ts, all_yaws = [], [], [], [], [], [], [], [], []
        for key in ResArr.keys():
            thisIdx = ResArr[key].ANTFRidx[0]            
            # with multi sat hits is not necessarily hit for all
            if thisIdx != None:
                all_AWs.append(ResArr[key].ANTAWs[thisIdx])
                all_AWps.append(ResArr[key].ANTAWps[thisIdx])
                all_delAxs.append(ResArr[key].ANTdelAxs[thisIdx])
                all_delCSs.append(ResArr[key].ANTdelCSs[thisIdx])
                all_vFs.append(ResArr[key].ANTvFs[thisIdx])
                all_vCSrs.append(ResArr[key].ANTvCSrs[thisIdx])
                all_Btors.append(ResArr[key].ANTBtors[thisIdx])
                all_Ts.append(ResArr[key].ANTlogTs[thisIdx])
                if OSP.simYaw:
                    all_yaws.append(ResArr[key].ANTyaws[thisIdx])
        if all_AWs != []:
            fitAWs = norm.fit(all_AWs)
            fitAWps = norm.fit(all_AWps)
            fitdelAxs = norm.fit(all_delAxs)
            fitdelCSs = norm.fit(all_delCSs)
            fitvFs = norm.fit(all_vFs)
            fitvCSrs = norm.fit(all_vCSrs)
            fitBtors = norm.fit(all_Btors)
            fitTs    = norm.fit(all_Ts)
            if OSP.simYaw:
                fityaws = norm.fit(all_yaws)

            axes[0].text(0.97, 0.96, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\\pm$'+'{:2.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes, color=col1)
            axes[1].text(0.97, 0.9,  'AW$_{\\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\\pm$'+'{:2.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color=col2)
            try:
                axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
            except:
                pass
            axes[2].text(0.97, 0.96, '$\\delta_{Ax}$: '+'{:4.2f}'.format(fitdelAxs[0])+'$\\pm$'+'{:4.2f}'.format(fitdelAxs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes, color=col1)
            axes[3].text(0.97, 0.9, '$\\delta_{CS}$: '+'{:4.2f}'.format(fitdelCSs[0])+'$\\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color=col2)
            axes[4].text(0.97, 0.96, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\\pm$'+'{:2.0f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes, color=col1)
            axes[5].text(0.97, 0.9, 'v$_{Exp}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\\pm$'+'{:2.0f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color=col2)
            axes[6].text(0.97, 0.96, 'B: '+'{:4.1f}'.format(fitBtors[0])+'$\\pm$'+'{:3.1f}'.format(fitBtors[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes, color=col1)
            axes[7].text(0.97, 0.9,  'log(T): '+'{:4.1f}'.format(fitTs[0])+'$\\pm$'+'{:3.1f}'.format(fitTs[1])+' K', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color=col2)     
            if OSP.simYaw:
                axes[8].text(0.97, 0.84,  'yaw: '+'{:4.1f}'.format(fityaws[0])+'$\\pm$'+'{:3.1f}'.format(fityaws[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes, color=col3)
                  
        # Single run version
        else:
            thisIdx = ResArr[0].ANTFRidx[0]
            if thisIdx:
                axes[0].text(0.97, 0.96, 'AW: '+'{:4.1f}'.format(ResArr[0].ANTAWs[thisIdx])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes, color=col1)
                axes[1].text(0.97, 0.9,  'AW$_{\\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTAWps[thisIdx])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color=col2)
                axes[2].text(0.97, 0.96, '$\\delta_{Ax}$: '+'{:4.2f}'.format(ResArr[0].ANTdelAxs[thisIdx]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes, color=col1)
                axes[3].text(0.97, 0.9, '$\\delta_{CS}$: '+'{:4.2f}'.format(ResArr[0].ANTdelCSs[thisIdx]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color=col2)
                axes[4].text(0.97, 0.96, 'v$_F$: '+'{:4.0f}'.format(ResArr[0].ANTvFs[thisIdx])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes, color=col1)
                axes[5].text(0.97, 0.9, 'v$_{Exp}$: '+'{:4.0f}'.format(ResArr[0].ANTvCSrs[thisIdx])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color=col2)
                axes[6].text(0.97, 0.96, 'B$_{t}$: '+'{:4.1f}'.format(ResArr[0].ANTBtors[thisIdx])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes, color=col1)
                axes[7].text(0.97, 0.9,  'log(T): '+'{:4.1f}'.format(ResArr[0].ANTlogTs[thisIdx])+' K', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color=col2) 
                if OSP.simYaw:
                    axes[8].text(0.97, 0.84,  'yaw: '+'{:4.1f}'.format(ResArr[0].ANTyaws[thisIdx])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes, color=col3) 
    
    # |----------- Expand AW axis to add room for text -----------|
    yl = axes[0].get_ylim()
    axes[0].set_ylim([yl[0], 1.1*yl[1]])
                     
    # |----------- Labels -----------|
    axes[0].set_ylabel('AW, AW$_{\\perp}$ ('+degree+')')
    axes[2].set_ylabel('$\\delta_{Ax}$, $\\delta_{CS}$')
    axes[4].set_ylabel('v$_F$, v$_{Exp}$ (km/s)')
    axes[6].set_ylabel('B (nT)')
    axes[7].set_ylabel('log(T) (K)')
    if OSP.simYaw:
        axes[8].set_ylabel('Yaw ('+degree+')')
    axes[7].set_ylim([1,5.5])
    axes[6].set_ylim([1,1e5])
    if OSP.simYaw and (nEns>1):
        axes[8].set_ylim([-5*fityaws[1],5*fityaws[1]])
    axes[4].set_xlabel('Distance (R$_S$)')
    axes[6].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([rStart, rEnd])
    axes[6].set_yscale('log')
    
    plt.subplots_adjust(hspace=0.1, wspace=0.27,left=0.1,right=0.93,top=0.95,bottom=0.1)
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_DragLess.'+pO.figtag)
    plt.close() 
    


# |------------------------------------------------------------|
# |-------------------- ANTEATR histogram ---------------------|    
# |------------------------------------------------------------|
def makeAThisto(ResArr, dObj=None, DoY=None, satID=0, BFs=[None], satCols=None, satNames=['']):
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
    
    # |----------- Set up figure and repackage axis -----------|
    fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True)
    axes[0,0].set_ylabel('Counts')
    axes[1,0].set_ylabel('Counts')
    axes[2,0].set_ylabel('Counts')
    axes = [axes[1,1], axes[1,2], axes[0,0], axes[2,0], axes[2,1], axes[0,1], axes[1,0], axes[0,2], axes[2,2]]

    # |----------- Set up arrays to collect results -----------|
    all_vFs, all_vExps, all_TTs, all_durs, all_Bfs, all_Bms, all_ns, all_Kps, all_Ts, FC_times  = [], [], [], [], [], [], [], [], [], []
    allidx = []
    for key in ResArr.keys(): 
        if (not ResArr[key].FIDOmiss[satID]) and (not ResArr[key].fail):
            # figure out when hits FR, may not be last pt if doing internal FIDO
            if ResArr[key].sheathOnly[satID]:
                thisidx = ResArr[key].ANTshidx[satID]
            else:
                thisidx = ResArr[key].ANTFRidx[satID]
            if thisidx is not None:
                all_vFs.append(float(ResArr[key].ANTvFs[thisidx]))
                all_vExps.append(float(ResArr[key].ANTvCSrs[thisidx]))
                all_TTs.append(float(ResArr[key].ANTtimes[thisidx]))    
                all_durs.append(float(ResArr[key].ANTdur[satID]))
                all_Bfs.append(float(ResArr[key].ANTBpols[thisidx]))
                all_Bms.append(float(ResArr[key].ANTBtors[thisidx]))
                all_ns.append(float(ResArr[key].ANTns[thisidx]))
                all_Kps.append(float(ResArr[key].ANTKp0[satID]))
                all_Ts.append(float(ResArr[key].ANTlogTs[thisidx]))
                allidx.append(key)
    allidx = np.array(allidx)
    
    # |----------- Reorder things and set up labels -----------|
    # At some point changed the order things are plotted which is why this and the axes
    # array are in messy order. Could fix but not a priorty since works, just ugly
    ordData = np.array([all_vFs, all_vExps, all_TTs, all_Bfs, all_Bms, all_durs, all_Ts, all_ns, all_Kps]) 
    names = ['v$_F$ (km/s)', 'v$_{Exp}$ (km/s)', 'Transit Time (days)', 'B$_F$ (nT)', 'B$_C$ (nT)', 'Duration (hours)', 'log$_{10}$T (K)','n (cm$^{-3}$)', 'Kp']
    units = ['km/s', 'km/s', 'days', 'nT', 'nT', 'hr', 'log$_{10}$ K','cm$^{-3}$', '']
    fmts = ['{:.0f}','{:.0f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}']

    # |----------- Loop through each param and add histo to panel -----------|
    maxcount = 0
    for i in range(9):
        theseData = np.array(ordData[i])
        mean, std = np.mean(theseData), np.std(theseData)
        cutoff = 5 *std
        if i in [3,4]: cutoff = 3 * std
        newData = theseData[np.where(np.abs(theseData - mean) < cutoff)[0]]
        n, bins, patches = axes[i].hist(newData, bins=10, color='lightgray', histtype='bar', ec='black')
        axes[i].set_xlabel(names[i])
        maxn = np.max(n)
        if maxn > maxcount: maxcount = maxn
        # |----------- Add label showing final results -----------|
        if i != 2:
            axes[i].text(0.97, 0.92, fmts[i].format(mean)+'$\\pm$'+fmts[i].format(std)+ ' '+units[i], horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 
        else:
            if not OSP.noDate:
                yr = dObj.year
                base = datetime.datetime(yr, 1, 1, 0, 0)
                date = base + datetime.timedelta(days=(DoY+mean))   
                dateLabel = date.strftime('%b %d %H:%M')
                axes[i].text(0.97, 0.92, dateLabel+'$\\pm$'+'{:.1f}'.format(std*12)+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 
            axes[i].text(0.97, 0.82, fmts[i].format(mean)+'$\\pm$'+'{:.2f}'.format(std)+' days', horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 
                       
    #|--------------- Add BF lines ---------------|
    if BFs[0] or (BFs[0] == 0):
        allys = []
        for i in range(9):
            allys.append(axes[i].get_ylim())
    
        keycount = 0
        for key in BFs:
            # it might not be in allidx if bad at other sat
            if key in allidx:
                BFidx = np.where(BFs ==key)[0]
                mycol = satCols[keycount]
                if len(BFidx) > 1:
                    mycol = satCols[BFidx[0]]
                allkey = np.where(allidx == key)[0]
                for i in range(9):
                    myx = ordData[i][allkey[0]]
                    if i == 2:
                        axes[i].plot([myx, myx], [allys[i][0], allys[i][1]/1.1], '--', color=mycol, label=satNames[keycount])
                    else:
                        axes[i].plot([myx, myx], allys[i], '--', color=mycol)
            keycount += 1
    
        for i in range(9):
            axes[i].set_ylim(allys[i])  
        fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satCols))     

    # |----------- Make some room above lines so text can be read -----------|
    for i in range(9): axes[i].set_ylim(0, maxcount*1.2)
        
    # |----------- Prettify and save -----------|
    plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANThist'+satName+'.'+pO.figtag)
    plt.close() 
    


# |------------------------------------------------------------|
# |-------------------- ANTEATR contours ----------------------|    
# |------------------------------------------------------------|
def makeContours(ResArr, nEns, nFails, calcwid=95, plotwid=40, satID=0, satLocs=None, satNames=[''], allSats=None, satCols=None):
    # Start by filling in the area that corresponds to the CME in a convenient frame
    # then rotate it the frame where Earth is at [0,0] at the time of impact for all CMEs
    # using its exact position in "real" space (which will vary slightly between CMEs).
    # Also derive parameters based on Earth's coord system if there were changes in it's
    # position (i.e. vr for rhat corresponding to each potential shifted lat/lon)
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
        
    if allSats:
        satLoc = satLocs[satID]
    else:
        if not satLocs:
            try:
                satLoc = OSP.satPos
            except:
                satLoc = [0,0]
        else:
            satLoc = satLocs
        
            
    # |----------- Simulation parameters -----------|
    ngrid = 2*plotwid+1
    ncalc = 2*calcwid+1
    nThings = 39
    shiftx, shifty = calcwid, calcwid
    counter = 0
    
    # |----------- Get number of actual impacts -----------|
    hits = []
    for i in range(nEns):
        if (not ResArr[i].FIDOmiss[satID]) & (not ResArr[i].fail) & (ResArr[i].ANTFRidx[satID] is not None):
            hits.append(i)
    allGrid = np.zeros([len(hits), ngrid, ngrid, nThings])
    
    # |----------- Loop through each ensemble member -----------|
    for key in hits:
        newGrid = np.zeros([ncalc,ncalc,nThings])
        
        # |----------- Pull in from ResArr -----------|
        if OSP.doFC:
            thisLat = ResArr[key].FClats[-1]
            thisLon = ResArr[key].FClons[-1] 
            thisTilt  = ResArr[key].FCtilts[-1]
        else:
            thisLat = float(OSP.input_values['CMElat'])
            thisLon = float(OSP.input_values['CMElon'])
            thisTilt = float(OSP.input_values['CMEtilt'])
        
        # |----------- Get values at time of first impact  -----------|
        thisidx   = ResArr[key].ANTFRidx[satID]  
        thisAW    = ResArr[key].ANTAWs[thisidx]
        thisAWp   = ResArr[key].ANTAWps[thisidx]
        thisR     = ResArr[key].ANTrs[thisidx]
        thisDelAx = ResArr[key].ANTdelAxs[thisidx]
        thisDelCS = ResArr[key].ANTdelCSs[thisidx]
        thisDelCA = ResArr[key].ANTdelCSAxs[thisidx]
        thesevs = np.array([ResArr[key].ANTvFs[thisidx], ResArr[key].ANTvEs[thisidx], ResArr[key].ANTvBs[thisidx], ResArr[key].ANTvCSrs[thisidx], ResArr[key].ANTvCSps[thisidx], ResArr[key].ANTvAxrs[thisidx], ResArr[key].ANTvAxps[thisidx]]) 
        thisB0    = ResArr[key].ANTB0s[thisidx] * np.sign(float(OSP.input_values['FRB']))
        thisTau   = ResArr[key].ANTtaus[thisidx]
        thisCnm   = ResArr[key].ANTCnms[thisidx]
        thisPol   = int(float(OSP.input_values['FRpol']))
        thislogT  = ResArr[key].ANTlogTs[thisidx]
        thisn     = ResArr[key].ANTns[thisidx]
        
        # # |----------- Calculate the CME lengths -----------|
        # CMElens = [CMEnose, rEdge, rCent, rr, rp, Lr, Lp]
        CMElens = np.zeros(7)
        CMElens[0] = thisR
        CMElens[4] = np.tan(thisAWp*dtor) / (1 + thisDelCS * np.tan(thisAWp*dtor)) * CMElens[0]
        CMElens[3] = thisDelCS * CMElens[4]
        CMElens[6] = (np.tan(thisAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + thisDelAx * np.tan(thisAW*dtor))  
        CMElens[5] = thisDelAx * CMElens[6]
        CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
        CMElens[1] = CMElens[2] * np.tan(thisAW*dtor)
        
        # |----------- Find the axis location -----------|
        nFR = 31 # axis resolution
        thetas = np.linspace(-math.pi/2, math.pi/2, nFR)    
        sns = np.sign(thetas)
        xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(thetas)
        zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(thetas)) + np.sqrt(1 - np.cos(np.abs(thetas))))   
        top = [xFR[-1], 0.,  zFR[-1]+CMElens[3]]
        topSPH = CART2SPH(top)
        # br is constant but axis x changes -> AWp varies
        varyAWp = np.arctan(CMElens[4] / xFR)/dtor 
        # get the axis coords in spherical
        axSPH = CART2SPH([xFR,0.,zFR])
        # get the round outer edge from the CS at last thetaT
        thetaPs = np.linspace(0, math.pi/2,21)
        xEdge = xFR[-1] 
        yEdge = CMElens[4] * np.sin(thetaPs)
        zEdge = zFR[-1] + (thisDelCS * CMElens[4] * np.cos(thetaPs))
        edgeSPH = CART2SPH([xEdge, yEdge, zEdge])
        
        # |----------- Create the maps from latitude to other variables -----------|
        lat2theT =  CubicSpline(axSPH[1],thetas/dtor,bc_type='natural')
        lat2AWp = CubicSpline(axSPH[1],varyAWp,bc_type='natural')
        lat2xFR = CubicSpline(axSPH[1],xFR,bc_type='natural')
        minlat, maxlat = int(round(np.min(axSPH[1]))), int(round(np.max(axSPH[1])))
        minlon, maxlon = shiftx-int(round(thisAWp)),shiftx+int(round(thisAWp))
        lat2AWpEd = CubicSpline(edgeSPH[1][::-1],edgeSPH[2][::-1],bc_type='natural')    
        lon2TP = CubicSpline(edgeSPH[2],thetaPs/dtor,bc_type='natural') 
        # |----------- Check lat values to make sure stays in range -----------|
        maxn = ncalc-1
        if minlat+shifty < 0: minlat = -shifty
        if maxlat+1+shifty > maxn: maxlat = maxn-1-shifty
        minlat2, maxlat2 = int(round(maxlat)), int(round(topSPH[1]))  
        if maxlat2+1+shifty > maxn: maxlat2 = maxn-1-shifty
        
        # |----------- Loop through the latitude and fill in points that are inside CME -----------|
        for i in range(minlat, maxlat+1):
            # |----------- Calc which lons should be filled -----------|
            idy =  i+shifty
            nowAWp = np.round(lat2AWp(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            # Fill from minlon to maxlon
            newGrid[idy, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
                       
            # |----------- Pad the checked region by a few deg  -----------|
            newGrid[idy,  np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               
            # |----------- Calculate parametric theta T within the CME  -----------|
            # ThetaT - start with relation to lat
            thetaT = lat2theT(i)
            newGrid[idy,  np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,2] = thetaT 
            
            # |----------- Calculate the parametric theta P within the CME -----------|
            # ThetaP - making geometric approx to calc sinThetaP = axis_X tanLon / rperp
            # which ignores that this perp width is not at axis but axis + xCS (which is f(thetaP))
            # (CME is oriented so AWp is in lon direction before we rotate it)
            theselons = np.arange(-nowAWp, nowAWp+1)
            sinTP = lat2xFR(i) * np.tan(theselons*dtor) / CMElens[4]
            # Clean up any places with sinTP > 1 from our geo approx so it just maxes out not blows up
            sinTP[np.where(np.abs(sinTP) > 1)] = np.sign(sinTP[np.where(np.abs(sinTP) > 1)]) * 1
            thetaPs = np.arcsin(sinTP)/dtor
            newGrid[idy,  minlon:maxlon+1,3] = thetaPs
        
        # |----------- Fill in rounded ends of FR region   -----------|
        for i in range(minlat2, maxlat2):
            # Find the range to fill in lon
            idy  =  -i+shifty
            idy2 =  i+shifty
            nowAWp = np.round(lat2AWpEd(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            
            # Fill from minlon to maxlon
            newGrid[idy,  np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
            newGrid[idy2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
            
            # Pad things outside "correct" lon range
            newGrid[idy, np.maximum(0,minlon-2):np.minimum(maxn,maxlon-2)+1,1] = 1
            newGrid[idy2, np.maximum(0,minlon-2):np.minimum(maxn,maxlon-2)+1,1] = 1
           
            # Pad around the top and bottom of the CME
            if i == maxlat2-1: 
               newGrid[idy-1, np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               newGrid[idy-2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,1] = 1
               newGrid[idy2+1, np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               newGrid[idy2+2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,1] = 1
            
            # add angle things
            # ThetaT
            newGrid[idy, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,2] = -90
            newGrid[idy2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,2] = 90
            # ThetaP
            theseLons = np.array(range(minlon,maxlon+1))
            newGrid[idy, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,3] = lon2TP(np.abs(theseLons-90))*np.sign(theseLons-90)
            newGrid[idy2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,3] = lon2TP(np.abs(theseLons-90))*np.sign(theseLons-90)
                         
        # |----------- Clean up angles to stop trig funcs from blowing up  -----------|
        fullTT = newGrid[:,:,2]
        fullTT[np.where(np.abs(fullTT)<1e-5)] = 1e-5
        fullTT[np.where(np.abs(fullTT)>0.9*90)] = np.sign(fullTT[np.where(np.abs(fullTT)>.9*90)]) * 0.9 * 90
        newGrid[:,:,2] = fullTT * dtor
        fullTP = newGrid[:,:,3]
        fullTP[np.where(np.abs(fullTP)<1e-5)] = np.sign(fullTP[np.where(np.abs(fullTP)<1e-5)]) * 1e-5
        fullTP[np.where(np.abs(fullTP)>0.9*90)] = np.sign(fullTP[np.where(np.abs(fullTP)>0.9*90)]) * 0.9 * 90
        newGrid[:,:,3] = fullTP * dtor
        
        
        # |----------- Calculate all the vectors  -----------|
        # normal to the axis
        # turn off error from div by zero outside CME range
        np.seterr(divide='ignore', invalid='ignore')
        nAxX = 0.5 * (np.cos(np.abs(newGrid[:,:,2])) + np.sin(np.abs(newGrid[:,:,2])) / np.sqrt(1 - np.cos(np.abs(newGrid[:,:,2]))))
        nAxZ = thisDelAx * np.sin(newGrid[:,:,2])
        nAxMag = np.sqrt(nAxX**2 + nAxZ**2) # nAxY is zero in this frame
        newGrid[:,:,4] = nAxX / nAxMag *newGrid[:,:,0]
        newGrid[:,:,6] = nAxZ / nAxMag *newGrid[:,:,0]
        mag = np.sqrt(newGrid[:,:,4]**2 + newGrid[:,:,5]**2 + newGrid[:,:,6]**2)    
                
        # normal to the CS
        nCSx0 = np.cos(newGrid[:,:,3]) / thisDelCS
        nCSy = np.sin(newGrid[:,:,3])
        nCSx = nCSx0 * np.cos(newGrid[:,:,2])
        nCSz = nCSx0 * np.sin(newGrid[:,:,2])
        nCSmag = np.sqrt(nCSx**2 + nCSy**2 + nCSz**2)
        newGrid[:,:,7] = nCSx / nCSmag
        newGrid[:,:,8] = nCSy / nCSmag
        newGrid[:,:,9] = nCSz / nCSmag
        mag = np.sqrt(newGrid[:,:,7]**2 + newGrid[:,:,8]**2 + newGrid[:,:,9]**2)    
        
        # tangent to the axis
        # can just swap normal components since axis is in xz plan
        newGrid[:,:,10], newGrid[:,:,12] = -newGrid[:,:,6], newGrid[:,:,4]
        
        # tangent to the CS
        tCSx = nCSy * np.cos(newGrid[:,:,2])
        tCSz = -nCSy * np.sin(newGrid[:,:,2])
        tCSy = np.cos(newGrid[:,:,3]) / thisDelCS
        tCSmag = np.sqrt(tCSx**2 + tCSy**2 + tCSz**2)
        newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15] = -tCSx / tCSmag,  tCSy / tCSmag,  tCSz / tCSmag
        mag = np.sqrt(newGrid[:,:,14]**2 + newGrid[:,:,15]**2 + newGrid[:,:,13]**2)    
                
        # |----------- Calculate the velocities -----------|
        # local vAx
        newGrid[:,:,16] = np.sqrt(thisDelAx**2 * np.cos(newGrid[:,:,2])**2 + np.sin(newGrid[:,:,2])**2) * thesevs[6]
        # local vCS
        newGrid[:,:,17] = np.sqrt(thisDelCS**2 * np.cos(newGrid[:,:,3])**2 + np.sin(newGrid[:,:,3])**2) * thesevs[4]
        # vFront = vBulk * rhatCME + vAx * nAx + vExp * nCS
        newGrid[:,:,18] = thesevs[2] + newGrid[:,:,4] * newGrid[:,:,16] + newGrid[:,:,7] * newGrid[:,:,17]
        newGrid[:,:,19] = newGrid[:,:,5] * newGrid[:,:,16] + newGrid[:,:,8] * newGrid[:,:,17]
        newGrid[:,:,20] = newGrid[:,:,6] * newGrid[:,:,16] + newGrid[:,:,9] * newGrid[:,:,17]
        # vAx = vBulk * rhatCME + vAx * nAx + vExp * nCS
        newGrid[:,:,21] = thesevs[2] + newGrid[:,:,4] * newGrid[:,:,16] 
        newGrid[:,:,22] = newGrid[:,:,5] * newGrid[:,:,16] 
        newGrid[:,:,23] = newGrid[:,:,6] * newGrid[:,:,16] 
       
        # |----------- Coordinate Transformations -----------|
        # Calculate how far to shift the CME in lat/lon to put Earth at 0,0
        dLon = satLoc[1] - thisLon #+ ResArr[key].ANTtimes[-1] * 24 * 3600 * OSP.Sat_rot
        dLat =  - thisLat
        if dLon < -180:
            dLon += 360
        elif dLon > 180:
            dLon -=360
                        
        # |----------- Create background meshgrid and shift it  -----------|
        XX, YY = np.meshgrid(range(-shiftx,shiftx+1),range(-shifty,shifty+1))
        
        # |----------- Calculate local radial vector and dot with v -----------|
        rhatE = np.zeros([ncalc,ncalc,3])
        colat = (90 - YY) * dtor
        rhatE[:,:,0] = np.sin(colat) * np.cos(XX*dtor) 
        rhatE[:,:,1] = np.sin(colat) * np.sin(XX*dtor)
        rhatE[:,:,2] = np.cos(colat)
        newGrid[:,:,24] = rhatE[:,:,0] * newGrid[:,:,18] + rhatE[:,:,1] * newGrid[:,:,19] + rhatE[:,:,2] * newGrid[:,:,20]
        newGrid[:,:,25] = rhatE[:,:,0] * newGrid[:,:,21] + rhatE[:,:,1] * newGrid[:,:,22] + rhatE[:,:,2] * newGrid[:,:,23]
        
        # |----------- Shift the meshgrid -----------|
        XX = XX.astype(float) - dLon
        YY = YY.astype(float) - dLat

        # |----------- Rotate based on CME tilt -----------|
        newGrid = ndimage.rotate(newGrid, 90-thisTilt, reshape=False)
        # Force in/out to be 0/1 again
        newGrid[:,:,0] = np.rint(newGrid[:,:,0])
        newGrid[:,:,1] = np.rint(newGrid[:,:,1])
        
        # |----------- Rotate vectors we already found -----------|
        # Rotate the actual components of the normal and tangent vectors to correct lat/lon/tilt 
        # before using them to define velocity or magnetic vectors
        # normal to axis
        rotNAx =  cart2cart([newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6]], dLat, dLon, thisTilt)
        newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6] = rotNAx[0], rotNAx[1], rotNAx[2]
        # normal to CS
        rotNCS =  cart2cart([newGrid[:,:,7], newGrid[:,:,8], newGrid[:,:,9]], dLat, dLon, thisTilt)
        newGrid[:,:,7], newGrid[:,:,8], newGrid[:,:,9] = rotNCS[0], rotNCS[1], rotNCS[2]
        # tangent to axis
        rotTAx =  cart2cart([newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12]], dLat, dLon, thisTilt)
        newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12] = rotTAx[0], rotTAx[1], rotTAx[2]
        # tangent to CS
        rotTCS =  cart2cart([newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15]], dLat, dLon, thisTilt)
        newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15] = rotTCS[0], rotTCS[1], rotTCS[2]       
        # vFront
        rotvF = cart2cart([newGrid[:,:,18], newGrid[:,:,19], newGrid[:,:,20]], dLat, dLon, thisTilt)
        newGrid[:,:,18], newGrid[:,:,19], newGrid[:,:,20] = rotvF[0], rotvF[1], rotvF[2]
        # vAx
        rotAx = cart2cart([newGrid[:,:,21], newGrid[:,:,22], newGrid[:,:,23]], dLat, dLon, thisTilt)
        newGrid[:,:,21], newGrid[:,:,22], newGrid[:,:,23] = rotAx[0], rotAx[1], rotAx[2]
               
        # |----------- Local unit vecs for points other than 0,0 -----------|
        # radial/z
        rhatE = np.zeros([ncalc,ncalc,3])
        colat = (90 - YY) * dtor
        rhatE[:,:,0] = np.sin(colat) * np.cos(XX*dtor) 
        rhatE[:,:,1] = np.sin(colat) * np.sin(XX*dtor)
        rhatE[:,:,2] = np.cos(colat)

        # z
        zhatE = np.zeros([ncalc,ncalc,3])
        zhatE[:,:,0] = -np.abs(np.cos(colat)) * np.cos(XX*dtor)
        zhatE[:,:,1] = -np.cos(colat) * np.sin(XX*dtor)
        zhatE[:,:,2] = np.sin(colat)
        
        # y
        yhatE = np.zeros([ncalc,ncalc,3])
        yhatE[:,:,0] = -np.sin(XX*dtor)
        yhatE[:,:,1] = np.cos(XX*dtor)
        
        
        # |----------- Calculate the durations -----------|
        br = CMElens[3]
        bCS = 2 * np.cos(newGrid[:,:,3]) * br
        # need to rotate rhatE about axis tangent by actual CS pol ang (not parametric ang)
        realTP = np.arctan(np.tan(newGrid[:,:,3])/thisDelCS)
        # Pull axis vectors into convenient names
        tax, tay, taz    = newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12] 
        n1ax, n1ay, n1az = newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6]
        # Calculate other normal from cross product
        nax, nay, naz = n1ay * taz - n1az * tay, n1az * tax - n1ax * taz, n1ax * tay - n1ay * taz
        rdotn = rhatE[:,:,0] * nax + rhatE[:,:,1] * nay + rhatE[:,:,2] * naz 
        # Get the rhat with the part in the second normal direction removed so we can calc rotation angle
        rhatE2 =  np.zeros([ncalc,ncalc,3])
        rhatE2[:,:,0], rhatE2[:,:,1], rhatE2[:,:,2] = rhatE[:,:,0] - rdotn * nax, rhatE[:,:,1] - rdotn * nay, rhatE[:,:,2] - rdotn * naz
        rhatE2mag = np.sqrt(rhatE2[:,:,0]**2 + rhatE2[:,:,1]**2 + rhatE2[:,:,2]**2)
        rhatE2[:,:,0], rhatE2[:,:,1], rhatE2[:,:,2] = rhatE2[:,:,0] / rhatE2mag, rhatE2[:,:,1] / rhatE2mag, rhatE2[:,:,2] / rhatE2mag
        rdotnAx = rhatE2[:,:,0] * n1ax + rhatE2[:,:,1] * n1ay + rhatE2[:,:,2] * n1az
        # Stop from blowing up too much for oblique angles
        rdotnAx[np.where(rdotnAx < 0.01)] = 0.01
        wid = bCS / rdotnAx * newGrid[:,:,0]
        # Find the time it takes to cross b (half duration with no expansion)
        halfDurEst = 0.5 * wid * 7e5 / newGrid[:,:,24] / 3600.
        # Estimate amount CME grows in that time
        newbr = bCS + newGrid[:,:,17] * halfDurEst *3600/ 7e5 
        newbr = newbr * newGrid[:,:,0]
        growth = newbr / bCS
        growth[np.where(growth<1)] = 1
        # Take average b as original plus growth from half first transit time
        newGrid[:,:,26] = 2 * growth * halfDurEst 
        
        # |----------- Calculate the magnetic field -----------|
        # Toroidal field
        BtCoeff = thisDelCS * thisB0 * thisTau
        newGrid[:,:,27], newGrid[:,:,28], newGrid[:,:,29] = BtCoeff * newGrid[:,:,10], BtCoeff * newGrid[:,:,11], BtCoeff * newGrid[:,:,12]
        
        # Poloidal field
        BpCoeff = thisPol * 2 * thisDelCS / (1 + thisDelCS**2) * np.sqrt(thisDelCS**2 * np.sin(newGrid[:,:,3]*dtor)**2 + np.cos(newGrid[:,:,3]*dtor)**2) * np.abs(thisB0) / thisCnm
        newGrid[:,:,30], newGrid[:,:,31], newGrid[:,:,32] = BpCoeff * newGrid[:,:,13], BpCoeff * newGrid[:,:,14], BpCoeff * newGrid[:,:,15]
        
        # Local z hat components
        # Toroidal
        tempZ = zhatE[:,:,0] * newGrid[:,:,27] + zhatE[:,:,1] * newGrid[:,:,28] + zhatE[:,:,2] * newGrid[:,:,29]
        tempZ[np.isnan(tempZ)] = 0
        newGrid[:,:,33] = tempZ
        # Poloidal
        newGrid[:,:,34] = zhatE[:,:,0] * newGrid[:,:,30] + zhatE[:,:,1] * newGrid[:,:,31] + zhatE[:,:,2] * newGrid[:,:,32]
               
        # |----------- Calculate the Kp index -----------|
        # Start at the front (from Bpol)
        # Need clock ang in local frame -> need By 
        By = yhatE[:,:,0] * newGrid[:,:,30] + yhatE[:,:,1] * newGrid[:,:,31] # yhat has no z comp
        clockAng = np.arctan2(By, newGrid[:,:,34])
        Bperp = np.sqrt(By**2 + newGrid[:,:,34]**2)
        # Calculate Kp with the usual algorithm
        dphidt = np.power(newGrid[:,:,24] , 4/3.) * np.power(Bperp, 2./3.) *np.power(np.abs(np.sin(clockAng/2)), 8/3.)
        newGrid[:,:,35] = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
        # Repeat process for the center
        # Need to scale Bz, Btor decreases proportional to CS area, simplify to growth^2
        By = yhatE[:,:,0] * newGrid[:,:,27] + yhatE[:,:,1] * newGrid[:,:,28] # yhat has no z comp
        # clockAng is measured clockwise from 12 o'clock/North
        clockAng = np.arctan2(By, newGrid[:,:,33])*newGrid[:,:,0]        
        Bperp = np.sqrt(By**2 + newGrid[:,:,33]**2) / growth**2
        dphidt = np.power(newGrid[:,:,25] , 4/3.) * np.power(Bperp, 2./3.) *np.power( np.abs(np.sin(clockAng/2)), 8/3.)
        newGrid[:,:,36] = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
        # Adjust Btor dot z for expansion
        newGrid[:,:,33] = newGrid[:,:,33] / growth**2 * newGrid[:,:,0] 
        
        # |----------- Add temperature and density  -----------|
        newGrid[:,:,37] = thislogT * newGrid[:,:,1] # uniform temp within CME 
        newGrid[:,:,38] =    thisn * newGrid[:,:,1] # uniform dens within CME 
                      
        # |----------- Interpolate and cut out the smaller viewing window -----------|
        # Integer shift
        delxi, delyi = -shiftx-int(XX[0,0]), -shifty-int(YY[0,0])
        # Remainder from integer
        delx, dely = int(XX[0,0]) - XX[0,0], int(YY[0,0]) - YY[0,0]
        # Perform shift in x
        startidx = shiftx - plotwid + delxi
        if startidx > 0:
            leftX = newGrid[:,startidx:startidx+2*plotwid+1,:]
            rightX = newGrid[:,startidx+1:startidx+2*plotwid+2,:]
        else:
            sys.exit('startidx in makeContours went <0, try adjusting calcwid')
            
        subGrid = (1-delx) * leftX + delx * rightX
        # Perform shift in y
        startidy = shifty - plotwid + delyi
        botY = subGrid[startidy:startidy+2*plotwid+1, :]
        topY = subGrid[startidy+1:startidy+2*plotwid+2, :]
        subGrid = (1-dely) * botY + dely * topY
        # Clean up in/out smearing
        subGrid[:,:,0] = np.rint(subGrid[:,:,0])
        subGrid[:,:,1] = np.rint(subGrid[:,:,1])
        for i in range(nThings):
            subGrid[:,:,i] = subGrid[:,:,i] * subGrid[:,:,1]
        allGrid[counter,:,:,:] = subGrid
        counter += 1
    
    # |----------- Set up figure -----------|
    fig, axes = plt.subplots(2, 5, figsize=(11,6))
    cmap1 = plt.get_cmap("plasma",lut=10)
    cmap1.set_bad("k")
    # Reorder Axes
    axes = [axes[0,0], axes[0,1], axes[0,2], axes[0,3], axes[0,4], axes[1,0], axes[1,1], axes[1,2], axes[1,3], axes[1,4]]
    labels = ['Chance of Impact (%)', 'B$_z$ Front (nT)', 'v$_r$ Front (km/s)',  'Kp Front', 'n (cm$^{-1}$)', 'Duration (hr)', 'B$_z$ Center (nT)', 'v$_r$ Center (km/s)',  'Kp Center', 'log(T) (K)']    

    # |----------- Sum up properties in each grid cell -----------|
    nCMEs = np.sum(allGrid[:,:,:,1]*allGrid[:,:,:,0], axis=0)
    ngrid = 2 * plotwid+1
    toPlot = np.zeros([ngrid,ngrid,10])
    toPlot[:,:,0] = nCMEs / (nEns-nFails) * 100
    toPlot[:,:,1] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,34], axis=0) / nCMEs
    toPlot[:,:,2] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,24], axis=0) / nCMEs
    toPlot[:,:,3] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,35], axis=0) / nCMEs
    toPlot[:,:,4] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,38], axis=0) / nCMEs
    toPlot[:,:,5] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,26], axis=0) / nCMEs
    toPlot[:,:,6] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,33], axis=0) / nCMEs
    toPlot[:,:,7] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,25], axis=0) / nCMEs
    toPlot[:,:,8] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,36], axis=0) / nCMEs
    toPlot[:,:,9] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,37], axis=0) / nCMEs
    
    subXX, subYY = np.meshgrid(range(-plotwid,plotwid+1),range(-plotwid,plotwid+1)) 
    caxes = np.empty(10)
    divs = np.empty(10)
    
    # |----------- Plot the things and save -----------|    
    for i in range(len(axes)):
        axes[i].set_facecolor('k')
        axes[i].set_aspect('equal', 'box')
        
        toPlotNow = toPlot[:,:,i]
        cent, rng = np.mean(toPlotNow[nCMEs>0]), 1.5*np.std(toPlotNow[nCMEs>0])
        if i == 0: cent, rng = 50, 50
        toPlotNow[nCMEs==0] = np.inf
        if i == 5:
            c = axes[i].pcolormesh(subXX,subYY,toPlotNow,cmap=cmap1,  vmin=0, vmax=50, shading='auto')
        else:    
            c = axes[i].pcolormesh(subXX,subYY,toPlotNow,cmap=cmap1,  vmin=cent-rng, vmax=cent+rng, shading='auto')
        div = make_axes_locatable(axes[i])
        if i < 5:
            cax = div.append_axes("top", size="5%", pad=0.05)
        else:
            cax = div.append_axes("bottom", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
        if i < 5:
            cax.xaxis.set_ticks_position('top')
            cax.set_title(labels[i], fontsize=12)    
        else:    
            cbar.set_label(labels[i], fontsize=12)                
            
        
        if allSats:
            for j in range(len(satLocs)):
                if i == 0:
                    axes[i].plot(satLocs[j][1] -satLocs[satID][1], satLocs[j][0], 'o', ms=12, mfc=satCols[j], mec='k', label=satNames[j])
                else:
                    axes[i].plot(satLocs[j][1] -satLocs[satID][1], satLocs[j][0], 'o', ms=12, mfc=satCols[j], mec='k')
        else:
            axes[i].plot(0, 0, 'o', ms=15, mfc='#98F5FF')
            
        if i > 4:
            axes[i].xaxis.set_ticks_position('top') 
        else:
            axes[i].tick_params(axis='x', which='major', pad=5)
            axes[i].set_xlabel('Lon ($^{\\circ}$)')
        if i not in [0,5]: 
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('Lat ($^{\\circ}$)')
    if allSats:
        fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satLocs))
    plt.xticks(fontsize=10)    
    plt.subplots_adjust(wspace=0.2, hspace=0.55,left=0.1,right=0.95,top=0.84,bottom=0.12)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Contours'+satName+'.'+pO.figtag)   
    plt.close() 
    


# |------------------------------------------------------------|
# |----------------- ANTEATR perc contours --------------------|    
# |------------------------------------------------------------|
def makePercMap(ResArr, nEns, nFails, calcwid=95, plotwid=40, satID=0, satLocs=None, satNames=[''], allSats=None, satCols=None):
    # Start by filling in the area that corresponds to the CME in a convenient frame
    # then rotate it the frame where Earth is at [0,0] at the time of impact for all CMEs
    # using its exact position in "real" space (which will vary slightly between CMEs).
    # Also derive parameters based on Earth's coord system if there were changes in it's
    # position (i.e. vr for rhat corresponding to each potential shifted lat/lon)
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
        
    if allSats:
        satLoc = satLocs[satID]
    else:
        if not satLocs:
            try:
                satLoc = OSP.satPos
            except:
                satLoc = [0,0]
        else:
            satLoc = satLocs
        
            
    # |----------- Simulation parameters -----------|
    ngrid = 2*plotwid+1
    ncalc = 2*calcwid+1
    nThings = 2
    shiftx, shifty = calcwid, calcwid
    counter = 0
    
    # |----------- Get number of actual impacts -----------|
    hits = []
    for i in range(nEns):
        if (not ResArr[i].FIDOmiss[satID]) & (not ResArr[i].fail) & (ResArr[i].ANTFRidx[satID] is not None):
            hits.append(i)
    allGrid = np.zeros([len(hits), ngrid, ngrid, nThings])
    
    # |----------- Loop through each ensemble member -----------|
    for key in hits:
        newGrid = np.zeros([ncalc,ncalc,nThings])
        
        # |----------- Pull in from ResArr -----------|
        if OSP.doFC:
            thisLat = ResArr[key].FClats[-1]
            thisLon = ResArr[key].FClons[-1] 
            thisTilt  = ResArr[key].FCtilts[-1]
        else:
            thisLat = float(OSP.input_values['CMElat'])
            thisLon = float(OSP.input_values['CMElon'])
            thisTilt = float(OSP.input_values['CMEtilt'])
        
        # |----------- Get values at time of first impact  -----------|
        thisidx   = ResArr[key].ANTFRidx[satID]  
        thisAW    = ResArr[key].ANTAWs[thisidx]
        thisAWp   = ResArr[key].ANTAWps[thisidx]
        thisR     = ResArr[key].ANTrs[thisidx]
        thisDelAx = ResArr[key].ANTdelAxs[thisidx]
        thisDelCS = ResArr[key].ANTdelCSs[thisidx]
        
        # # |----------- Calculate the CME lengths -----------|
        # CMElens = [CMEnose, rEdge, rCent, rr, rp, Lr, Lp]
        CMElens = np.zeros(7)
        CMElens[0] = thisR
        CMElens[4] = np.tan(thisAWp*dtor) / (1 + thisDelCS * np.tan(thisAWp*dtor)) * CMElens[0]
        CMElens[3] = thisDelCS * CMElens[4]
        CMElens[6] = (np.tan(thisAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + thisDelAx * np.tan(thisAW*dtor))  
        CMElens[5] = thisDelAx * CMElens[6]
        CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
        CMElens[1] = CMElens[2] * np.tan(thisAW*dtor)
        
        # |----------- Find the axis location -----------|
        nFR = 31 # axis resolution
        thetas = np.linspace(-math.pi/2, math.pi/2, nFR)    
        sns = np.sign(thetas)
        xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(thetas)
        zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(thetas)) + np.sqrt(1 - np.cos(np.abs(thetas))))   
        top = [xFR[-1], 0.,  zFR[-1]+CMElens[3]]
        topSPH = CART2SPH(top)
        # br is constant but axis x changes -> AWp varies
        varyAWp = np.arctan(CMElens[4] / xFR)/dtor 
        # get the axis coords in spherical
        axSPH = CART2SPH([xFR,0.,zFR])
        # get the round outer edge from the CS at last thetaT
        thetaPs = np.linspace(0, math.pi/2,21)
        xEdge = xFR[-1] 
        yEdge = CMElens[4] * np.sin(thetaPs)
        zEdge = zFR[-1] + (thisDelCS * CMElens[4] * np.cos(thetaPs))
        edgeSPH = CART2SPH([xEdge, yEdge, zEdge])
        
        # |----------- Create the maps from latitude to other variables -----------|
        lat2theT =  CubicSpline(axSPH[1],thetas/dtor,bc_type='natural')
        lat2AWp = CubicSpline(axSPH[1],varyAWp,bc_type='natural')
        lat2xFR = CubicSpline(axSPH[1],xFR,bc_type='natural')
        minlat, maxlat = int(round(np.min(axSPH[1]))), int(round(np.max(axSPH[1])))
        minlon, maxlon = shiftx-int(round(thisAWp)),shiftx+int(round(thisAWp))
        lat2AWpEd = CubicSpline(edgeSPH[1][::-1],edgeSPH[2][::-1],bc_type='natural')    
        lon2TP = CubicSpline(edgeSPH[2],thetaPs/dtor,bc_type='natural') 
        # |----------- Check lat values to make sure stays in range -----------|
        maxn = ncalc-1
        if minlat+shifty < 0: minlat = -shifty
        if maxlat+1+shifty > maxn: maxlat = maxn-1-shifty
        minlat2, maxlat2 = int(round(maxlat)), int(round(topSPH[1]))  
        if maxlat2+1+shifty > maxn: maxlat2 = maxn-1-shifty
        
        # |----------- Loop through the latitude and fill in points that are inside CME -----------|
        for i in range(minlat, maxlat+1):
            # |----------- Calc which lons should be filled -----------|
            idy =  i+shifty
            nowAWp = np.round(lat2AWp(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            # Fill from minlon to maxlon
            newGrid[idy, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
                       
            # |----------- Pad the checked region by a few deg  -----------|
            newGrid[idy,  np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               
        
        # |----------- Fill in rounded ends of FR region   -----------|
        for i in range(minlat2, maxlat2):
            # Find the range to fill in lon
            idy  =  -i+shifty
            idy2 =  i+shifty
            nowAWp = np.round(lat2AWpEd(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            
            # Fill from minlon to maxlon
            newGrid[idy,  np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
            newGrid[idy2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
            
            # Pad things outside "correct" lon range
            newGrid[idy, np.maximum(0,minlon-2):np.minimum(maxn,maxlon-2)+1,1] = 1
            newGrid[idy2, np.maximum(0,minlon-2):np.minimum(maxn,maxlon-2)+1,1] = 1
           
            # Pad around the top and bottom of the CME
            if i == maxlat2-1: 
               newGrid[idy-1, np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               newGrid[idy-2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,1] = 1
               newGrid[idy2+1, np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               newGrid[idy2+2, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,1] = 1
            
       
        # |----------- Coordinate Transformations -----------|
        # Calculate how far to shift the CME in lat/lon to put Earth at 0,0
        dLon = satLoc[1] - thisLon #+ ResArr[key].ANTtimes[-1] * 24 * 3600 * OSP.Sat_rot
        dLat =  - thisLat
        if dLon < -180:
            dLon += 360
        elif dLon > 180:
            dLon -=360
                        
        # |----------- Create background meshgrid and shift it  -----------|
        XX, YY = np.meshgrid(range(-shiftx,shiftx+1),range(-shifty,shifty+1))
        
        
        # |----------- Shift the meshgrid -----------|
        XX = XX.astype(float) - dLon
        YY = YY.astype(float) - dLat

        # |----------- Rotate based on CME tilt -----------|
        newGrid = ndimage.rotate(newGrid, 90-thisTilt, reshape=False)
        # Force in/out to be 0/1 again
        newGrid[:,:,0] = np.rint(newGrid[:,:,0])
        newGrid[:,:,1] = np.rint(newGrid[:,:,1])
                              
        # |----------- Interpolate and cut out the smaller viewing window -----------|
        # Integer shift
        delxi, delyi = -shiftx-int(XX[0,0]), -shifty-int(YY[0,0])
        # Remainder from integer
        delx, dely = int(XX[0,0]) - XX[0,0], int(YY[0,0]) - YY[0,0]
        # Perform shift in x
        startidx = shiftx - plotwid + delxi
        if startidx > 0:
            leftX = newGrid[:,startidx:startidx+2*plotwid+1,:]
            rightX = newGrid[:,startidx+1:startidx+2*plotwid+2,:]
        else:
            sys.exit('startidx in makeContours went <0, try adjusting calcwid')
            
        subGrid = (1-delx) * leftX + delx * rightX
        # Perform shift in y
        startidy = shifty - plotwid + delyi
        botY = subGrid[startidy:startidy+2*plotwid+1, :]
        topY = subGrid[startidy+1:startidy+2*plotwid+2, :]
        subGrid = (1-dely) * botY + dely * topY
        # Clean up in/out smearing
        subGrid[:,:,0] = np.rint(subGrid[:,:,0])
        subGrid[:,:,1] = np.rint(subGrid[:,:,1])
        for i in range(nThings):
            subGrid[:,:,i] = subGrid[:,:,i] * subGrid[:,:,1]
        allGrid[counter,:,:,:] = subGrid
        counter += 1
    
    # |----------- Set up figure -----------|
    fig = plt.figure(figsize=(11,6))
    axes = fig.add_subplot(111)
    fig.set_size_inches(8,9) 
    cmap1 = plt.get_cmap("plasma")
    cmap1.set_bad("k")
    

    # |----------- Sum up properties in each grid cell -----------|
    nCMEs = np.sum(allGrid[:,:,:,1]*allGrid[:,:,:,0], axis=0)
    ngrid = 2 * plotwid+1
    toPlot = np.zeros([ngrid,ngrid,10])
    toPlot[:,:,0] = nCMEs / (nEns-nFails) * 100
    
    subXX, subYY = np.meshgrid(range(-plotwid,plotwid+1),range(-plotwid,plotwid+1)) 
    caxes = np.empty(10)
    divs = np.empty(10)
    
    # |----------- Plot the things -----------|    
    axes.set_facecolor('k')
    axes.set_aspect('equal', 'box')
        
    toPlotNow = toPlot[:,:,0]
    cent, rng = 50, 50
    toPlotNow[nCMEs==0] = np.inf
    c = axes.pcolormesh(subXX,subYY,toPlotNow,cmap=cmap1,  vmin=cent-rng, vmax=cent+rng, shading='auto')
    div = make_axes_locatable(axes)
    cax = div.append_axes("top", size="4%", pad=0.05)
    cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=13)
    cax.xaxis.set_ticks_position('top')
    cax.xaxis.set_major_formatter(PercentFormatter())        
        
    if allSats:
        for j in range(len(satLocs)):
            axes.plot(satLocs[j][1] -satLocs[satID][1], satLocs[j][0], 'o', ms=12, mfc=satCols[j], mec='k', label=satNames[j])
        axes.legend(loc='lower left', fancybox=True, ncol=1, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5)
    else:
        axes.plot(0, 0, 'o', ms=15, mfc='#98F5FF')
    
     # |-----------Make pretty and save -----------| 
    axes.tick_params(axis='x', which='major', pad=5)
    axes.set_xlabel('Longitude [$^{\circ}$]', fontsize=18)
    axes.set_ylabel('Latitude [$^{\circ}$]', fontsize=18)   
    axes.yaxis.tick_right()
    axes.yaxis.set_label_position('right')
    axes.xaxis.set_ticks_position('both')
    axes.yaxis.set_ticks_position('both')
            
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_PercMap'+satName+'.'+pO.figtag, bbox_inches='tight')   
    plt.close()    



# |------------------------------------------------------------|
# |------------------ Enlilesque contours ---------------------|    
# |------------------------------------------------------------|
def enlilesque(ResArr, key=0, doColorbar=True, doSat=True, bonusTime=0, merLon=0, planes='both', vel0=300, vel1=None, satNames=[''], satCols=[None]):
    # |----------- Select the desired ensemble member -----------|
    thisCME = ResArr[key]
    
    # |----------- Pick an educated upper contour limit if not given one -----------|
    if vel1 == None:
        maxvF = np.max(thisCME.ANTvFs)
        vel1 = (int(maxvF/50) + 1) * 50
        print ('Setting upper contour at ', vel1, ' km/s')
    
    # |----------- Check if doing equatorial, meridional, or both -----------|
    if planes in ['Both', 'both', 'BOTH']:
        doEq, doMer = True, True
    elif planes in ['Eq', 'eq', 'EQ']:
        doEq, doMer = True, False
    elif planes in ['Mer', 'mer', 'MER']:
        doEq, doMer = False, True
    else:
        sys.exit('Unrecognized keyword in planes in enlilesque. Pick from [both, eq, mer]')
    
    # |----------- Define solar rotation rate -----------|
    # |------------ (Used in MH background) -------------|
    rotrate = 360. / (24.47 * 24 ) # 24.47 day rot at equator, in deg/hor
    
    # |------------- Pull in MH parameters --------------|
    if OSP.doMH:
        MHarea = OSP.MHarea
        MHdist = OSP.MHdist
        areaRs = MHarea *1e10 / (7e5)**2
        CHrad  = np.sqrt(areaRs/np.pi)
        CHang  = np.arctan(CHrad)
        t0, xs0, vs, a1 = emp.getxmodel(MHarea, MHdist)
        vfuncs = emp.getvmodel(MHarea)
        nfuncs = emp.getnmodel(MHarea)
        Brfuncs = emp.getBrmodel(MHarea)
        Blonfuncs = emp.getBlonmodel(MHarea)
        Tfuncs = emp.getTmodel(MHarea)
        vlonfuncs = emp.getvlonmodel(MHarea)
        HSSfuncs = [nfuncs, vfuncs, Brfuncs, Blonfuncs, Tfuncs, vlonfuncs]
    
    # |----------- Pull in satellite parameters -----------|
    if doSat:
        if 'satPath' in OSP.input_values:
            satPath = OSP.input_values['satPath']
            satPaths = OSP.satPathWrapper(satPath)
        else:
            satPaths = OSP.satPathWrapper(OSP.satPos)
        if satNames[0] == '':
            satNames = satPaths[-1]
        # set generic sat colors if not given a set
        if satCols[0] == None:
            satCols = ['r', '#880E4F', '#C2185B', '#EC407A', '#F48FB1', '#FF6F00', '#FFA000', '#FFC107', '#FFE082']
               
        satPaths = satPaths[:-1]
        nSat = len(satNames)    
    

    # |----------- Set up grid -----------|
    nTheta = (360*2)+1
    nRs = 121 
    angs = np.radians(np.linspace(-180, 180, nTheta))
    dTheta = 360. / (nTheta -1)
    if OSP.doMH:
        CHangs = np.where(np.abs(angs) <= CHang)
    rs = np.linspace(0, 1.2, nRs)
    r, theta = np.meshgrid(rs, angs)
    
    # |----------- Pull in constant CME parameters -----------|
    # |------------ (position only changes in FC) ------------|
    CMElat  = thisCME.FClats[-1]
    CMElon  = thisCME.FClons[-1] 
    CMEtilt = thisCME.FCtilts[-1]
    # Keep it from being perfectly vertical/horizontal to avoid div 0 issues
    if CMEtilt == 0:
        CMEtilt = 0.00001
    elif CMEtilt == 90:
        CMEtilt = 89.99999
    elif CMEtilt == -90:
        CMEtilt = -89.99999
        
    # |----------- Determine if mostly vertical -----------|
    isVert = False
    if np.abs(CMEtilt) > 60.:
        isVert = True
    tANT0 = thisCME.ANTtimes[0]
    
    # |----------- Figure out number of pics to make -----------|
    # |--------------- (Set to save every 5 Rs) ----------------|
    dR = 5 # rs
    r0 = int(thisCME.ANTrs[0]/dR)*dR
    rend = int(thisCME.ANTrs[-1]/dR)*dR
    npics = int((rend-r0)/dR)+1

    # |----------- Loop through the time steps -----------|
    for iii in range(npics+bonusTime):
        if iii < npics:
            idx = np.min(np.where(thisCME.ANTrs >=r0 + iii*dR ))
        else:
            idx = np.min(np.where(thisCME.ANTrs >=r0 + (npics-1)*dR ))
        print ('At number ', iii+1, ' out of ', npics+bonusTime, 'at distance of ', r0 + iii*dR, ' Rs')
        
        # |----------- Set up mini grids -----------|
        valuesMer = np.ones((angs.size, rs.size))
        valuesEq = np.ones((angs.size, rs.size))
        isCMEMer  = np.zeros((angs.size, rs.size))
        isCMEEq  = np.zeros((angs.size, rs.size))
        thistime = thisCME.ANTtimes[idx] - tANT0 # in days
        
        # |----------- Hack to let it continue beyond ANT sim length ------------|
        # Assume that it keeps moving at same speed/size 
        # (Can get here by adding bonusTime) 
        if iii >= npics:
            # get CME radial speed
            thisv = thisCME.ANTvFs[idx]
            deltatime = (iii-npics +1)*dR * 7e10 / (thisv *1e5) / (3600*24)
            thistime += deltatime
            
        # |----------- Get sat positions at this time -----------|
        if doSat:
            satRs, satLats, satLons = [], [], []
            try:
                # |----------- Use function for path -----------|
                for i in range(nSat):
                    satLats.append(satPaths[i][0](thistime*24*3600))
                    satLons.append(satPaths[i][1](thistime*24*3600)-CMElon)
                    satRs.append(satPaths[i][2](thistime*24*3600)/215.)
            except:
                # |----------- Use orbital speed for path -----------|
                for i in range(nSat):
                    satLats.append(satPaths[i][0])
                    satLons.append(satPaths[i][1]-CMElon + satPaths[i][3] * thistime*24*3600/dtor)
                    # !!! NEED TO CONFIRM THAT THESE SHOULD BE DIFF FOR .SATS VERSUS NOT
                    if 'satPath' in OSP.input_values:
                        if (OSP.input_values['satPath'][-5:] == '.sats'):
                            satRs.append(satPaths[i][2]/215.)
                    else:
                        satRs.append(satPaths[i][2]/215./7e10)
    
        # |----------- Fill in the background SW before adding CME on top -----------|
        # |----------- Check if OSP was given a v -----------|
        try:
            vSW = OSP.vSW
        # |----------- Otherwise set at default -----------|
        except:
            vSW = 300
        # |----------- Assign to everyone if not doing MH -----------|    
        if not OSP.doMH:
            if doMer:
                valuesMer = vSW * np.ones(valuesMer.shape)
            if doEq:
                valuesEq = vSW * np.ones(valuesEq.shape)
        # |----------- Or add in a MEOW-HiSS HSS -----------|        
        else:
            for j in range(len(rs)):
                # Meridional Slice
                if doMer:
                    valuesMer[:,j] = vSW
                    t1 = t0 + thistime*24
                    thispoint = emp.getHSSprops(rs[j], t1, t0, xs0, vs, a1, [nfuncs, vfuncs, Brfuncs, Blonfuncs, Tfuncs, vlonfuncs])
                    valuesMer[CHangs,j] = ((thispoint[1]-1) * np.sqrt((CHang -np.abs(angs[CHangs]))/CHang) + 1 )*vSW
                # Equatorial Slice
                if doEq:
                    for i in range(len(angs)):
                        ang = angs[i]
                        dtang = -ang / rotrate / dtor
                        t1 = t0 + dtang + thistime*24
                        thispoint = emp.getHSSprops(rs[j], t1, t0, xs0, vs, a1, [nfuncs, vfuncs, Brfuncs, Blonfuncs, Tfuncs, vlonfuncs])
                        valuesEq[i,j] = thispoint[1] * vSW
                
        # |----------- Add the CME to the contours -----------|
        # |------------ Get evolved CME properties -----------|
        CMEr  = thisCME.ANTrs[idx]
        if iii >= npics:
            CMEr += (iii-npics +1)*dR
        CMEAW = thisCME.ANTAWs[idx]
        CMEAWp = thisCME.ANTAWps[idx]
        deltaAx = thisCME.ANTdelAxs[idx]
        deltaCS = thisCME.ANTdelCSs[idx]
        vExps = [thisCME.ANTvFs[idx], thisCME.ANTvEs[idx], thisCME.ANTvBs[idx], thisCME.ANTvCSrs[idx], thisCME.ANTvCSps[idx], thisCME.ANTvAxrs[idx], thisCME.ANTvAxps[idx]]
        # |----------- Add sheath if it exists  -----------|
        try:
            sheathwid = thisCME.PUPwids[idx]
        except:
            sheathwid = 0
        yaw = thisCME.ANTyaws[idx]

        # |----------- Calc CME size  -----------|
        CMElens = np.zeros(7)
        CMElens[0] = CMEr
        CMElens[4] = np.tan(CMEAWp*dtor) / (1 + deltaCS * np.tan(CMEAWp*dtor)) * CMElens[0]
        CMElens[3] = deltaCS * CMElens[4]
        CMElens[6] = (np.tan(CMEAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + deltaAx * np.tan(CMEAW*dtor))  
        CMElens[5] = deltaAx * CMElens[6]
        CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
        CMElens[1] = CMElens[2] * np.tan(CMEAW*dtor)

        # |----------- Calc the axis in CME frame  -----------|
        nFR = 51 # axis resolution
        thetaFR = np.linspace(-np.pi/2, np.pi/2, nFR)    
        sns = np.sign(thetaFR)    
        xFR = CMElens[2] + deltaAx * CMElens[6] * np.cos(thetaFR)
        zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(thetaFR)) + np.sqrt(1 - np.cos(np.abs(thetaFR)))) 
        # |----------- Rotate out of the CME fram -----------|
        axvec = [xFR,0,zFR]
        axvec = roty([xFR-CMEr,0,zFR], -yaw)
        axvec = [axvec[0]+CMEr, 0, axvec[2]]
        axvec1 = rotx(axvec,-(90-CMEtilt))
        xyzpos = roty(axvec1,-CMElat)
        axLats = np.arctan2(xyzpos[2], np.sqrt(xyzpos[0]**2 + xyzpos[1]**2)) / dtor
        axLons = np.arctan2(xyzpos[1], xyzpos[0]) / dtor
        if axLons[0] > axLons[-1]: axLons = axLons[::-1]
        axRs = np.sqrt(xyzpos[0]**2 + xyzpos[1]**2 + xyzpos[2]**2)
        Lon2R = CubicSpline(axLons,axRs,bc_type='natural')
        Lat2R = CubicSpline(axLats,axRs,bc_type='natural')
 
        # |----------- Check if points are within CME (Meridional) -----------|
        if doMer:
            # |----------- Find min/max lat of CME to narrow region to check  -----------|
            dLat = 50
            minLat, maxLat = int(np.min(axLats))-dLat, int(np.max(axLats))+dLat
            idx1 =  np.where(angs/dtor >= np.floor(np.min(axLats)))[0][0]
            idx2 = np.where(angs/dtor >= np.ceil(np.max(axLats)))[0][0]
            # |----------- Use AWp instead of AW if not vertical -----------|
            if not isVert:
                minLat = int(CMElat - 0.5*CMEAWp)
                maxLat = int(CMElat + 0.5*CMEAWp)
                if maxLat > 180:
                    maxLat -= 360.
                idx1 =  np.where(angs/dtor >= np.floor(minLat))[0][0]
                idx2 = np.where(angs/dtor >= np.ceil(maxLat))[0][0]
            
            # |----------- Indices corresponding to min/maxLat -----------|
            idx_min = np.min(np.where(angs >= minLat*dtor))
            idx_max = np.max(np.where(angs <= maxLat*dtor))+1

            # |----------- Loop through and actually check if in CME  -----------|
            for i in np.arange(idx_min-dLat, idx_max+dLat, 1):
                # |----------- Get rad dist of axis here -----------|
                if not isVert:
                    axR = Lat2R(0)
                else:
                    if i < idx1:
                        axR = Lat2R(angs[idx1]/dtor)
                    elif i > idx2:
                        axR = Lat2R(angs[idx2]/dtor)
                    else:
                        axR = Lat2R(angs[i]/dtor)
                # |----------- Get quick range in r for CME/sheath here -----------|
                minR = (axR - 1.5*CMElens[3])/215
                maxR = (axR + 1.5*CMElens[3]+sheathwid)/215
                # |----------- Downselect from quick check -----------|
                mightBin = np.where((rs >= minR) & (rs <= maxR))[0]
                # |----------- Do the rigorous check -----------|
                for j in mightBin:
                    thispos = np.array([rs[j]*215, angs[i]/dtor, merLon])
                    CMEpos = np.array([CMElat, 0, CMEtilt])
                    vpmag, maxrFR, thisPsi, parat = whereAmI(thispos, CMEpos, CMElens, deltaAx, deltaCS, yaw=yaw)
                    rbar = vpmag/maxrFR
                    # |----------- Update v if is in CME  -----------|
                    if rbar <= 1:
                        # get velocity if in CME
                        vCMEframe, vCSVec = getvCMEframe(rbar, thisPsi, parat, deltaAx, deltaCS, vExps)
                        valuesMer[i,j] = np.sqrt(vCMEframe[0]**2 + vCMEframe[1]**2 + vCMEframe[2]**2)
                        isCMEMer[i,j] = 1.
                    # |----------- Update v if in sheath -----------|
                    elif (vpmag  < maxrFR + sheathwid) & (thispos[0] > axR):#(rs[j]*215. > CMElens[0]):
                        valuesMer[i,j] = vExps[0]
        
        # |----------- Check if points are within CME (Equatorial) -----------|
        if doEq:
            # |----------- Find min/max lon of CME to narrow region to check  -----------|
            dLon = 50
            minLon, maxLon = int(np.min(axLons))-dLon, int(np.max(axLons))+dLon
            idx1 =  np.where(angs/dtor >= np.floor(np.min(axLons)))[0][0]
            idx2 = np.where(angs/dtor >= np.ceil(np.max(axLons)))[0][0]
            # |----------- Use AWp instead of AW if not vertical -----------|
            if isVert:
                minLon = int(- 0.5*CMEAWp)
                maxLon = int(0.5*CMEAWp)
                if maxLon > 180:
                    maxLon -= 360.
                idx1 =  np.where(angs/dtor >= np.floor(minLon))[0][0]
                idx2 = np.where(angs/dtor >= np.ceil(maxLon))[0][0]
                    
            # |----------- Indices corresponding to min/maxLon -----------|
            idx_min = np.min(np.where(angs >= minLon*dtor))
            idx_max = np.max(np.where(angs <= maxLon*dtor))+1
            
            # |----------- Loop through and actually check if in CME  -----------|
            for i in np.arange(idx_min-dLon, idx_max+dLon, 1):
                # |----------- Get rad dist of axis here -----------|
                if isVert:
                    axR = Lon2R(0)
                else:
                    if i < idx1:
                        axR = Lon2R(angs[idx1]/dtor)
                    elif i > idx2:
                        axR = Lon2R(angs[idx2]/dtor)
                    else:
                        axR = Lon2R(angs[i]/dtor)                     
                # |----------- Get quick range in r for CME/sheath here -----------|
                minR = (axR - 1.5*CMElens[3])/215
                maxR = (axR + 1.5*CMElens[3]+sheathwid)/215
                # |----------- Downselect from quick check -----------|
                mightBin = np.where((rs >= minR) & (rs <= maxR))[0]
                 # |----------- Do the rigorous check -----------|
                for j in mightBin:
                    thispos = np.array([rs[j]*215, 0, angs[i%(nTheta-1)]/dtor])
                    CMEpos = np.array([CMElat, 0, CMEtilt])
                    vpmag, maxrFR, thisPsi, parat = whereAmI(thispos, CMEpos, CMElens, deltaAx, deltaCS, yaw=yaw)
                    rbar = vpmag/maxrFR
                    # |----------- Update v if is in CME  -----------|
                    if rbar <= 1:
                        # get velocity if in CME
                        vCMEframe, vCSVec = getvCMEframe(rbar, thisPsi, parat, deltaAx, deltaCS, vExps)
                        valuesEq[i,j] = np.sqrt(vCMEframe[0]**2 + vCMEframe[1]**2 + vCMEframe[2]**2)
                        isCMEEq[i,j] = 1.
                    # |----------- Update v if is in sheath  -----------|
                    elif (vpmag  < maxrFR + sheathwid) & (thispos[0] > axR): #(rs[j]*215. > CMElens[0]):
                        valuesEq[i,j] = vExps[0]

        # |----------- Plot the results  -----------|
        # Subtle differences in set up if doing one or two panels
        if doMer and doEq:
            if doColorbar:        
                fig, ax = plt.subplots(1,2, subplot_kw=dict(projection='polar'), figsize=(11,5))
            else:
                fig, ax = plt.subplots(1,2, subplot_kw=dict(projection='polar'), figsize=(10,5))
        else:
            if doColorbar:        
                fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='polar'), figsize=(5,6))
            else:
                fig, ax = plt.subplots(1,1, subplot_kw=dict(projection='polar'), figsize=(5,5.5))
            ax = [ax, ax]
        
        # |----------- Set contour levels  -----------|
        levels = np.linspace(vel0, vel1, 30)    
        if doMer:
            CS = ax[0].contourf(theta, r, valuesMer, levels=levels, cmap=cm.inferno, extend='both' )
            ax[0].contour(theta, r, isCMEMer, [0.1,1], colors='w')
            ax[0].fill_between(2*angs, np.zeros(len(angs)), np.zeros(len(angs))+0.1, color='yellow', zorder=10)
        if doEq:
            CS = ax[1].contourf(theta, r, valuesEq, levels=levels, cmap=cm.inferno, extend='both')
            ax[1].contour(theta, r, isCMEEq, [0.1,1], colors='w')
            ax[1].fill_between(2*angs, np.zeros(len(angs)), np.zeros(len(angs))+0.1, color='yellow', zorder=10)
            
        # |----------- Add in satellite(s)  -----------|     
        if doSat:
            for i in range(nSat):
                col = satCols[i]
                ax[0].plot(0,0, 'o', color=col, markeredgecolor='w', markersize=10, zorder=1, label=satNames[i])
                if doMer:
                    if np.abs(satLons[i]) < 10:
                        ax[0].plot(satLats[i]*dtor, satRs[i], 'o', color=col, markeredgecolor='w', markersize=10)     
                if doEq:   
                    ax[1].plot(satLons[i]*dtor, satRs[i], 'o', color=col, markeredgecolor='w', markersize=10) 
            fig.legend(loc='lower left', fancybox=True, ncol=1, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5) 
                    
        # |----------- Labels and Tick Mars -----------|
        if doMer:
            ax[0].set_xticklabels([])
            ax[0].set_rticks([0.2, 0.4, 0.6, 0.8, 1, 1.2])
            ax[0].set_yticklabels(['', '0.4', '', '0.8', '', '1.2'])
            ax[0].set_rlabel_position(180)
            rlabels = ax[0].get_ymajorticklabels()
            for label in rlabels:
                label.set_color('white')
            ax[0].set_rlim(0,1.2)
            ax[0].set_title('Meridional plane')
            if not OSP.noDate:
                ax[0].set_title((OSP.dObj + datetime.timedelta(days=thistime)).strftime("%d %b %Y %H:%M:%S"))
            else:
                ax[0].set_title("{:#.2f}".format(thistime)+ ' days')
        if doEq:    
            ax[1].set_xticklabels([])
            ax[1].set_rticks([0.2, 0.4, 0.6, 0.8, 1, 1.2])
            ax[1].set_yticklabels(['', '0.4', '', '0.8', '', '1.2'])
            ax[1].set_rlabel_position(180)
            rlabels = ax[1].get_ymajorticklabels()
            for label in rlabels:
                label.set_color('white')
            ax[1].set_rlim(0,1.2)
            ax[1].set_title('Equitorial plane')
            if not OSP.noDate:
                ax[1].set_title((OSP.dObj + datetime.timedelta(days=thistime)).strftime("%d %b %Y %H:%M:%S"))
            else:
                ax[1].set_title("{:#.2f}".format(thistime)+ ' days')
        
        # |----------- Colorbar stuff -----------|
        if doColorbar:
            if doEq and doMer:
                cbar_ax = fig.add_axes([0.35, 0.12, 0.3, 0.025])
            else:
                cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.025])
            cbar = fig.colorbar(CS, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('v (km/s)')
            dv = 100
            if vel1-vel0 > 400:
                dv=200
            cbar.set_ticks(np.arange(vel0,vel1+1,dv))
        
        # |----------- Make pretty and save  -----------|
        if doEq and doMer:
            plt.subplots_adjust(left=0.03,right=0.97,top=0.93,bottom=0.15, wspace=0.05)        
        else:
            plt.subplots_adjust(left=0.03,right=0.97,top=0.95,bottom=0.13, wspace=0.05)
        
        countstr = str(iii)
        countstr = countstr.zfill(3)
        plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Enlilesque'+countstr+'.'+pO.figtag)
        plt.close()
