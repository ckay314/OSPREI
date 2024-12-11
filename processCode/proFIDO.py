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

import processOSPREI as pO
import setupPro as sP

# Helper functions for reordering legend (from the interwebs!)
def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    #ax.legend(handles, labels)
    return(handles, labels)

def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

# |------------------------------------------------------------|
# |---------------- In situ spaghetti plot --------------------|    
# |------------------------------------------------------------|
def makeISplot(ResArr, dObj, DoY, SWpadF=12, SWpadB = 15, HiLite=None, plotn=False, tightDates=False, setTrange=False, useObsLim=False, satID=0, silent=True, satNames=[''], hasObs=False, ObsData=[None], BFs=[None], satCols=None):
    # |------------- Get sat and date params from inputs --------------|
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
    if dObj:    
        yr = dObj.year    
        
    # |------------- Set up figure --------------|    
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8,12))
    
    # |------------- Holders for bounds for figure --------------|
    mindate = None
    maxdate = None
    
    # |------------- Loop through ensemble results --------------|
    for key in ResArr.keys():
        lab = None
        multiBF = False
        # |------------- Generic case colors --------------|
        lw, co, zord, lab = 2, 'DarkGray', 2, [None]
        # |--------------- Seed case colors ---------------|
        if key == 0:
            lw, co, zord, lab = 4, 'k', 11, ['Seed']
            idx = [999]
        # |-------------- HiLite case colors --------------|    
        elif HiLite is not None:
            if key == HiLite:
                lw, co, zord, lab = 4, 'DarkMagenta', 11, [None]
        # |---------------- BF case colors ----------------|    
        if key in BFs:
            idx = np.where(BFs ==key)[0]
            if len(idx) > 1:
                BFidxs = idx
                idx = [BFidxs[0]]
                multiBF = True       
            lw, co, zord = 3, satCols[idx][0], 9 
            lab = satNames[idx]      
 
        # |------------- Make sure this member is an impact --------------|        
        if ResArr[key].FIDOtimes[satID] is not None: 
            # |------------- No date (generic time) mode --------------|
            if OSP.noDate:
                dates = ResArr[key].FIDOtimes[satID]
            # |------------- Convert FIDO time to real dats --------------|    
            else:
                base = datetime.datetime(yr, 1, 1, 0, 0)
                if not OSP.doANT:
                    dates = np.array([base + datetime.timedelta(days=(i-1)) for i in ResArr[key].FIDOtimes[satID]])
                else:
                    dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes[satID]])
                    
            # |------------- Plot the flux rope --------------|
            if not ResArr[key].sheathOnly[satID]:
                nowIdx = ResArr[key].FIDO_FRidx[satID]
                if lab[0]:
                    if multiBF:
                        for iii in BFidxs:
                            axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[satID][nowIdx], linewidth=lw, color=co, zorder=zord, label=satNames[iii])
                    else:
                        axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[satID][nowIdx], linewidth=lw, color=co, zorder=zord, label=lab)
                            
                else:
                    axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[satID][nowIdx]/1e6, linewidth=lw, color=co, zorder=zord)
                # Option to plot either n or Kp
                if OSP.isSat or plotn:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
                else:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[satID][nowIdx], linewidth=lw, color=co, zorder=zord)
            
                # |------------- Establish min/max date if not set --------------|    
                if mindate is None: 
                    mindate = dates[nowIdx[0]]
                    maxdate = dates[nowIdx[-1]]

            # |------------- Plot the sheath --------------|
            # Have to make sure to connect the last sheath 
            # point to the front of the FR
            if len(ResArr[key].FIDO_shidx[satID]) != 0:
                nowIdx = ResArr[key].FIDO_shidx[satID]
                if not ResArr[key].sheathOnly[satID]:
                    nowIdx = np.append(nowIdx, ResArr[key].FIDO_FRidx[satID][0])
                axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[satID][nowIdx]/1e6, '--', linewidth=lw, color=co, zorder=zord)
                if OSP.isSat or plotn:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                else:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[satID][nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                
                # |------------- Establish min/max date if not set --------------|    
                if mindate is None: 
                    mindate = dates[nowIdx[0]]
                    maxdate = dates[nowIdx[-1]]
 
                # |------------- Print the arrival times of the sheath  --------------|
                if not silent:
                    print('ATs: ', dates[nowIdx[0]].strftime('%Y-%m-%dT%H:%M'), dates[ResArr[key].FIDO_FRidx[satID][0]].strftime('%Y-%m-%dT%H:%M'), dates[ResArr[key].FIDO_FRidx[satID][-1]].strftime('%Y-%m-%dT%H:%M'))
                    
            # |------------- No sheath, print the ATs of the FR --------------|        
            else:
                if not silent:
                    print('ATs: ', dates[ResArr[key].FIDO_FRidx[satID][0]].strftime('%Y-%m-%dT%H:%M'), dates[ResArr[key].FIDO_FRidx[satID][-1]].strftime('%Y-%m-%dT%H:%M'))
                    
            # |------------- Add padding of ambient SW surrounding the event --------------|
            if len(ResArr[key].FIDO_SWidx[satID]) > 0:
                # |------------- Get current front/back --------------|
                if len(ResArr[key].FIDO_shidx[satID]) != 0:
                    if not ResArr[key].sheathOnly[satID]:
                        frontEnd, backStart = dates[ResArr[key].FIDO_shidx[satID][0]], dates[ResArr[key].FIDO_FRidx[satID][-1]]
                    else:
                        frontEnd, backStart = dates[ResArr[key].FIDO_shidx[satID][0]], dates[ResArr[key].FIDO_shidx[satID][-1]]
                else:
                    frontEnd, backStart = dates[ResArr[key].FIDO_FRidx[satID][0]], dates[ResArr[key].FIDO_FRidx[satID][-1]]
                
                # |------------- Add extra padding around event --------------|    
                if OSP.noDate:
                    frontStart, backEnd = frontEnd-SWpadF, backStart+SWpadB
                else:
                    frontStart, backEnd = frontEnd-datetime.timedelta(hours=SWpadF), backStart+datetime.timedelta(hours=SWpadB)
                
                # |------------- Find corresponding indices --------------|
                frontIdx = np.where((dates>=frontStart) & (dates <=frontEnd))[0]
                backIdx = np.where((dates>=backStart) & (dates <=backEnd))[0]
                
                # |------------- Plot the ambient SW --------------|
                for nowIdx in [frontIdx, backIdx]:
                    axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[satID][nowIdx]/1e6, ':', linewidth=lw, color=co, zorder=zord)
                    if OSP.isSat or plotn:
                        axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                    else:
                        axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[satID][nowIdx], ':', linewidth=lw, color=co, zorder=zord)
            
            # |------------- Update min/max date from this event --------------|   
            if len(ResArr[key].FIDO_SWidx[satID]) > 0:    
                if dates[frontIdx[0]] < mindate: mindate = dates[frontIdx[0]]
                if dates[backIdx[-1]] > maxdate: maxdate = dates[backIdx[-1]]  
            else:
                # will be either sheath or FR as appropriate
                if dates[nowIdx[0]] < mindate: mindate = dates[nowIdx[0]]
                if dates[nowIdx[-1]] > maxdate: maxdate = dates[nowIdx[-1]]
    
    # |------------- Option to scale down range around CME --------------|                
    if tightDates:
        lesstight = 6 
        hasSheath = False
        if isinstance(OSP.obsShstart[satID], float): 
            mindate = base + datetime.timedelta(days=(OSP.obsShstart[satID]-1))
        else:
            mindate = base + datetime.timedelta(days=(OSP.obsFRstart[satID]-1))
        mindate = mindate - datetime.timedelta(hours=lesstight) 
        maxdate = base + datetime.timedelta(days=(OSP.obsFRend[satID]-1)) + datetime.timedelta(hours=lesstight)      
    
    # |------------- Labels --------------|
    axes[0].set_ylabel('B (nT)')
    if OSP.isSat:
        axes[1].set_ylabel('B$_R$ (nT)')
        axes[2].set_ylabel('B$_T$ (nT)')
        axes[3].set_ylabel('B$_N$ (nT)')
    else:
        axes[1].set_ylabel('B$_x$ (nT)')
        axes[2].set_ylabel('B$_y$ (nT)')
        axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('v (km/s)')
    axes[5].set_ylabel('T (MK)')
    if OSP.isSat or plotn:
        axes[6].set_ylabel('n (cm$^{-3}$)')
    else:
        axes[6].set_ylabel('Kp')
    
    # |------------- Set up axes --------------|
    if not OSP.noDate:
        # Set up date format
        maxduration = (maxdate - mindate).days+(maxdate - mindate).seconds/3600./24.
        startplot = mindate -datetime.timedelta(hours=3)
        endplot = maxdate +datetime.timedelta(hours=3)
        hr0 = 0
        if startplot.hour > 12: hr0=12
        pltday0 = datetime.datetime(startplot.year, startplot.month, startplot.day, hr0, 0)
        pltdays = np.array([pltday0 + datetime.timedelta(hours=((i)*12)) for i in range(int(maxduration+1)*2+1)])
        axes[4].set_xticks(pltdays[1:])
        myFmt = mdates.DateFormatter('%Y %b %d %H:%M ')
        axes[4].xaxis.set_major_formatter(myFmt)
        axes[4].set_xlim([startplot, endplot])
        
        if setTrange:
            axes[5].set_ylim([0,1])
    else:
        axes[6].set_xlabel('Time (days)')
    
    # |------------- Add observations  --------------|
    if hasObs:
        obscol = 'b'
        axes[0].plot(ObsData[satID][0,:], ObsData[satID][1,:], linewidth=4, color=obscol, label='Obs')
        axes[1].plot(ObsData[satID][0,:], ObsData[satID][2,:], linewidth=4, color=obscol)
        axes[2].plot(ObsData[satID][0,:], ObsData[satID][3,:], linewidth=4, color=obscol)
        axes[3].plot(ObsData[satID][0,:], ObsData[satID][4,:], linewidth=4, color=obscol)
        axes[4].plot(ObsData[satID][0,:], ObsData[satID][6,:], linewidth=4, color=obscol)
        axes[5].plot(ObsData[satID][0,:], ObsData[satID][7,:]/1e6, linewidth=4, color=obscol)    
        if OSP.isSat or plotn:
            axes[6].plot(ObsData[satID][0,:], ObsData[satID][5,:], linewidth=4, color=obscol)
        elif hasKp:
            axes[6].plot(ObsData[satID][0,:], ObsData[satID][8,:], linewidth=4, color=obscol)
        
        # |------------- Option to set ylim from obs values --------------|    
        if useObsLim:
            B = ObsData[satID][1,:]
            Blim = 1.5*np.max(np.abs(B[np.where(B < 99999)]))
            axes[0].set_ylim([0, Blim])
            Bx = ObsData[satID][2,:]
            Bxlim = 1.5*np.max(np.abs(Bx[np.where(Bx < 99999)]))
            By = ObsData[satID][3,:]
            Bylim = 1.5*np.max(np.abs(By[np.where(By < 99999)]))
            Bz = ObsData[satID][4,:]
            Bzlim = 1.5*np.max(np.abs(Bz[np.where(Bz < 99999)]))
            axes[1].set_ylim([-Bxlim, Bxlim])
            axes[2].set_ylim([-Bylim, Bylim])
            axes[3].set_ylim([-Bzlim, Bzlim])
            vidx = ObsData[satID][6,:] < 99999
            if np.count_nonzero(vidx) != 0:
                vlim = np.max(ObsData[satID][6,vidx])
                axes[4].set_ylim([0, 1.5*vlim])
            tidx = ObsData[satID][7,:] < 99999999
            if np.count_nonzero(tidx) != 0:
                tlim = np.max(ObsData[satID][7,tidx])/1e6
                axes[5].set_ylim([0, 1.25*tlim])
            if OSP.isSat or plotn:
                nidx = ObsData[satID][5,:] < 9999999
                if np.count_nonzero(nidx) != 0:
                    nlim = np.max(ObsData[satID][5,nidx])
                    axes[6].set_ylim([0, 1.5*nlim])
            elif hasKp:
                axes[6].set_ylim([0, 1.5*np.max(ObsData[satID][8,:])])
            
        # |------------- Add in observed sheath/FR start/stop --------------|
        givenDates = [0, 0, 0]
        if isinstance(OSP.obsShstart[satID], float): 
            if OSP.obsShstart[satID] < 366:
                givenDates[0] = base + datetime.timedelta(days=(OSP.obsShstart[satID]-1))
        if isinstance(OSP.obsFRstart[satID], float): 
            if OSP.obsFRstart[satID] < 366:
                givenDates[1] = base + datetime.timedelta(days=(OSP.obsFRstart[satID]-1))
        if isinstance(OSP.obsFRend[satID], float): 
            if OSP.obsFRend[satID] < 366:
                givenDates[2] = base + datetime.timedelta(days=(OSP.obsFRend[satID]-1))
        for ax in axes:
            yl = ax.get_ylim()
            for aDate in givenDates:
                ax.plot([aDate, aDate], yl, 'k--', zorder=0)
            ax.set_ylim(yl)
                           
        # |------------- Take out yticks if too many --------------|
        for i in range(5):
            yticks = axes[i].yaxis.get_major_ticks()
            if len(yticks) > 6:
                ticks2hide = np.array(range(len(yticks)-1))[::2]
                for j in ticks2hide:
                    yticks[j].label1.set_visible(False)
    
    # |------------- Prettify and save --------------|
    if BFs[0]:
        goodOrder = ['Obs', 'Seed']
        for name in satNames:
            goodOrder.append(name)
        handles, labels = reorderLegend(axes[0], goodOrder)
        fig.legend(handles, labels, loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=(len(labels)))
        
        
    if not OSP.noDate: fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.15)
    if (HiLite is not None):
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_IS'+satName+'_EnsMem'+str(HiLite)+'.'+pO.figtag)
    else:
        plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_IS'+satName+'.'+pO.figtag)    
    plt.close() 



# |------------------------------------------------------------|
# |-------------- Histogram with sheath data ------------------|    
# |------------------------------------------------------------|
def makeallIShistos(ResArr, dObj, DoY, satID=0, satNames=[''], BFs=[None], satCols=None):
    satName = satNames[satID]
    yr = dObj.year 
    
    # |------------- Set up figure and holder arrays --------------|     
    fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True)
    axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2], axes[2,0], axes[2,1], axes[2,2]]
    all_AT   = []
    all_dur  = []
    all_durS = []
    all_vS   = []
    all_vF   = []
    all_vE   = []
    all_B    = []
    all_Bz   = []
    all_Kp   = []
    all_Keys = []
        
    # |------------- Collect the results into holders --------------| 
    for key in ResArr.keys(): 
        if ResArr[key].hasSheath[satID]:
            if (not ResArr[key].FIDOmiss[satID]) and (not ResArr[key].fail):
                # might have FR impact with no sheath, this checks if empty
                if len(ResArr[key].FIDO_shidx[satID]) != 0:
                    all_AT.append(ResArr[key].FIDOtimes[satID][ResArr[key].FIDO_shidx[satID][0]])
                else:
                    all_AT.append(ResArr[key].FIDOtimes[satID][ResArr[key].FIDO_FRidx[satID][0]])
                all_durS.append(ResArr[key].SITdur[satID])
                all_dur.append(ResArr[key].FIDO_FRdur[satID])
                all_vS.append(ResArr[key].SITvSheath[satID])
                if ResArr[key].sheathOnly[satID]:
                    all_vF.append(ResArr[key].FIDOvs[satID][ResArr[key].FIDO_shidx[satID][0]])
                    all_vE.append(0)
                else:    
                    all_vF.append(ResArr[key].FIDOvs[satID][ResArr[key].FIDO_FRidx[satID][0]])
                    all_vE.append(ResArr[key].FIDO_FRexp[satID])
                all_Bz.append(np.min(ResArr[key].FIDOBzs[satID]))
                all_B.append(np.max(ResArr[key].FIDOBs[satID]))
                all_Kp.append(np.max(ResArr[key].FIDOKps[satID]))
                all_Keys.append(key)
    all_Keys = np.array(all_Keys)    
                   
    # |------------- Make histogram structures --------------| 
    hc = 'lightgray'
    n1, bins, patches = axes[0].hist(all_AT, bins=10, color=hc, histtype='bar', ec='black')
    n2, bins, patches = axes[1].hist(all_durS, bins=10, color=hc, histtype='bar', ec='black')
    n3, bins, patches = axes[2].hist(all_dur, bins=10, color=hc, histtype='bar', ec='black')
    n4, bins, patches = axes[3].hist(all_vS, bins=10, color=hc, histtype='bar', ec='black')
    n5, bins, patches = axes[4].hist(all_vF, bins=10, color=hc, histtype='bar', ec='black')
    n6, bins, patches = axes[5].hist(all_vE, bins=10, color=hc, histtype='bar', ec='black')
    n7, bins, patches = axes[6].hist(all_B, bins=10, color=hc, histtype='bar', ec='black')
    n8, bins, patches = axes[7].hist(all_Bz, color=hc, histtype='bar', ec='black')
    n9, bins, patches = axes[8].hist(all_Kp, color=hc, histtype='bar', ec='black')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6), np.max(n7), np.max(n8), np.max(n9)])
    
    # |------------- Set figure limit based on counts --------------| 
    axes[0].set_ylim(0, maxcount*1.1)
    
    # |------------- Get normal fit to histos --------------| 
    fitAT = norm.fit(all_AT)
    fitDurS = norm.fit(all_durS)
    fitDur = norm.fit(all_dur)
    fitvS = norm.fit(all_vS)
    fitvF = norm.fit(all_vF)
    fitvE = norm.fit(all_vE)
    fitB = norm.fit(all_B)
    fitBz = norm.fit(all_Bz)
    fitKp = norm.fit(all_Kp)
    
    # |------------- Add text showing final results --------------| 
    if (not OSP.noDate) &  (not math.isnan(fitAT[0])):
        base = datetime.datetime(yr, 1, 1, 0, 0)
        # add in FC time (if desired)
        date = base+datetime.timedelta(days=(fitAT[0]+DoY))
        dateLabel = date.strftime('%b %d %H:%M')
        axes[0].text(0.97, 0.92, dateLabel+'$\\pm$'+'{:.1f}'.format(fitAT[1]*24)+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes) 
    axes[0].text(0.97, 0.82, '{:.2f}'.format(fitAT[0])+'$\\pm$'+'{:.2f}'.format(fitAT[1]) + ' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.92, '{:.1f}'.format(fitDurS[0])+'$\\pm$'+'{:.1f}'.format(fitDurS[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.92, '{:.1f}'.format(fitDur[0])+'$\\pm$'+'{:.1f}'.format(fitDur[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.92, '{:.0f}'.format(fitvS[0])+'$\\pm$'+'{:.0f}'.format(fitvS[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.92, '{:.0f}'.format(fitvF[0])+'$\\pm$'+'{:.0f}'.format(fitvF[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)    
    axes[5].text(0.97, 0.92, '{:.0f}'.format(fitvE[0])+'$\\pm$'+'{:.0f}'.format(fitvE[1])+ ' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    axes[6].text(0.97, 0.92, '{:.1f}'.format(fitB[0])+'$\\pm$'+'{:.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
    axes[7].text(0.97, 0.92, '{:.1f}'.format(fitBz[0])+'$\\pm$'+'{:.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)
    axes[8].text(0.97, 0.92, '{:.1f}'.format(fitKp[0])+'$\\pm$'+'{:.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)    
    
    # |------------- Add labels --------------| 
    axes[0].set_xlabel('Transit Time (days)')
    axes[1].set_xlabel('Sheath Duration (hours)')
    axes[2].set_xlabel('CME Duration (hours)')
    axes[3].set_xlabel('v$_{S}$ (km/s)')
    axes[4].set_xlabel('v$_F$ (km/s)')
    axes[5].set_xlabel('v$_{Exp}$ (km/s)')
    axes[6].set_xlabel('max B (nT)')
    axes[7].set_xlabel('min Bz (nT)')
    axes[8].set_xlabel('max Kp')
    for i in range(9): axes[i].set_ylabel('Counts')    
    for i in range(9): axes[i].set_ylim(0, maxcount*1.2)
    
    #|--------------- Add BF lines ---------------|
    if BFs[0] or (BFs[0] == 0):
        allys = []
        # |------------- Get current ylims --------------| 
        for i in range(9):
            allys.append(axes[i].get_ylim())
        # |------------- Loop through and add BF values --------------| 
        keycount = 0
        for key in BFs:
            if key in all_Keys:
                myCol = satCols[keycount]
                BFidxs = np.where(BFs == key)[0]
                if len(BFidxs) > 1:
                    myCol = satCols[BFidxs[0]]
                nowKey = np.where(all_Keys == key)[0]
                nowKey = nowKey[0]
                myxs = [all_AT[nowKey], all_durS[nowKey], all_dur[nowKey], all_vS[nowKey], all_vF[nowKey], all_vE[nowKey], all_B[nowKey], all_Bz[nowKey], all_Kp[nowKey]]
                for i in range(9):
                    myx = myxs[i]
                    if i == 0:
                        axes[i].plot([myx, myx], [allys[i][0], allys[i][1]/1.3], '--', color=myCol, label=satNames[keycount], lw=3)
                    else:
                        axes[i].plot([myx, myx], [allys[i][0], allys[i][1]/1.15], '--', color=myCol, lw=3)
                keycount += 1
        # |------------- Reset lims and add legend --------------|         
        for i in range(9):
            axes[i].set_ylim(allys[i])  
        fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satCols))
        
    # |------------- Prettify and save --------------| 
    plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_allIShist'+satName+'.'+pO.figtag)
    plt.close() 



# |------------------------------------------------------------|
# |------------- Histogram without sheath data ----------------|    
# |------------------------------------------------------------|
def makeFIDOhistos(ResArr, dObj, DoY, satID=0, satNames=[''], BFs=[None], satCols=None):
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
    yr = dObj.year 
    
    # |------------- Set up figure and holder arrays --------------| 
    fig, axes = plt.subplots(2, 3, figsize=(9,7), sharey=True)
    axes = [axes[1,0], axes[1,2], axes[1,1], axes[0,0], axes[0,1], axes[0,2]]
    all_AT  = []
    all_dur = []
    all_B   = []
    all_Bz  = []
    all_Kp  = []
    all_vF  = []
    all_vE  = []
    all_Keys = []
    
    # |------------- Collect the results into holders --------------| 
    for key in ResArr.keys(): 
        if not ResArr[key].FIDOmiss[satID]:
            if ResArr[key].sheathOnly[satID]:
                all_AT.append(ResArr[key].FIDOtimes[satID][ResArr[key].FIDO_shidx[satID][0]])
                all_vF.append(ResArr[key].FIDOvs[satID][ResArr[key].FIDO_shidx[satID][0]])                
            else:
                all_AT.append(ResArr[key].FIDOtimes[satID][ResArr[key].FIDO_FRidx[satID][0]])
                all_vF.append(ResArr[key].FIDOvs[satID][ResArr[key].FIDO_FRidx[satID][0]])
            all_dur.append(ResArr[key].FIDO_FRdur[satID])
            all_Bz.append(np.min(ResArr[key].FIDOBzs[satID]))
            all_Kp.append(np.max(ResArr[key].FIDOKps[satID]))
            all_B.append(np.max(ResArr[key].FIDOBs[satID]))
            all_vE.append(ResArr[key].FIDO_FRexp[satID] )
            all_Keys.append(key)
    all_Keys = np.array(all_Keys)
            
    # |------------- Make histogram structures --------------| 
    hc = 'lightgray'
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color=hc, histtype='bar', ec='black')
    n2, bins, patches = axes[1].hist(all_Bz, bins=10, color=hc, histtype='bar', ec='black')
    n3, bins, patches = axes[2].hist(all_B, bins=10, color=hc, histtype='bar', ec='black')
    n4, bins, patches = axes[3].hist(all_AT, bins=10, color=hc, histtype='bar', ec='black')
    n5, bins, patches = axes[4].hist(all_vF, bins=10, color=hc, histtype='bar', ec='black')
    n6, bins, patches = axes[5].hist(all_vE, bins=10, color=hc, histtype='bar', ec='black')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6)])
    axes[0].set_ylim(0, maxcount*1.1)
    
    # |------------- Get normal fit to histos --------------| 
    fitDur = norm.fit(all_dur)
    fitBz = norm.fit(all_Bz)
    fitB  = norm.fit(all_B)
    fitAT = norm.fit(all_AT)
    fitvF  = norm.fit(all_vF)
    fitvE  = norm.fit(all_vE)
    axes[0].text(0.97, 0.95, '{:4.1f}'.format(fitDur[0])+'$\\pm$'+'{:4.1f}'.format(fitDur[1])+' hours', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, '{:4.1f}'.format(fitBz[0])+'$\\pm$'+'{:4.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.95, '{:4.1f}'.format(fitB[0])+'$\\pm$'+'{:4.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    if (not OSP.noDate) &  (not math.isnan(fitAT[0])):
        base = datetime.datetime(yr, 1, 1, 0, 0)
        # add in FC time (if desired)
        date = base+datetime.timedelta(days=(fitAT[0]+DoY))
        dateLabel = date.strftime('%b %d %H:%M')
        axes[3].text(0.97, 0.95, dateLabel+'$\\pm$'+'{:.1f}'.format(fitAT[1]*24)+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)    
        axes[3].text(0.97, 0.85, '{:4.1f}'.format(fitAT[0])+'$\\pm$'+'{:4.1f}'.format(fitAT[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    else:
        axes[3].text(0.97, 0.95, '{:4.1f}'.format(fitAT[0])+'$\\pm$'+'{:4.1f}'.format(fitAT[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.95, '{:4.1f}'.format(fitvF[0])+'$\\pm$'+'{:4.1f}'.format(fitvF[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
    axes[5].text(0.97, 0.95, '{:4.1f}'.format(fitvE[0])+'$\\pm$'+'{:4.1f}'.format(fitvE[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    
    # |------------- Add labels --------------| 
    axes[0].set_xlabel('Duration (hours)')
    axes[1].set_xlabel('Minimum B$_z$ (nT)')
    axes[2].set_xlabel('Maximum B (nT)')
    axes[3].set_xlabel('Transit time (hr)')
    axes[4].set_xlabel('v$_F$ (km/s)')
    axes[5].set_xlabel('v$_{Exp}$ (km/s)')
    for i in range(6): axes[i].set_ylabel('Counts')    
    
    #|--------------- Add BF lines ---------------|
    if BFs[0] or (BFs[0] == 0):
        allys = []
        # |------------- Get current ylims --------------| 
        for i in range(6):
            allys.append(axes[i].get_ylim())
        # |------------- Loop through and add BF values --------------| 
        keycount = 0
        for key in BFs:
            myCol = satCols[keycount]
            BFidxs = np.where(BFs == key)[0]
            if len(BFidxs) > 1:
                myCol = satCols[BFidxs[0]]
            nowKey = np.where(all_Keys == key)[0]
            nowKey = nowKey[0]
            myxs = [all_dur[nowKey], all_Bz[nowKey], all_B[nowKey], all_AT[nowKey], all_vF[nowKey], all_vE[nowKey]]
            for i in range(6):
                myx = myxs[i]
                if i == 3:
                    axes[i].plot([myx, myx], [allys[i][0], allys[i][1]/1.3], '--', color=myCol, label=satNames[keycount], lw=3)
                else:
                    axes[i].plot([myx, myx], [allys[i][0], allys[i][1]/1.15], '--', color=myCol, lw=3)
            keycount += 1
        # |------------- Reset lims and add legend --------------|         
        for i in range(6):
            axes[i].set_ylim(allys[i])  
        fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(satCols))
    
    # |------------- Prettify and save --------------| 
    plt.subplots_adjust(wspace=0.15, hspace=0.25,left=0.12,right=0.95,top=0.92,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_FIDOhist'+satName+'.'+pO.figtag)
    plt.close() 



# |------------------------------------------------------------|
# |------------------- Heat map timeline ----------------------|    
# |------------------------------------------------------------|
def makeAllprob(ResArr, dObj, DoY, pad=6, plotn=False, satID=0, silent=True, satNames=[''], hasObs=False, ObsData=[None], BFs=[None], satCols=None):
    satName = satNames[satID]
    if len(satName)>1:
        satName = '_'+satName
    yr = dObj.year 
    
    # |------------- Find time range over full set --------------| 
    mindate = None
    maxdate = None    
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes[satID] is not None:
            dates = ResArr[key].FIDOtimes[satID]
            # take subset corresponding to sheath/FR
            if len(ResArr[key].FIDO_shidx[satID]) != 0:
                minidx = ResArr[key].FIDO_shidx[satID][0]
            else:
                minidx = ResArr[key].FIDO_FRidx[satID][0]
            if ResArr[key].sheathOnly[satID]:
                maxidx = np.max(np.where(ResArr[key].regions[satID] == 0))
            else:
                maxidx = ResArr[key].FIDO_FRidx[satID][-1]
            dates = dates[minidx:maxidx+1]
            # save the extreme times to know plot range
            if mindate is None: 
                mindate = np.min(dates)
                maxdate = np.max(dates)
            if np.min(dates) < mindate: mindate = np.min(dates)
            if np.max(dates) > maxdate: maxdate = np.max(dates)
    # pad a certain number of hrs
    mindate = mindate - pad/24.
    maxdate = maxdate + (pad+6)/24
    plotlen = (maxdate - mindate)*24
    nx = int(plotlen/3.)#+1
    
    # |------------- Set up grid arrays --------------| 
    nBins = 10
    allArr = np.zeros([7, nBins, nx])
    # want first cell to start at first multiple of 3 hrs before first arrival
    hrs3 = int((mindate+DoY - int(mindate+DoY))*8)
    startPlot = int(mindate+DoY) + hrs3/8.
    # Calculate the values at grid edges
    gridtimes = np.array([startPlot + i*3./24. for i in range(nx+1)])
    centtimes = np.array([startPlot + 1.5/24. + i*3./24. for i in range(nx+1)])
    
    # |------------- Find min/max values for each parameter from sim --------------| 
    minmax = np.zeros([7,2])
    minmax[4,0] = 400 # set v min to 350 km/s so will probably include steady state vSW range
    allmins = [[] for i in range(7)]
    allmaxs = [[] for i in range(7)]
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes[satID] is not None:
            thisRes = ResArr[key]
            if OSP.isSat or plotn:
                allParams = [thisRes.FIDOBs[satID], thisRes.FIDOBxs[satID], thisRes.FIDOBys[satID], thisRes.FIDOBzs[satID], thisRes.FIDOvs[satID], thisRes.FIDOtems[satID], thisRes.FIDOns[satID]]
            else:
                allParams = [thisRes.FIDOBs[satID], thisRes.FIDOBxs[satID], thisRes.FIDOBys[satID], thisRes.FIDOBzs[satID], thisRes.FIDOvs[satID], thisRes.FIDOtems[satID], thisRes.FIDOKps[satID]]
            for i in range(7):
                thismin, thismax = np.min(allParams[i]), np.max(allParams[i]) 
                allmins[i].append(thismin)
                allmaxs[i].append(thismax)
                if thismin < minmax[i,0]: minmax[i,0] = thismin
                if thismax > minmax[i,1]: minmax[i,1] = thismax 
    # Check if max set by extreme outliers (more than 95 percentile + std) and reset if so
    for i in range(7):
        if minmax[i][1] > np.percentile(allmaxs[i],95) + np.std(allmaxs[i]):
            minmax[i][1] = np.percentile(allmaxs[i],95)
        if (minmax[i][0] != 0) & (minmax[i][0] < np.percentile(allmins[i],5) - np.std(allmins[i])):
            minmax[i][0] = np.percentile(allmins[i],5)
 
    # |------------- Check obs min/max if needed --------------| 
    if (not OSP.noDate) and (ObsData[satID] is not None): 
        obsIdx = [1, 2, 3, 4, 6, 7, 5]
        if not (OSP.isSat or plotn):
            obsIdx[-1] = 8
        for i in range(7):
            thisParam = ObsData[satID][obsIdx[i],:].astype(float)
            thisParam =  thisParam[~np.isnan(thisParam.astype(float))]
            try:
                thismin, thismax = np.min(thisParam), np.max(thisParam)
                if thismin < minmax[i,0]: minmax[i,0] = thismin
                if thismax > minmax[i,1]: minmax[i,1] = thismax
            except:
                pass

    # |------------- Fill in grid from simulation --------------| 
    counter = 0
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes[satID] is not None:
            counter += 1
            thisTime = ResArr[key].FIDOtimes[satID] + DoY
            thisRes = ResArr[key]
            # |------------- Collect the params --------------| 
            if OSP.isSat or plotn:
                allParams = [thisRes.FIDOBs[satID], thisRes.FIDOBxs[satID], thisRes.FIDOBys[satID], thisRes.FIDOBzs[satID], thisRes.FIDOvs[satID], thisRes.FIDOtems[satID], thisRes.FIDOns[satID]]
            else:
                allParams = [thisRes.FIDOBs[satID], thisRes.FIDOBxs[satID], thisRes.FIDOBys[satID], thisRes.FIDOBzs[satID], thisRes.FIDOvs[satID], thisRes.FIDOtems[satID], thisRes.FIDOKps[satID]]
            
            # |------------- Loop through params --------------| 
            for j in range(7):
                thismin = minmax[j,0]
                thisrange = minmax[j,1] - minmax[j,0]
                thisbin = thisrange / nBins
                thisParam = allParams[j]
                # |------------- Fit spline to simulated profile --------------| 
                try:
                    thefit = CubicSpline(thisTime, thisParam)
                except:
                    # might have duplicate at front
                    thefit = CubicSpline(thisTime[1:], thisParam[1:])
                # |------------- Get the param value at center of grid times --------------|     
                tidx = np.where((gridtimes >= thisTime[0]-3./24.) & (gridtimes <= thisTime[-1]))[0]
                thisCentT = centtimes[tidx]
                thisRes = ((thefit(thisCentT) -  thismin)/thisbin).astype(int)
                # |------------- Fill in grid --------------| 
                for i in range(len(thisRes)-1):
                    thisCell = thisRes[i]
                    if thisCell < 0: thisCell=0
                    if thisCell >= nBins: thisCell = nBins-1
                    allArr[j,thisCell,tidx[i]] += 1
                    
    # |------------- Set up x-axis date formats --------------|                 
    if not OSP.noDate:
        # convert x axis to dates
        labelDays = gridtimes[np.where(2*gridtimes%1 == 0)]
        base = datetime.datetime(yr, 1, 1, 0, 0)
        dates = np.array([base + datetime.timedelta(days=i) for i in labelDays])    
        dateLabels = [i.strftime('%Y %b %d %H:%M ') for i in dates]    
        plotStart = base + datetime.timedelta(days=(gridtimes[0]))
        
    # |------------- Convert counts to a percent --------------|     
    allPerc = allArr/float(counter)*100
    
    # |------------- Set up color map and masking --------------| 
    cmap1 = plt.get_cmap("plasma",lut=10)
    cmap1.set_bad("w")
    allMasked = np.ma.masked_less(allPerc,0.01)
             
    # |------------- Set up figure and add heat map --------------|                  
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8,12))
    uplim = 0.9*np.max(allMasked)
    for i in range(7):
        ys = np.linspace(minmax[i,0], minmax[i,1],nBins+1)
        XX, YY = np.meshgrid(gridtimes,ys)
        # draw a grid because mask away a lot of it
        for x in gridtimes: axes[i].plot([x,x],[ys[0],ys[-1]], c='LightGrey')
        for y in ys: axes[i].plot([gridtimes[0],gridtimes[-1]],[y,y], c='LightGrey')
        subVals = allMasked[i,:,:]
        subIdx = np.where(subVals < 100.)[0]
        myMax = 10*int(np.max(subVals[subIdx])/10 - 1)
        if myMax < 20: myMax = 20
        c = axes[i].pcolormesh(XX,YY,allMasked[i,:,:], cmap=cmap1, edgecolors='k', vmin=0, vmax=myMax)
        
    # |------------- Add observations to the figure --------------| 
    if ObsData[satID] is not None:
        # need to convert obsdate in datetime fmt to frac dates
        obsDates =  np.copy(ObsData[satID][0,:])
        for i in range(len(obsDates)):
             obsDates[i] = (obsDates[i].timestamp()-plotStart.timestamp())/24./3600. +gridtimes[0]
        cs = ['w', 'b']
        lw = [6,3]
        for i in range(2):
            if i == 1:
                axes[0].plot(obsDates, ObsData[satID][1,:], linewidth=lw[i], color=cs[i], zorder=5, label='Obs')
            else:
                axes[0].plot(obsDates, ObsData[satID][1,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[1].plot(obsDates, ObsData[satID][2,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[2].plot(obsDates, ObsData[satID][3,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[3].plot(obsDates, ObsData[satID][4,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[4].plot(obsDates, ObsData[satID][6,:], linewidth=lw[i], color=cs[i], zorder=5)    
            axes[5].plot(obsDates, ObsData[satID][7,:], linewidth=lw[i], color=cs[i], zorder=5)
            if OSP.isSat or plotn:
                axes[6].plot(obsDates, ObsData[satID][5], linewidth=lw[i], color=cs[i], zorder=5)
            else:
                axes[6].plot(obsDates, ObsData[satID][8,:], linewidth=lw[i], color=cs[i], zorder=5)
    
    #|--------------- Add BF lines ---------------|
    didBF = False
    if BFs[0] or (BFs[0] == 0):
        didBF = True
        BFcount = 0
        for key in BFs:
            try:
                if OSP.noDate:
                    dates2 = ResArr[key].FIDOtimes[satID]
                elif ResArr[key].FIDOtimes[satID][0]:
                    dates2 = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes[satID]])
                    for i in range(len(dates2)):
                         dates2[i] = (dates2[i].timestamp()-plotStart.timestamp())/24./3600. +gridtimes[0]
                else:
                    dates2 = [None]
            except:
                dates2 = [None]
            
            
            
            BFidx = np.where(BFs ==key)[0]
            if len(BFidx) > 1:
                mycol = satCols[BFidx[0]]
            else:
                mycol = satCols[BFidx][0]
            
            thiscol = ['w', mycol]
            lw = [3,2]
    
            if dates2[0] != None:
                for i in range(2):
                    if i == 1:
                        axes[0].plot(dates2, ResArr[key].FIDOBs[satID], linewidth=lw[i], color=thiscol[i], zorder=6, label=satNames[BFcount])
                    else:
                        axes[0].plot(dates2, ResArr[key].FIDOBs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    axes[1].plot(dates2, ResArr[key].FIDOBxs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    axes[2].plot(dates2, ResArr[key].FIDOBys[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    axes[3].plot(dates2, ResArr[key].FIDOBzs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    axes[4].plot(dates2, ResArr[key].FIDOvs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    axes[5].plot(dates2, ResArr[key].FIDOtems[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    if OSP.isSat or plotn:
                        axes[6].plot(dates2, ResArr[key].FIDOns[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
                    else:
                        axes[6].plot(dates2, ResArr[key].FIDOKps[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            BFcount += 1
        
    # |------------- Add in ensemble seed --------------| 
    try:
        if OSP.noDate:
            dates2 = ResArr[0].FIDOtimes[satID]
        elif ResArr[0].FIDOtimes[satID][0]:
            dates2 = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[0].FIDOtimes[satID]])
            for i in range(len(dates2)):
                 dates2[i] = (dates2[i].timestamp()-plotStart.timestamp())/24./3600. +gridtimes[0]
        else:
            dates2 = [None]
    except:
        dates2 = [None]
    thiscol = ['w', 'k']
    lw = [6,3]   
    if dates2[0] != None:
        for i in range(2):
            if i == 1:
                axes[0].plot(dates2, ResArr[0].FIDOBs[satID], linewidth=lw[i], color=thiscol[i], zorder=6, label='Seed')
            else:
                axes[0].plot(dates2, ResArr[0].FIDOBs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            axes[1].plot(dates2, ResArr[0].FIDOBxs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            axes[2].plot(dates2, ResArr[0].FIDOBys[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            axes[3].plot(dates2, ResArr[0].FIDOBzs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            axes[4].plot(dates2, ResArr[0].FIDOvs[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            axes[5].plot(dates2, ResArr[0].FIDOtems[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            if OSP.isSat or plotn:
                axes[6].plot(dates2, ResArr[0].FIDOns[satID], linewidth=lw[i], color=thiscol[i], zorder=6)
            else:
                axes[6].plot(dates2, ResArr[0].FIDOKps[satID], linewidth=lw[i], color=thiscol[i], zorder=6)

    # |------------- Add legend if needed --------------| 
    if didBF or (ObsData[satID] is not None):
        goodOrder = ['Obs','Seed']
        for name in satNames:
            goodOrder.append(name)
        handles, labels = reorderLegend(axes[0], goodOrder)
        fig.legend(handles, labels, loc='lower center', fancybox=False, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=len(labels))
        
    # |------------- Set up labels, ranges, make pretty and save --------------|     
    axes[0].set_xlim(gridtimes[0],gridtimes[-1])    
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('v (km/s)')
    axes[5].set_ylabel('T (K)')
    if OSP.isSat or plotn:
        axes[6].set_ylabel('n (cm$^{-3}$)')
    else:
        axes[6].set_ylabel('Kp')        
    
    xlims = list(axes[6].get_xlim())
    xlims[1] = xlims[1]- 3/24.
    axes[6].set_xlim(xlims)
    if not OSP.noDate: 
        plt.xticks(labelDays, dateLabels)
        fig.autofmt_xdate()
    plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.15, hspace=0.1)
    
    ax0pos = axes[0].get_position()
    fig.subplots_adjust(top=0.9)
    cbar_ax = fig.add_axes([ax0pos.x0, 0.94, ax0pos.width, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_title('Percentage Chance')        
    plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_allPerc'+satName+'.'+pO.figtag)   
    plt.close()