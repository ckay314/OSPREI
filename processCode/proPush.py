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

global dtor
dtor = math.pi / 180.


# make label text size bigger
plt.rcParams.update({'font.size':14})
figtag = '.png'

# Set up the path variable
# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
import OSPREI as OSP
mainpath = OSP.mainpath
sys.path.append(os.path.abspath(OSP.codepath)) #MTMYS

from ForeCAT_functions import rotx, roty, rotz, SPH2CART, CART2SPH
from CME_class import cart2cart
from ANT_PUP import lenFun, getvCMEframe, whereAmI
import empHSS as emp

import processOSPREI as proOSP

# SWpadB set manually to make sure long enough for all cases, cutoff using xlim
def makeISplot(allRes, SWpadF=30, SWpadB = 40, bfCase=None, plotn=False, tightDates=False, setTrange=False):
    
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(12,8))
    mindate = None
    maxdate = None
    ResArr = allRes[0]

    lw, co, zord = 5, '#332288', 11
    #cos = ['r', 'yellow', 'orange', '#88CCEE', 'b']
    cos = [['#660000', '#9F000F', '#E41B17', '#E55451'] ,['#151B54', '#0000A5', '#0909FF', '#488AC7']]
    zos = [15, 10, 11, 12, 13]
    counter = -1
    for ResArr in allRes:        
        counter += 1
        co = cos[counter]    
        zord = zos[counter] 
        
        key = 0  
        whichSat = 1
        for whichSat in [0,1,2,3]:  
            if ResArr[key].FIDOtimes[whichSat] is not None: #not ResArr[key].miss:
                if OSP.noDate:
                    dates = ResArr[key].FIDOtimes[whichSat]
                else:
                    base = datetime.datetime(proOSP.yr, 1, 1, 0, 0)
                    if not OSP.doANT:
                        dates = np.array([base + datetime.timedelta(days=(i-1)) for i in ResArr[key].FIDOtimes[whichSat]])
                    else:
                        dates = np.array([base + datetime.timedelta(days=(i+proOSP.DoY)) for i in ResArr[key].FIDOtimes[whichSat]])
            
                # plot the flux rope
                nowIdx = ResArr[key].FIDO_FRidx[whichSat]
                maxB = np.max(allRes[0][0].FIDOBs[whichSat][nowIdx])
                maxn = np.max(allRes[0][0].FIDOns[whichSat][nowIdx])
                maxT = np.max(allRes[0][0].FIDOtems[whichSat][nowIdx])
                
                axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[whichSat][nowIdx]/maxB, linewidth=lw, color=co[whichSat], zorder=zord)
                axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[whichSat][nowIdx]/maxB, linewidth=lw, color=co[whichSat], zorder=zord)
                axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[whichSat][nowIdx]/maxB, linewidth=lw, color=co[whichSat], zorder=zord)
                axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[whichSat][nowIdx]/maxB, linewidth=lw, color=co[whichSat], zorder=zord)
                axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[whichSat][nowIdx], linewidth=lw, color=co[whichSat], zorder=zord)
                axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[whichSat][nowIdx]/maxT, linewidth=lw, color=co[whichSat], zorder=zord)
                if OSP.isSat or plotn:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[whichSat][nowIdx]/maxn, linewidth=lw, color=co[whichSat], zorder=zord)
                else:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[whichSat][nowIdx], linewidth=lw, color=co[whichSat], zorder=zord)
                if mindate is None: 
                    mindate = dates[nowIdx[0]]
                    maxdate = dates[nowIdx[-1]]

                # plot the sheath (and the FR start so connected)
                if len(ResArr[key].FIDO_shidx[whichSat]) != 0:
                    nowIdx = ResArr[key].FIDO_shidx[whichSat]
                    nowIdx = np.append(nowIdx, ResArr[key].FIDO_FRidx[whichSat][0])
                    axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[whichSat][nowIdx]/maxB, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[whichSat][nowIdx]/maxB, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[whichSat][nowIdx]/maxB, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[whichSat][nowIdx]/maxB, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[whichSat][nowIdx], '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[whichSat][nowIdx]/maxT, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    if OSP.isSat or plotn:
                        axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[whichSat][nowIdx]/maxn, '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    else:
                        axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[whichSat][nowIdx], '--', linewidth=lw, color=co[whichSat], zorder=zord)
                    #axes[4].plot(dates[nowIdx], ResArr[key].FIDOKps[nowIdx], '--', linewidth=2, color='DarkGray')
            
                # plot SW outside of sh+FR
                if len(ResArr[key].FIDO_SWidx[whichSat]) > 0:
                    if len(ResArr[key].FIDO_shidx[whichSat]) != 0:
                        frontEnd, backStart = dates[ResArr[key].FIDO_shidx[whichSat][0]], dates[ResArr[key].FIDO_FRidx[whichSat][-1]]
                    else:
                        frontEnd, backStart = dates[ResArr[key].FIDO_FRidx[whichSat][0]], dates[ResArr[key].FIDO_FRidx[whichSat][-1]]
                    
                    SWpadF, SWpadB = 6/24, 12/24
                    if OSP.noDate:
                        frontStart, backEnd = frontEnd-SWpadF, backStart+SWpadB
                    else:
                        frontStart, backEnd = frontEnd-datetime.timedelta(hours=SWpadF), backStart+datetime.timedelta(hours=SWpadB)
                    frontIdx = np.where((dates>=frontStart) & (dates <=frontEnd))[0]
                    backIdx = np.where((dates>=backStart) & (dates <=backEnd))[0]
                    for nowIdx in [frontIdx, backIdx]:
                        axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[whichSat][nowIdx]/maxB,  linewidth=lw/2, color=co[whichSat], zorder=zord)
                        axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[whichSat][nowIdx]/maxB,  linewidth=lw/2, color=co[whichSat], zorder=zord)
                        axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[whichSat][nowIdx]/maxB,  linewidth=lw/2, color=co[whichSat], zorder=zord)
                        axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[whichSat][nowIdx]/maxB, linewidth=lw/2, color=co[whichSat], zorder=zord)
                        axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[whichSat][nowIdx],linewidth=lw/2, color=co[whichSat], zorder=zord)
                        axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[whichSat][nowIdx]/maxT,  linewidth=lw/2, color=co[whichSat], zorder=zord)
                        if OSP.isSat or plotn:
                            axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[whichSat][nowIdx]/maxn,  linewidth=lw/2, color=co[whichSat], zorder=zord)
                        else:
                            axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[whichSat][nowIdx],  linewidth=lw/2, color=co[whichSat], zorder=zord)
                
                if len(ResArr[key].FIDO_SWidx) > 0:    
                    if dates[frontIdx[0]] < mindate: mindate = dates[frontIdx[0]]
                    if dates[backIdx[-1]] > maxdate: maxdate = dates[backIdx[-1]]  
                else:
                    # will be either sheath or FR as appropriate
                    if dates[nowIdx[0]] < mindate: mindate = dates[nowIdx[0]]
                    if dates[nowIdx[-1]] > maxdate: maxdate = dates[nowIdx[-1]]
                    
        if tightDates:
            lesstight = 12 
            hasSheath = False
            if isinstance(OSP.obsShstart, float): 
                mindate = base + datetime.timedelta(days=(OSP.obsShstart-1))
            else:
                mindate = base + datetime.timedelta(days=(OSP.obsFRstart-1))
            mindate = mindate - datetime.timedelta(hours=lesstight) 
            maxdate = base + datetime.timedelta(days=(OSP.obsFRend-1)) + datetime.timedelta(hours=lesstight+4)      
    
        axes[0].set_ylabel('B/B$_{max}$')
        axes[1].set_ylabel('B$_x$/B$_{max}$')
        axes[2].set_ylabel('B$_y$/B$_{max}$')
        axes[3].set_ylabel('B$_z$/B$_{max}$')
        axes[4].set_ylabel('v (km/s)')
        axes[5].set_ylabel('T/T$_{max}$')
        if OSP.isSat or plotn:
            axes[6].set_ylabel('n/n$_{max}$')
        else:
            axes[6].set_ylabel('Kp')
    
        if not OSP.noDate:
            # Set up date format
            maxduration = (maxdate - mindate).days+(maxdate - mindate).seconds/3600./24.
            # THESES ARE HARDCODED FOR PSPS
            #startplot = mindate -datetime.timedelta(hours=3)
            #endplot = maxdate +datetime.timedelta(hours=3)
            startplot = datetime.datetime(2022, 1, 28, 0, 0)
            endplot = datetime.datetime(2022, 2, 1, 0, 0)
            hr0 = 0
            if startplot.hour > 12: hr0=12
            pltday0 = datetime.datetime(startplot.year, startplot.month, startplot.day, hr0, 0)
            pltdays = np.array([pltday0 + datetime.timedelta(hours=((i)*12)) for i in range(int(maxduration+1)*2+1)])
            axes[4].set_xticks(pltdays[1:])
            myFmt = mdates.DateFormatter('%Y %b %d %H:%M ')
            axes[4].xaxis.set_major_formatter(myFmt)
            axes[4].set_xlim([startplot, endplot])
        
            if setTrange:
                axes[5].set_ylim([0,1e6])
        
        obsCol = 'k'#'#882255'
        obslw = 6
        if ObsData is not None:
            axes[0].plot(ObsData[0,:], ObsData[1,:], linewidth=obslw, color=obsCol)
            axes[1].plot(ObsData[0,:], ObsData[2,:], linewidth=obslw, color=obsCol)
            axes[2].plot(ObsData[0,:], ObsData[3,:], linewidth=obslw, color=obsCol)
            axes[3].plot(ObsData[0,:], ObsData[4,:], linewidth=obslw, color=obsCol)
            axes[4].plot(ObsData[0,:], ObsData[6,:], linewidth=obslw, color=obsCol)
            axes[5].plot(ObsData[0,:], ObsData[7,:], linewidth=obslw, color=obsCol)    
            if OSP.isSat or plotn:
                axes[6].plot(ObsData[0,:], ObsData[5,:], linewidth=4, color=obsCol)
            elif hasKp:
                axes[6].plot(ObsData[0,:], ObsData[8,:], linewidth=4, color=obsCol)

            # check if have obs starts/stop
            givenDates = [0, 0, 0]
            if isinstance(OSP.obsShstart, float): 
                if OSP.obsShstart < 366:
                    givenDates[0] = base + datetime.timedelta(days=(OSP.obsShstart-1))
            if isinstance(OSP.obsFRstart, float): 
                if OSP.obsFRstart < 366:
                    givenDates[1] = base + datetime.timedelta(days=(OSP.obsFRstart-1))
            if isinstance(OSP.obsFRend, float): 
                if OSP.obsFRend < 366:
                    givenDates[2] = base + datetime.timedelta(days=(OSP.obsFRend-1))
            for ax in axes:
                yl = ax.get_ylim()
                for aDate in givenDates:
                    ax.plot([aDate, aDate], yl, 'k--', zorder=0)
                ax.set_ylim(yl)
            
                
            # take out ticks if too many
            for i in range(5):
                yticks = axes[i].yaxis.get_major_ticks()
                if len(yticks) > 6:
                    ticks2hide = np.array(range(len(yticks)-1))[::2]
                    for j in ticks2hide:
                        yticks[j].label1.set_visible(False)
    if False:
        dx =0.07
        plt.gcf().text(0.2+dx, 0.955, 'L23', fontsize=14, weight="bold", color=cos[1])
        plt.gcf().text(0.28+dx, 0.955, 'L23+PUP', fontsize=14, weight="bold", color=cos[2])
        plt.gcf().text(0.47+dx, 0.955, 'K23', fontsize=14, weight="bold", color=cos[3])
        plt.gcf().text(0.55+dx, 0.955, 'K23*', fontsize=14, weight="bold", color=cos[4])
        plt.gcf().text(0.635+dx, 0.955, 'K23*+HSS', fontsize=14, weight="bold", color=cos[0])
    if not OSP.noDate: fig.autofmt_xdate()
    axes[6].set_xlabel('Time (days)')
            
    plt.subplots_adjust(hspace=0.15,left=0.15,right=0.95,top=0.99,bottom=0.1)
    plt.savefig(OSP.Dir+'/figISpushNoPush.png')    
    #plt.show()


# Modify this to loop as needed
'''OSP.setupOSPREI()    
ResArr = proOSP.txt2obj(0)
OGname = OSP.thisName
OGdoPUP = OSP.doPUP

OSP.thisName = '20220126VL_olddrag'
OSP.doPUP = False
ResArr2 = proOSP.txt2obj(0)

OSP.thisName = '20220126VL_olddragPUP'
OSP.doPUP = True
ResArr3 = proOSP.txt2obj(0)

OSP.thisName = '20220126redo_olddrag_noMH'
ResArr4 = proOSP.txt2obj(0)

OSP.thisName = '20220126redo_noMH'
ResArr5 = proOSP.txt2obj(0)


allRes = [ResArr, ResArr2, ResArr3, ResArr4, ResArr5]
alldoPUPs = [True,  False, True, True]'''

if False:
    OSP.setupOSPREI()    
    ResArr = proOSP.txt2obj(0)
    OGname = OSP.thisName
    OGdoPUP = OSP.doPUP

    OSP.thisName = '20220126F10'
    OSP.doPUP = True
    ResArr2 = proOSP.txt2obj(0)

    OSP.thisName = '20220126F20'
    OSP.doPUP = True
    ResArr3 = proOSP.txt2obj(0)

    allRes = [ResArr, ResArr2, ResArr3]
    alldoPups = [True, True, True]

# slow Push case
if True:   
    OSP.setupOSPREI()    
    ResArr = proOSP.txt2obj(0)
    OGname = OSP.thisName
    OGdoPUP = OSP.doPUP

    OSP.thisName = 'slownoPush'
    OSP.doPUP = True
    ResArr2 = proOSP.txt2obj(0)
    
    allRes = [ResArr, ResArr2]
    alldoPups = [True, True]


# set back to first
OSP.thisName = OGname
    
# Pull in observational data
global ObsData
ObsData = None
if OSP.ObsDataFile is not None:
    ObsData = proOSP.readInData()
    
    
# Make multi IS plot
makeISplot(allRes)

