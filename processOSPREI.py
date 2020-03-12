import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import datetime
import matplotlib.dates as mdates


# make label text size bigger
plt.rcParams.update({'font.size':16})

# Set up the path variable
mainpath = '/Users/ckay/OSPREI/' #MTMYS

# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
sys.path.append(os.path.abspath(mainpath+'codes/')) #MTMYS
import OSPREI  as OSP
from ForeCAT_functions import rotx, roty, rotz, SPH2CART, CART2SPH
from CME_class import cart2cart


# object that will hold all the results for an individual ensemble member
class EnsRes:
	"Container for all the results of a OSPREI ensemble member"
	def __init__(self, name):
            self.ranFC      = False
            self.ranANT     = False
            self.ranFIDO    = False
            self.myID       = None
            self.name       = name
            
            # ForeCAT things
            self.FCtimes    = None
            self.FCrs       = None
            self.FClats     = None
            self.FClons     = None
            self.FCtilts    = None
            self.FCangs     = None
            self.FCvrs      = None
            self.FCvdefs    = None
            
            # ANTEATR things
            self.ANTtime    = None
            self.ANTvtot    = None
            self.ANTvbulk   = None
            self.ANTvexp    = None
            self.ANTAW      = None
            self.ANTr       = None
            self.ANTlonF    = None # sat lon at end
            self.ANTlonS    = None # sat lon when CME at sat r
            self.ANTnsw     = None
            self.ANTvsw     = None
            
            # FIDO things
            self.FIDOtimes  = None
            self.FIDOBxs    = None
            self.FIDOBys    = None
            self.FIDOBzs    = None
            self.FIDOBs     = None
            self.FIDOvs     = None
            self.FIDOKps    = None
            
            # flags for misses -> no ANT/FIDOdata
            self.miss = True
            
            # Dictionary for ensemble things
            self.EnsVal = {}
 

def txt2obj():
    ResArr = {}
    
    global yr, mon, day, DoY
    yr  = int(OSP.date[0:4])
    mon = int(OSP.date[4:6])
    day = int(OSP.date[6:])
    # Check if we were given a time
    try:
        hrs, mins = int(OSP.time[:2]), int(OSP.time[3:])
    except:
        hrs, mins = 0, 0
    dObj = datetime.datetime(yr, mon, day,hrs,mins)
    dNewYear = datetime.datetime(yr, 1, 1, 0,0)
    DoY = (dObj - dNewYear).days + (dObj - dNewYear).seconds/3600./24.
    

    if OSP.doFC:
        FCfile = OSP.Dir+'/ForeCATresults'+OSP.thisName+'.dat'
        FCdata = np.genfromtxt(FCfile, dtype=float, encoding='utf8')
        ids = FCdata[:,0].astype(int)
        unFCids = np.unique(ids)
        nRuns = np.max(ids)
    
        for i in unFCids:
            thisRes = EnsRes(OSP.thisName)
            thisRes.myID = i
            myidxs = np.where(ids==i)[0]
            thisRes.FCtimes = FCdata[myidxs,1]
            thisRes.FCrs    = FCdata[myidxs,2]
            thisRes.FClats  = FCdata[myidxs,3]
            thisRes.FClons  = FCdata[myidxs,4]
            thisRes.FCtilts = FCdata[myidxs,5]
            thisRes.FCangs  = FCdata[myidxs,6]
            thisRes.FCvrs   = FCdata[myidxs,7]
            thisRes.FCdefs  = FCdata[myidxs,8]
        
            # Put it in the results array        
            ResArr[i] = thisRes
        
    if OSP.doANT:
        ANTfile = OSP.Dir+'/ANTEATRresults'+OSP.thisName+'.dat'
        ANTdata = np.genfromtxt(ANTfile, dtype=float, encoding='utf8')
        # get the unique ANTEATR ideas (only one row per id here)
        # might have some missing if it misses
        ANTids = ANTdata[:,0].astype(int)
        for i in ANTids:
            # Check if we already have set up the objects
            # If not do now
            if not OSP.doFC:  
                thisRes = EnsRes(OSP.thisName)
                thisRes.myID = i
            else:
                thisRes = ResArr[i]
            # Set as an impact not a miss
            thisRes.miss = False
            myidxs = np.where(ANTids==i)[0]            
            thisRes.ANTtime  = ANTdata[myidxs,1]
            thisRes.ANTr     = ANTdata[myidxs,2]
            thisRes.ANTvtot  = ANTdata[myidxs,3]
            thisRes.ANTvbulk = ANTdata[myidxs,4]
            thisRes.ANTvexp  = ANTdata[myidxs,5]
            thisRes.ANTAW    = ANTdata[myidxs,6]
            #thisRes.ANTlonF  = ANTdata[counter,7]
            #thisRes.ANTlonS  = ANTdata[counter,8]
            #thisRes.ANTnsw   = ANTdata[counter,9]
            #thisRes.ANTvsw   = ANTdata[counter,10]
            ResArr[thisRes.myID] = thisRes

    if OSP.doFIDO:
        FIDOfile = OSP.Dir+'/FIDOresults'+OSP.thisName+'.dat'
        FIDOdata = np.genfromtxt(FIDOfile, dtype=float, encoding='utf8')
        ids = FIDOdata[:,0].astype(int)
        unFIDOids = np.unique(ids)
        for i in unFIDOids:
            if OSP.doFC or OSP.doANT:
                thisRes = ResArr[i]
            else:
                thisRes = EnsRes(OSP.thisNam)
            # Set as an impact not a miss (might not have run ANTEATR)
            thisRes.miss = False
            myidxs = np.where(ids==i)[0]
            thisRes.FIDOtimes = FIDOdata[myidxs,1]
            thisRes.FIDOBs    = FIDOdata[myidxs,2]
            thisRes.FIDOBxs   = FIDOdata[myidxs,3]
            thisRes.FIDOBys   = FIDOdata[myidxs,4]
            thisRes.FIDOBzs   = FIDOdata[myidxs,5]
            thisRes.FIDOvs    = FIDOdata[myidxs,6]  
            Bvec = [thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs]
            Kp, BoutGSM   = calcKp(Bvec, DoY, thisRes.FIDOvs) 
            thisRes.FIDOKps   = Kp
    
    # if we ran an ensemble load up the initial parameters for each member        
    if len(ResArr.keys()) > 1:
        ENSfile = OSP.Dir+'/EnsembleParams'+OSP.thisName+'.dat' 
        ENSdata = np.genfromtxt(ENSfile, dtype=None, encoding='utf8')
        global varied
        varied = ENSdata[0][1:] 
        nvar = len(varied) 
        for i in range(len(ENSdata)-1):
            for j in range(nvar):
                row = ENSdata[i+1].astype(float)
                ResArr[int(row[0])].EnsVal[varied[j]] = row[j+1]  
        # sort varied according to a nice order
        myOrder = ['ilat', 'ilon', 'tilt', 'Cdperp', 'rstart', 'shapeA', 'shapeB', 'raccel1', 'raccel2', 'vrmin', 'vrmax', 'AWmin', 'AWmax', 'AWr', 'maxM', 'rmaxM', 'shapeB0', 'Cd', 'SSscale', 'FR_B0', 'CME_vExp', 'CME_v1AU', 'nSW', 'vSW']   
        varied = sorted(varied, key=lambda x: myOrder.index(x))   
        
    return ResArr

def readInData():
    dataIn = np.genfromtxt(OSP.ObsDataFile, dtype=float, encoding='utf8', skip_header=10)
    # Need to check if goes over into new year...
    base = datetime.datetime(int(dataIn[0,0]), 1, 1, 1, 0)
    obsDTs = np.array([base + datetime.timedelta(days=int(dataIn[i,1])-1, seconds=int(dataIn[i,2]*3600)) for i in range(len(dataIn[:,0]))])
            
    dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7], dataIn[:,8], dataIn[:,9]/10.])
    return dataOut

def calcKp(Bout, CMEstart, CMEv):
    fracyear = CMEstart / 365.
    rotang = 23.856 * np.sin(6.289 * fracyear + 0.181) + 8.848
    GSMBx = []
    GSMBy = []
    GSMBz = []
    for i in range(len(Bout[0])):
        vec = [Bout[0][i], Bout[1][i], Bout[2][i]]
        GSMvec = rotx(vec, -rotang)
        GSMBx.append(GSMvec[0])
        GSMBy.append(GSMvec[1])
        GSMBz.append(GSMvec[2])
    # calculate Kp 
    GSMBy = np.array(GSMBy)
    GSMBz = np.array(GSMBz)
    Bt = np.sqrt(GSMBy**2 + GSMBz**2)
    thetac = np.arctan2(np.abs(GSMBy), GSMBz)
    dphidt = np.power(CMEv, 4/3.) * np.power(Bt, 2./3.) * np.power(np.sin(thetac/2),8/3.) 
    # Mays/Savani expression, best behaved for high Kp
    Kp = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
    BoutGSM = np.array([GSMBx, GSMBy, GSMBz]) 
    return Kp, BoutGSM

def makeCPAplot(ResArr):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,10))
    if nEns > 1:
        fakers = np.linspace(1.1,10.05,100, endpoint=True)
        splineVals = np.zeros([nEns, 100, 3])
        means = np.zeros([100, 3])
        stds  = np.zeros([100, 3])
        lims  = np.zeros([100, 2, 3])
    
        i = 0
        # Repackage profiles
        for key in ResArr.keys():
            # Fit a spline to data since may be different lengths since take different times
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClats,bc_type='natural')
            splineVals[i,:, 0] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClons,bc_type='natural')
            splineVals[i,:, 1] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCtilts,bc_type='natural')
            splineVals[i,:, 2] = thefit(fakers)    
            i += 1
        for i in range(3):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color='LightGray') 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color='DarkGray') 
        
    # Plot the seed profile
    axes[0].plot(ResArr[0].FCrs, ResArr[0].FClats, linewidth=4, color='b')
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FClons, linewidth=4, color='b')
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCtilts, linewidth=4, color='b')
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_latfs, all_lonfs, all_tiltfs = [], [], []
        for key in ResArr.keys():
            all_latfs.append(ResArr[key].FClats[-1])
            all_lonfs.append(ResArr[key].FClons[-1])
            all_tiltfs.append(ResArr[key].FCtilts[-1])
        fitLats = norm.fit(all_latfs)
        fitLons = norm.fit(all_lonfs)
        fitTilts = norm.fit(all_tiltfs)
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(fitLats[0])+'$\pm$'+'{:4.1f}'.format(fitLats[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(fitLons[0])+'$\pm$'+'{:4.1f}'.format(fitLons[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(fitTilts[0])+'$\pm$'+'{:4.1f}'.format(fitTilts[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    else:
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClats[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClons[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClons[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
                  
    # Labels
    axes[0].set_ylabel('Latitude ('+degree+')')
    axes[1].set_ylabel('Longitude ('+degree+')')
    axes[2].set_ylabel('Tilt ('+degree+')')
    axes[2].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([1.01,10.15])
    plt.subplots_adjust(hspace=0.1,left=0.1,right=0.95,top=0.95,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_CPA.png')

def makeDragplot(ResArr):
    # This it makes more sense to plot this versus distance instead of time
    # Make more physical sense and numbers from histo more relevant for forecasting
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(9,8))
    
    # this isn't the exact end for all cases but don't really care in this figure
    # since more showing trend with distance and it flattens
    rStart = ResArr[0].ANTr[0]
    rEnd = ResArr[0].ANTr[-1]
    
    # get number of impacts, may be less than nEns
    nImp = 0
    for i in range(nEns):
        if not ResArr[i].miss:
            nImp += 1
    
    # Arrays to hold spline results
    fakers = np.linspace(rStart,rEnd-5,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, 4])
    means = np.zeros([100, 4])
    stds  = np.zeros([100, 4])
    lims  = np.zeros([100, 2, 4])
    
    if nEns > 1:
        i = 0
        for key in ResArr.keys():
            if not ResArr[key].miss:
                thisArr = ResArr[key]
                # fit splines to each profile
                thisR = thisArr.ANTr
                # check to make sure the last point isnt the same as second to last
                # which happens if hits at a printstep time
                # if so add negligible amount to make it happy
                if thisR[-1] == thisR[-2]:
                    thisR[-1] += 0.01
                thefit = CubicSpline(thisR, thisArr.ANTvtot,bc_type='natural')
                splineVals[i,:, 0] = thefit(fakers)
                thefit = CubicSpline(thisR, thisArr.ANTvbulk,bc_type='natural')
                splineVals[i,:, 1] = thefit(fakers)
                thefit = CubicSpline(thisR, thisArr.ANTvexp,bc_type='natural')
                splineVals[i,:, 2] = thefit(fakers)
                thefit = CubicSpline(thisR, thisArr.ANTAW,bc_type='natural')
                splineVals[i,:, 3] = thefit(fakers)
                i+=1
        for i in range(4):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color='LightGray') 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color='DarkGray') 
            
    thisArr = ResArr[0]
    axes[0].plot(thisArr.ANTr, thisArr.ANTvtot, 'b', linewidth=3)
    axes[1].plot(thisArr.ANTr, thisArr.ANTvbulk, 'b', linewidth=3)
    axes[2].plot(thisArr.ANTr, thisArr.ANTvexp, 'b', linewidth=3)
    axes[3].plot(thisArr.ANTr, thisArr.ANTAW, 'b', linewidth=3)
    
    # Add the final position as text
    if nEns > 1:
        all_vt, all_vb, all_ve, all_aw = [], [], [], []
        for key in ResArr.keys():
            if not ResArr[key].miss:
                all_vt.append(ResArr[key].ANTvtot[-1])
                all_vb.append(ResArr[key].ANTvbulk[-1])
                all_ve.append(ResArr[key].ANTvexp[-1])
                all_aw.append(ResArr[key].ANTAW[-1])
        fitvt = norm.fit(all_vt)
        fitvb = norm.fit(all_vb)
        fitve = norm.fit(all_ve)
        fitaw = norm.fit(all_aw)
    if nEns > 1:
        axes[0].text(0.97, 0.85, '{:4.1f}'.format(fitvt[0])+'$\pm$'+'{:4.1f}'.format(fitvt[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.85, '{:4.1f}'.format(fitvb[0])+'$\pm$'+'{:4.1f}'.format(fitvb[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.85, '{:4.1f}'.format(fitve[0])+'$\pm$'+'{:4.1f}'.format(fitve[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '{:4.1f}'.format(fitaw[0])+'$\pm$'+'{:4.1f}'.format(fitaw[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    else:
        axes[0].text(0.97, 0.85, '{:4.1f}'.format(ResArr[0].ANTvtot[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.85, '{:4.1f}'.format(ResArr[0].ANTvbulk[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.85, '{:4.1f}'.format(ResArr[0].ANTvexp[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '{:4.1f}'.format(ResArr[0].ANTAW[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
        
                
    axes[3].set_xlim([rStart,rEnd])   
    for i in range(4):
        yticks = axes[i].yaxis.get_major_ticks()
        if len(yticks) > 5:
            ticks2hide = np.array(range(len(yticks)))[::2]
            for j in ticks2hide:
                yticks[j].label1.set_visible(False)
    
    axes[0].set_ylabel('Front Velocity\n(km/s)')         
    axes[1].set_ylabel('Bulk Velocity\n(km/s)')         
    axes[2].set_ylabel('Exp. Velocity\n(km/s)') 
    axes[3].set_ylabel('Angular Width\n('+'$^\circ$'+')') 
    axes[3].set_xlabel('Radial Distance (R$_S$)')        
            
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.1)
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Drag.png')
    

def makeAThisto(ResArr):
    fig, axes = plt.subplots(1, 2, figsize=(8,5), sharey=True)
    all_times = []
    all_vels  = []
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if not ResArr[key].miss:
            all_times.append(ResArr[key].ANTtime[-1])
            all_vels.append(ResArr[key].ANTvtot[-1])
    
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_times, bins=10, color='#882255')
    n2, bins, patches = axes[1].hist(all_vels, bins=10, color='#882255')
    maxcount = np.max(np.maximum(n1, n2))
    axes[0].set_ylim(0, maxcount*1.2)
    
    # Add the mean and sigma from a normal fit
    fitTime = norm.fit(all_times)
    fitVels = norm.fit(all_vels)
    axes[0].text(0.97, 0.89, '{:4.2f}'.format(fitTime[0])+'$\pm$'+'{:4.2f}'.format(fitTime[1])+' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, str(int(fitVels[0]))+'$\pm$'+str(int(fitVels[1]))+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    base = datetime.datetime(yr, 1, 1, 0, 0)
    date = base + datetime.timedelta(days=(DoY+fitTime[0]))   
    dateLabel = date.strftime('%Y %b %d %H:%M ')
    axes[0].text(0.97, 0.95, dateLabel+'$\pm$'+'{:3.1f}'.format(fitTime[1]*24)+' hrs', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    
    
    
    # Take out half the ticks for readability
    for i in range(2):
        xticks = axes[i].xaxis.get_major_ticks()
        ticks2hide = np.array(range(len(xticks)-1))[::2]+1
        for j in ticks2hide:
            xticks[j].label1.set_visible(False)
    
    # Labels
    axes[0].set_xlabel('Transit Time (days)')
    axes[1].set_xlabel('Velocity at Arrival (km/s)')
    axes[0].set_ylabel('Counts')
    plt.subplots_adjust(hspace=0.1,left=0.1,right=0.95,top=0.95,bottom=0.15)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANT.png')
     
def makeISplot(ResArr):
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(8,10))
    mindate = None
    maxdate = None
    for key in ResArr.keys():
        if not ResArr[key].miss:
            #datesNUM = ResArr[key].FIDOtimes+DoY   
            #dates = datetime.datetime(yr, 1, 1) + datetime.timedelta(datesNUM - 1)
            base = datetime.datetime(yr, 1, 1, 1, 0)
            #dates = ResArr[key].FIDOtimes
            dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes])
            if mindate is None: 
                mindate = np.min(dates)
                maxdate = np.max(dates)
            axes[0].plot(dates, ResArr[key].FIDOBs, linewidth=2, color='DarkGray')
            axes[1].plot(dates, ResArr[key].FIDOBxs, linewidth=2, color='DarkGray')
            axes[2].plot(dates, ResArr[key].FIDOBys, linewidth=2, color='DarkGray')
            axes[3].plot(dates, ResArr[key].FIDOBzs, linewidth=2, color='DarkGray')
            axes[4].plot(dates, ResArr[key].FIDOKps, linewidth=2, color='DarkGray')
            if np.min(dates) < mindate: mindate = np.min(dates)
            if np.max(dates) > maxdate: maxdate = np.max(dates)        
    
    # Plot the ensemble seed
    dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[0].FIDOtimes])
    axes[0].plot(dates, ResArr[0].FIDOBs, linewidth=4, color='b')
    axes[1].plot(dates, ResArr[0].FIDOBxs, linewidth=4, color='b')
    axes[2].plot(dates, ResArr[0].FIDOBys, linewidth=4, color='b')
    axes[3].plot(dates, ResArr[0].FIDOBzs, linewidth=4, color='b')
    axes[4].plot(dates, ResArr[0].FIDOKps, linewidth=4, color='b')
    # Make Kps integers only
    Kpmin, Kpmax = int(np.min(ResArr[0].FIDOKps)), int(np.max(ResArr[0].FIDOKps))+1
    axes[4].set_yticks(range(Kpmin, Kpmax+2))
    
    
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('Kp')
    
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
    
    # take out ticks if too many
    for i in range(5):
        yticks = axes[i].yaxis.get_major_ticks()
        if len(yticks) > 6:
            ticks2hide = np.array(range(len(yticks)-1))[::2]
            for j in ticks2hide:
                yticks[j].label1.set_visible(False)
                
    if ObsData is not None:
        axes[0].plot(ObsData[0,:], ObsData[1,:], linewidth=4, color='m')
        axes[1].plot(ObsData[0,:], ObsData[2,:], linewidth=4, color='m')
        axes[2].plot(ObsData[0,:], ObsData[3,:], linewidth=4, color='m')
        axes[3].plot(ObsData[0,:], ObsData[4,:], linewidth=4, color='m')
        axes[4].plot(ObsData[0,:], ObsData[7,:], linewidth=4, color='m')
        

    
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.15)
     
    #plt.show()
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_IS.png')    
    
def makeFIDOhistos(ResArr):
    fig, axes = plt.subplots(1, 3, figsize=(11,5), sharey=True)
    all_dur = []
    all_Bz  = []
    all_Kp  = []
    
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if not ResArr[key].miss:
            all_dur.append(ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])
            all_Bz.append(np.min(ResArr[key].FIDOBzs))
            all_Kp.append(np.max(ResArr[key].FIDOKps))
            
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color='#882255')
    n2, bins, patches = axes[1].hist(all_Bz, bins=10, color='#882255')
    n3, bins, patches = axes[2].hist(all_Kp, bins=10, color='#882255')
    maxcount = np.max(np.maximum(n1, n2, n3))
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitDur = norm.fit(all_dur)
    fitBz = norm.fit(all_Bz)
    fitKp = norm.fit(all_Kp)
    axes[0].text(0.97, 0.95, '{:4.2f}'.format(fitDur[0])+'$\pm$'+'{:4.2f}'.format(fitDur[1])+' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, '{:4.1f}'.format(fitBz[0])+'$\pm$'+'{:4.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.95, '{:4.1f}'.format(fitKp[0])+'$\pm$'+'{:4.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    
    # Labels
    axes[0].set_xlabel('Duration (days)')
    axes[1].set_xlabel('Minimum B$_z$ (nT)')
    axes[2].set_xlabel('Maximum Kp')
    axes[0].set_ylabel('Counts')    
    
    plt.subplots_adjust(hspace=0.1,left=0.1,right=0.95,top=0.95,bottom=0.15)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_FIDOhist.png')
    
def makeEnsplot(ResArr):
    # At max want to show variation with lat, lon, tilt, AT, v1AU
    # duration, Bz, Kp (8 vals) but depends on what we ran
    deg = '('+'$^\circ$'+')'
    allNames = ['Lat\n'+deg, 'Lon\n'+deg, 'Tilt\n'+deg, 'Transit\nTime\n(days)', 'v$_{1AU}$\n(km/s)', 'Dur\n(days)', 'min Bz\n(nT)', 'max Kp']
    
    nVert = 0
    if OSP.doFC: nVert += 3
    if OSP.doANT: nVert += 2
    if OSP.doFIDO: nVert += 3
    # number of vertical plots depends on num params varied
    nHoriz = len(varied)
    
    # group EnsRes once to avoid doing in each plot
    nRuns = len(ResArr.keys()) # might need to change to throw out misses
    EnsVal = np.zeros([nHoriz, nRuns])
    i = 0
    for key in ResArr.keys():
        j = 0
        for item in varied:
            EnsVal[j,i] = ResArr[key].EnsVal[item]
            j += 1
        i += 1
        
    # group the results
    OSPres = np.zeros([nRuns, nVert])
    counter = 0
    i = 0
    for key in ResArr.keys():
        if OSP.doFC:
            OSPres[i,0] = ResArr[key].FClats[-1]
            OSPres[i,1] = ResArr[key].FClons[-1]
            OSPres[i,2] = ResArr[key].FCtilts[-1]
            counter += 2
        if OSP.doANT:
            # set to seed if missing that point -> lets code run and 
            # won't be visible since already plotting seed
            if ResArr[key].miss:
                OSPres[i,counter+1] = ResArr[0].ANTtime[-1]
                OSPres[i,counter+2] = ResArr[0].ANTvtot[-1]
            else:     
                OSPres[i,counter+1] = ResArr[key].ANTtime[-1]
                OSPres[i,counter+2] = ResArr[key].ANTvtot[-1]
            counter += 2
        if OSP.doFIDO:
            if ResArr[key].miss:
                OSPres[i,counter+1] = (ResArr[0].FIDOtimes[-1]-ResArr[0].FIDOtimes[0])
                OSPres[i,counter+2] = np.min(ResArr[0].FIDOBzs)
                OSPres[i,counter+3] = np.max(ResArr[0].FIDOKps)
            else:
                OSPres[i,counter+1] = (ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])
                OSPres[i,counter+2] = np.min(ResArr[key].FIDOBzs)
                OSPres[i,counter+3] = np.max(ResArr[key].FIDOKps)
        i += 1
        counter = 0    
    
    fig, axes = plt.subplots(nVert, nHoriz, figsize=(nHoriz+1,nVert+1))
    for i in range(nVert-1):
        for j in range(nHoriz):
            axes[i,j].set_xticklabels([])
    for j in range(nHoriz-1):
        for i in range(nVert):
            axes[i,j+1].set_yticklabels([])
    
    for i in range(nHoriz):
        for j in range(nVert):
            axes[j,i].plot(EnsVal[i,:], OSPres[:,j],'o')
            
    # Take out tick marks for legibilililility
    for i in range(nVert):
        yticks = axes[i,0].yaxis.get_major_ticks()
        ticks2hide = np.array(range(len(yticks)-1))[::2]
        for j in ticks2hide:
            yticks[j].label1.set_visible(False)      
        yticks[-1].label1.set_visible(False)  
        yticks[0].label1.set_visible(False)  
    for i in range(nHoriz):
        xticks = axes[-1,i].xaxis.get_major_ticks()
        ticks2hide = np.array(range(len(xticks)-1))[::2]
        for j in ticks2hide:
            xticks[j].label1.set_visible(False)      
        xticks[-1].label1.set_visible(False)  
        xticks[0].label1.set_visible(False)  
        plt.setp(axes[-1,i].xaxis.get_majorticklabels(), rotation=70 )
        
    # Add labels
    for i in range(nHoriz): axes[-1,i].set_xlabel(varied[i])  
    for j in range(nVert):  axes[j,0].set_ylabel(allNames[j])  
        
    plt.subplots_adjust(hspace=0.01, wspace=0.01, bottom=0.15, top=0.95, right=0.99)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ENS.png')
    
def makeKpprob(ResArr):
    # get the time range for full set
    mindate = None
    maxdate = None    
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes is not None:
            dates = ResArr[key].FIDOtimes
            # save the extreme times to know plot range
            if mindate is None: 
                mindate = np.min(dates)
                maxdate = np.max(dates)
            if np.min(dates) < mindate: mindate = np.min(dates)
            if np.max(dates) > maxdate: maxdate = np.max(dates)
    plotlen = (maxdate - mindate)*24
    nx = int(plotlen/3.)+1
    KpArr = np.zeros([9, nx])
    # Calculate the values at cell midpoints
    Kptimes = np.array([mindate + 1.5/24 + i*3./24. for i in range(nx)]) 
    
    # fill in KpArr
    counter = 0
    for key in ResArr.keys():
        if not ResArr[key].miss:
            counter += 1
            tBounds = [ResArr[key].FIDOtimes[0], ResArr[key].FIDOtimes[-1]]
            thefit = CubicSpline(ResArr[key].FIDOtimes,ResArr[key].FIDOKps,bc_type='natural')
            tidx = np.where((Kptimes >= tBounds[0]) & (Kptimes <= tBounds[1]))[0]
            KpRes = thefit(Kptimes[tidx]).astype(int)
            for i in range(len(KpRes)):
                KpArr[KpRes[i],tidx[i]] += 1
    
    ys = np.array(range(10))
    xs = np.array([mindate + i*3./24. for i in range(nx+1)])
    # convert x axis to dates
    label_day_range = [int((mindate+DoY)*2)/2., int((maxdate+DoY)*2+1)/2.]
    nLabs = int(2*(label_day_range[1]-label_day_range[0]))
    labelDays = [label_day_range[0] + 0.5 * i for i in range(nLabs+1)]
    xvals = np.array(labelDays)-DoY
    base = datetime.datetime(yr, 1, 1, 0, 0)
    dates = np.array([base + datetime.timedelta(days=i) for i in labelDays])    
    dateLabels = [i.strftime('%Y %b %d %H:%M ') for i in dates]    
        
    # Set up date format
    maxduration = (dates[-1] - dates[0]).days+(dates[-1] - dates[0]).seconds/3600./24.
    hr0 = 0
    if dates[0].hour > 12: hr0 = 12
    pltday0 = datetime.datetime(dates[0].year, dates[0].month, dates[0].day, hr0, 0)
    pltdays = np.array([pltday0 + datetime.timedelta(hours=((i)*12)) for i in range(int(maxduration)*2+1)])
    dateLabels = [i.strftime('%Y %b %d %H:%M ') for i in pltdays]
    dNewYear = datetime.datetime(yr, 1, 1)
    startDoY = (pltday0 - dNewYear).days + (pltday0 - dNewYear).seconds/3600./24.
    labelDoY = (pltdays[0]-dNewYear).days + (pltdays[0] - dNewYear).seconds/3600./24.
    
    XX, YY = np.meshgrid(xs,ys)
    KpPerc = KpArr/float(counter)*100
    
    cmap1 = cm.get_cmap("plasma",lut=10)
    cmap1.set_bad("w")
    Kpm = np.ma.masked_less(KpPerc,0.01)
    
                 
    fig, axes = plt.subplots(1, 1, figsize=(8,7))
    # draw a grid because mask away a lot of it
    for x in xs: axes.plot([x,x],[ys[0],ys[-1]], c='LightGrey')
    for y in ys: axes.plot([xs[0],xs[-1]],[y,y], c='LightGrey')
    c = axes.pcolor(XX,YY,Kpm, cmap=cmap1, edgecolors='k', vmin=0, vmax=100)
    axes.set_xlim(mindate,maxdate)
    axes.set_ylabel('Kp Index')
    cbar = fig.colorbar(c, ax=axes)
    cbar.set_label('Percentage Chance', rotation=270, labelpad=15)
    plt.xticks(xvals[1:], dateLabels[1:])
    fig.autofmt_xdate()
    plt.subplots_adjust(left=0.15,right=0.95,top=0.95,bottom=0.2)
    
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Kp.png')    
                               
def makeImpContours(ResArr):
    # While I would like this to be on a globe, there is not an easy
    # way that doesn't require non-basic libraries so just look +/- X deg
    # away from disk center for now and plot flat
    dtor = 3.14159/180.
    plotwid = 30
    ngrid = 2*plotwid+1
    impCounter = np.zeros([ngrid, ngrid])
    lats = np.linspace(-plotwid, plotwid,ngrid).astype(int)
    lons = np.linspace(-plotwid, plotwid,ngrid).astype(int)
    
    
    for key in ResArr.keys():
        thisLat = ResArr[key].FClats[-1]
        thisLon = ResArr[key].FClons[-1]-OSP.satPos[1]
        thisTilt = ResArr[key].FCtilts[-1]
        thisAW   = ResArr[key].FCangs[-1]
        thisR    = ResArr[key].FCrs[-1]
        # Calculate shape parameters
        thisA = OSP.shape[0]  # check if these are in ensembles!!!!
        thisB = OSP.shape[1]
        CdivR = np.tan(thisAW*dtor) / (1. + thisB + np.tan(thisAW*dtor) * (thisA + thisB))
        BdivR = thisB *CdivR
        AdivR = thisA * CdivR
        fullWid = (BdivR + CdivR) * thisR
        crossWid = BdivR * thisR
        
        
        # find max and min lat for each CME
        nosePoint = cart2cart([thisR,0.,0.], thisLat, thisLon, thisTilt)
        topPoint  = cart2cart([thisR,0.,fullWid], thisLat, thisLon, thisTilt)
        botPoint  = cart2cart([thisR,0,-fullWid], thisLat, thisLon, thisTilt)
        topPoint  = CART2SPH(topPoint)
        botPoint  = CART2SPH(botPoint)
        minY, maxY = int(botPoint[1]), int(topPoint[1])
        
        # find how many cells up and down we can go within plot
        if minY < lats[0]: minY = lats[0]
        if maxY > lats[-1]: maxY = lats[-1]
        toCheck = np.array(range(minY, maxY+1))
        
        # Find the location of the axis
        # Calculate the center line first
        nFR = 11
        mid = int(nFR/2)
        thetas = np.linspace(-math.pi/2, math.pi/2, nFR)
        rcent = thisR - (thisA+thisB) * CdivR*thisR
        
        xFR = rcent + thisA*CdivR*thisR*np.cos(thetas)
        zFR = CdivR*thisR * np.sin(thetas) 
       
        #zs = np.linspace(-fullWid,fullWid, 20)
        
        points =  CART2SPH(cart2cart([xFR,0.,zFR], thisLat, thisLon, thisTilt))
        
        # Fit a spline to lon as function of lat and r as function of lat
        theLonfit = CubicSpline(points[1],points[2],bc_type='natural')
        theRfit = CubicSpline(points[1],points[0],bc_type='natural')
        # calculate center lons for each lat
        myXs = theLonfit(toCheck)
        myRs = theRfit(toCheck)

        # Fill in the impact points
        # calc width based on tilt
        xWid = crossWid / np.sin(thisTilt*dtor)
        for i in range(len(myXs)):
            thisPoint = SPH2CART([myRs[i],toCheck[i], myXs[i]])
            unitLon = np.array([-np.sin(myXs[i]*dtor), np.cos(myXs[i]*dtor), 0.])
            # technically, the lat changes a bit because R increases doing it this
            # way but this is more accurate than just tan(ang) = wid / R and much
            # quicker than fully accurate model
            edgePoint = thisPoint + xWid*unitLon
            lonWid = int(CART2SPH(edgePoint)[2] - myXs[i])+1
             
            for j in range(-lonWid,lonWid):
                x = int(myXs[i])+j
                y = toCheck[i]
                if (x >= lons[0]) & (x <= lons[-1]):
                    impCounter[y+plotwid,x+plotwid]+=1
                
    cmap1 = cm.get_cmap("plasma",lut=10)
    cmap1.set_bad("w")
    impPerc = impCounter/nEns*100
    impPercM = np.ma.masked_less(impPerc,0.01)
    
    fig, axes = plt.subplots(1, 1, figsize=(8,7))   
    XX, YY = np.meshgrid(lons,lats)
    
    for x in range(-plotwid,plotwid): 
        axes.plot([x,x],[-plotwid,plotwid], c='k', linewidth=0.25)
        axes.plot([-plotwid,plotwid], [x,x], c='k', linewidth=0.25)
    
    
    # normalize by nEns, shouldn't have ForeCAT misses
    c = axes.pcolor(XX,YY,impPercM, cmap=cmap1, edgecolors='k', vmin=0, vmax=100)
    
    
    # Get mean arrival time -> plot Earth shifted
    all_times = []
    for key in ResArr.keys():
        if not ResArr[key].miss:
            all_times.append(ResArr[key].ANTtime[-1])
    # satellite position at time of impact
    #axes.plot(0,OSP.satPos[0],'o', ms=15, mfc='#98F5FF')
    dlon = np.mean(all_times) * OSP.Sat_rot
    axes.plot(dlon,OSP.satPos[0],'o', ms=15, mfc='#98F5FF')
    
    satImp = int(impPerc[int(OSP.satPos[0]+plotwid), int(dlon+plotwid)])
    axes.text(0.97, 0.95, str(satImp) + '% chance of impact' , horizontalalignment='right', verticalalignment='center', transform=axes.transAxes, color='r')
        
    cbar = fig.colorbar(c, ax=axes)
    cbar.set_label('Chance of Impact', rotation=270, labelpad=15)
    
    axes.set_xlim([-plotwid,plotwid])
    axes.set_ylim([-plotwid,plotwid])
    
    axes.set_ylabel('Latitude ('+'$^\circ$'+')')
    axes.set_xlabel('Longitude ('+'$^\circ$'+')')
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Imp.png')    
    
# Get all the parameters from text files and sort out 
# what we actually ran
OSP.setupOSPREI()
ResArr = txt2obj()

global ObsData
ObsData = None
if OSP.ObsDataFile is not None:
    ObsData = readInData()
    

global nEns
nEns = len(ResArr.keys())
    
# Make CPA plot
makeCPAplot(ResArr)  

# Make drag profile
makeDragplot(ResArr)

# Make in situ plot
makeISplot(ResArr)

# Ensemble plots
if nEns > 1:
    # Make arrival time hisogram 
    makeAThisto(ResArr)
    
    # FIDO histos- duration, minBz
    makeFIDOhistos(ResArr)
    
    # Ensemble input-output plot
    makeEnsplot(ResArr)
    
    # Kp probability timeline
    makeKpprob(ResArr)

    # Make location contour plot
    makeImpContours(ResArr)



        