import numpy as np
import math
import matplotlib.pyplot as plt
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
from ForeCAT_functions import rotx, roty, rotz


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
            self.ANTvr      = None
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
            
            # Dictionary for ensemble things
            self.EnsVal = {}
 

def txt2obj():
    ResArr = {}
    
    global yr, mon, day, DoY
    yr  = int(OSP.date[0:4])
    mon = int(OSP.date[4:6])
    day = int(OSP.date[6:])
    dObj = datetime.date(yr, mon, day)
    dNewYear = datetime.date(yr, 1, 1)
    DoY = (dObj - dNewYear).days + 1

    if OSP.doFC:
        FCfile = OSP.Dir+'/ForeCATresults'+OSP.thisName+'.dat'
        FCdata = np.genfromtxt(FCfile, dtype=float)
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
        ANTdata = np.genfromtxt(ANTfile, dtype=float)
        # get the unique ANTEATR ideas (only one row per id here)
        # might have some missing if it misses
        unANTids = ANTdata[:,0].astype(int)
        counter = 0
        for i in unANTids:
            # Check if we already have set up the objects
            # If not do now
            if not OSP.doFC:  
                thisRes = EnsRes(OSP.thisNam)
                thisRes.myID = i
            else:
                thisRes = ResArr[i]
            
            thisRes.ANTtime = ANTdata[counter,1]
            thisRes.ANTvr   = ANTdata[counter,2]
            thisRes.ANTr    = ANTdata[counter,3]
            thisRes.ANTlonF = ANTdata[counter,4]
            thisRes.ANTlonS = ANTdata[counter,5]
            thisRes.ANTnsw  = ANTdata[counter,6]
            thisRes.ANTvsw  = ANTdata[counter,7]
            ResArr[thisRes.myID] = thisRes
            counter += 1

    if OSP.doFIDO:
        FIDOfile = OSP.Dir+'/FIDOresults'+OSP.thisName+'.dat'
        FIDOdata = np.genfromtxt(FIDOfile, dtype=float)
        ids = FIDOdata[:,0].astype(int)
        unFIDOids = np.unique(ids)
        for i in unFIDOids:
            if OSP.doFC or OSP.doANT:
                thisRes = ResArr[i]
            else:
                thisRes = EnsRes(OSP.thisNam)
            myidxs = np.where(ids==i)[0]
            thisRes.FIDOtimes = FIDOdata[myidxs,1]
            thisRes.FIDOBxs   = FIDOdata[myidxs,2]
            thisRes.FIDOBys   = FIDOdata[myidxs,3]
            thisRes.FIDOBzs   = FIDOdata[myidxs,4]
            thisRes.FIDOBs    = FIDOdata[myidxs,5]
            thisRes.FIDOvs    = FIDOdata[myidxs,6]  
            Bvec = [thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs]
            Kp, BoutGSM   = makeKp(Bvec, DoY, thisRes.FIDOvs) 
            thisRes.FIDOKps   = Kp
    
    # if we ran an ensemble load up the initial parameters for each member        
    if len(ResArr.keys()) > 1:
        ENSfile = OSP.Dir+'/EnsembleParams'+OSP.thisName+'.dat' 
        ENSdata = np.genfromtxt(ENSfile, dtype=None)
        global varied
        varied = ENSdata[0][1:] 
        nvar = len(varied) 
        for i in range(len(ENSdata)-1):
            for j in range(nvar):
                row = ENSdata[i+1].astype(float)
                ResArr[int(row[0])].EnsVal[varied[j]] = row[j+1]                    
        
    return ResArr

def makeKp(Bout, CMEstart, CMEv):
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
    
    degree = unichr(176)
    
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

def makeAThisto(ResArr):
    fig, axes = plt.subplots(1, 2, figsize=(8,5))
    all_times = []
    all_vels  = []
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if ResArr[key].ANTtime != None:
            all_times.append(ResArr[key].ANTtime)
            all_vels.append(ResArr[key].ANTvr)
    axes[0].hist(all_times, bins=10)
    axes[1].hist(all_vels, bins=10)
    
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_times, bins=10)
    n2, bins, patches = axes[1].hist(all_vels, bins=10)
    maxcount = np.max(np.maximum(n1, n2))
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitTime = norm.fit(all_times)
    fitVels = norm.fit(all_vels)
    axes[0].text(0.97, 0.95, '{:4.2f}'.format(fitTime[0])+'$\pm$'+'{:4.2f}'.format(fitTime[1])+' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, str(int(fitVels[0]))+'$\pm$'+str(int(fitVels[1]))+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    
    
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
        if ResArr[key].FIDOtimes is not None:
            #datesNUM = ResArr[key].FIDOtimes+DoY   
            #dates = datetime.datetime(yr, 1, 1) + datetime.timedelta(datesNUM - 1)
            base = datetime.datetime(yr, 1, 1, 1, 0)
            #dates = ResArr[key].FIDOtimes
            dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes])
            if mindate is None: 
                mindate = np.min(dates)
                maxdate = np.max(dates)
            axes[0].plot(dates, ResArr[key].FIDOBxs, linewidth=2, color='DarkGray')
            axes[1].plot(dates, ResArr[key].FIDOBys, linewidth=2, color='DarkGray')
            axes[2].plot(dates, ResArr[key].FIDOBzs, linewidth=2, color='DarkGray')
            axes[3].plot(dates, ResArr[key].FIDOBs, linewidth=2, color='DarkGray')
            axes[4].plot(dates, ResArr[key].FIDOKps, linewidth=2, color='DarkGray')
            if np.min(dates) < mindate: mindate = np.min(dates)
            if np.max(dates) > maxdate: maxdate = np.max(dates)        
    
    # Plot the ensemble seed
    dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[0].FIDOtimes])
    axes[0].plot(dates, ResArr[0].FIDOBxs, linewidth=4, color='b')
    axes[1].plot(dates, ResArr[0].FIDOBys, linewidth=4, color='b')
    axes[2].plot(dates, ResArr[0].FIDOBzs, linewidth=4, color='b')
    axes[3].plot(dates, ResArr[0].FIDOBs, linewidth=4, color='b')
    axes[4].plot(dates, ResArr[0].FIDOKps, linewidth=4, color='b')
    
    # Make Kps integers only
    Kpmin, Kpmax = int(np.min(ResArr[0].FIDOKps)), int(np.max(ResArr[0].FIDOKps))+1
    axes[4].set_yticks(range(Kpmin, Kpmax+1))
    
    
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('Kp')
    
    # Set up date format
    maxduration = (maxdate - mindate).days+(maxdate - mindate).seconds/3600./24.
    hr0 = 0
    if mindate.hour > 12: hr0 = 12
    pltday0 = datetime.datetime(mindate.year, mindate.month, mindate.day, hr0, 0)
    pltdays = np.array([pltday0 + datetime.timedelta(hours=((i+1)*12)) for i in range(int(maxduration)*2+1)])
    axes[4].set_xticks(pltdays)
    myFmt = mdates.DateFormatter('%Y %b %d %H:%M ')
    axes[4].xaxis.set_major_formatter(myFmt)
    
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
        if ResArr[key].ANTtime != None:
            all_dur.append(ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])
            all_Bz.append(np.min(ResArr[key].FIDOBzs))
            all_Kp.append(np.max(ResArr[key].FIDOKps))
            
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_dur, bins=10)
    n2, bins, patches = axes[1].hist(all_Bz, bins=10)
    n3, bins, patches = axes[2].hist(all_Kp, bins=10)
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
    deg = '('+unichr(176)+')'
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
            if ResArr[key].ANTtime is None:
                OSPres[i,counter+1] = ResArr[0].ANTtime
                OSPres[i,counter+2] = ResArr[0].ANTvr
            else:     
                OSPres[i,counter+1] = ResArr[key].ANTtime
                OSPres[i,counter+2] = ResArr[key].ANTvr
            counter += 2
        if OSP.doFIDO:
            if ResArr[key].FIDOtimes is None:
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
    
    
# Get all the parameters from text files and sort out 
# what we actually ran
OSP.setupOSPREI()
ResArr = txt2obj()

global nEns
nEns = len(ResArr.keys())
    
# Make CPA plot
#makeCPAplot(ResArr)  

# Make arrival time hisogram if more than run one
#if nEns > 1:
#   makeAThisto(ResArr)

# Make in situ plot
#makeISplot(ResArr)

# FIDO histos? duration, minBz
#if nEns > 1:
#    makeFIDOhistos(ResArr)

# Make location contour plot

# Make Kp probability as function of t?

# Ensemble plots
if nEns > 1:
    makeEnsplot(ResArr)
        