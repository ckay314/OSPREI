import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm, pearsonr
import datetime
import matplotlib.dates as mdates

global dtor
dtor = math.pi / 180.


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
from PARADE import lenFun


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
            self.FCAWs      = None
            self.FCAWps     = None
            self.FCdelAxs   = None
            self.FCdelCSs   = None
            self.FCvFs      = None
            self.FCvBs      = None
            self.FCvEs      = None
            self.FCvAxrs    = None
            self.FCvAxps    = None
            self.FCvCSrs    = None
            self.FCvCSps    = None
            
            
            # ANTEATR things
            self.ANTtimes   = None
            self.ANTrs      = None
            self.ANTAWs     = None
            self.ANTAWps    = None
            self.ANTdelAxs  = None
            self.ANTdelCSs  = None
            self.ANTdelCSAxs = None
            self.ANTvFs     = None
            self.ANTvBs     = None
            self.ANTvEs     = None
            self.ANTvAxrs   = None
            self.ANTvAxps   = None
            self.ANTvCSrs   = None
            self.ANTvCSps   = None
            self.ANTB0s     = None
            self.ANTCnms    = None
            
            # FIDO things
            self.FIDOtimes  = None
            self.FIDOBxs    = None
            self.FIDOBys    = None
            self.FIDOBzs    = None
            self.FIDOBs     = None
            self.FIDOvs     = None
            self.FIDOKps    = None
            self.FIDOnormrs = None
            self.isSheath   = None
            self.hasSheath  = False 
            
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
            thisRes.FCAWs   = FCdata[myidxs,6]
            thisRes.FCAWps  = FCdata[myidxs,7]
            thisRes.FCvFs   = FCdata[myidxs,8]
            thisRes.FCvBs   = FCdata[myidxs,9]
            thisRes.FCvEs   = FCdata[myidxs,10]
            thisRes.FCvAxrs = FCdata[myidxs,11]
            thisRes.FCvAxps = FCdata[myidxs,12]
            thisRes.FCvCSrs = FCdata[myidxs,13]
            thisRes.FCvCSps = FCdata[myidxs,14]
            thisRes.FCdefs  = FCdata[myidxs,15]
            thisRes.FCdelAxs = FCdata[myidxs,16]
            thisRes.FCdelCSs = FCdata[myidxs,17]
        
            # Put it in the results array        
            ResArr[i] = thisRes
        
    if OSP.doANT:
        ANTfile = OSP.Dir+'/ANTEATRresults'+OSP.thisName+'.dat'
        ANTdata = np.genfromtxt(ANTfile, dtype=float, encoding='utf8')
        # get the unique ANTEATR ideas (only one row per id here)
        # might have some missing if it misses
        ANTids = ANTdata[:,0].astype(int)
        unANTids = np.unique(ANTdata[:,0].astype(int))
        for i in unANTids:
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
            thisRes.ANTtimes = ANTdata[myidxs,1]
            thisRes.ANTrs    = ANTdata[myidxs,2]
            thisRes.ANTAWs   = ANTdata[myidxs,3]
            thisRes.ANTAWps  = ANTdata[myidxs,4]
            thisRes.ANTdelAxs = ANTdata[myidxs,5]
            thisRes.ANTdelCSs = ANTdata[myidxs,6]
            thisRes.ANTdelCSAxs = ANTdata[myidxs,7]
            
            thisRes.ANTvFs   = ANTdata[myidxs,8]
            thisRes.ANTvBs   = ANTdata[myidxs,9]
            thisRes.ANTvEs   = ANTdata[myidxs,10]
            thisRes.ANTvAxrs = ANTdata[myidxs,11]
            thisRes.ANTvAxps = ANTdata[myidxs,12]
            thisRes.ANTvCSrs = ANTdata[myidxs,13]
            thisRes.ANTvCSps = ANTdata[myidxs,14]
            thisRes.ANTB0s   = ANTdata[myidxs,15]
            thisRes.ANTCnms  = ANTdata[myidxs,16]
            # tau is constant for now
            thisRes.ANTtaus  = ANTdata[myidxs,17]
            # assuming [m,n] = [0,1]
            thisRes.ANTBtors = thisRes.ANTdelCSs * thisRes.ANTB0s * thisRes.ANTtaus
            thisRes.ANTBpols = 2 * thisRes.ANTdelCSs * thisRes.ANTB0s / (thisRes.ANTdelCSs**2+1) / thisRes.ANTCnms
            ResArr[thisRes.myID] = thisRes
            # calc est (max) duration - for nose impact
            Deltabr = thisRes.ANTdelCSs[-1] * np.tan(thisRes.ANTAWps[-1]*dtor) / (1 + thisRes.ANTdelCSs[-1] * np.tan(thisRes.ANTAWps[-1]*dtor))
            thisRes.ANTrr = Deltabr * thisRes.ANTrs[-1]
            thisRes.ANTdur = 4 * thisRes.ANTrr * 7e10 * (2*thisRes.ANTvBs[-1]+3*thisRes.ANTvCSrs[-1]) / (2*thisRes.ANTvBs[-1]+thisRes.ANTvCSrs[-1])**2 / 1e5 / 3600.
            # calc Kp
            dphidt = np.power(thisRes.ANTvFs[-1], 4/3.) * np.power(thisRes.ANTBpols[-1], 2./3.) 
            # Mays/Savani expression, best behaved for high Kp
            thisRes.ANTKp0 = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
            # calc density 
            thisRes.ANTrp = thisRes.ANTrr / thisRes.ANTdelCSs[-1]
            alpha = np.sqrt(1+16*thisRes.ANTdelAxs[-1]**2)/4/thisRes.ANTdelAxs[-1]
            thisRes.ANTLp = (np.tan(thisRes.ANTAWs[-1]*dtor)*(1-Deltabr) - alpha*Deltabr)/(1+thisRes.ANTdelAxs[-1]*np.tan(np.tan(thisRes.ANTAWs[-1]*dtor))) * thisRes.ANTrs[-1]
            vol = math.pi*thisRes.ANTrr*thisRes.ANTrp *  lenFun(thisRes.ANTdelAxs[-1])*thisRes.ANTrs[-1]
            thisRes.ANTn = OSP.mass*1e15 / vol / 1.67e-24 / (7e10)**3

    if OSP.doFIDO:
        FIDOfile = OSP.Dir+'/FIDOresults'+OSP.thisName+'.dat'
        FIDOdata = np.genfromtxt(FIDOfile, dtype=float, encoding='utf8')
        ids = FIDOdata[:,0].astype(int)
        unFIDOids = np.unique(ids)
        
        
        if OSP.includeSIT:
            SITfile = OSP.Dir+'/SITresults'+OSP.thisName+'.dat'
            SITdata = np.genfromtxt(SITfile, dtype=float, encoding='utf8')
            SITids = SITdata[:,0].astype(int)
        
        for i in unFIDOids:
            if OSP.doFC or OSP.doANT:
                thisRes = ResArr[i]
            else:
                thisRes = EnsRes(OSP.thisName)
            # Set as an impact not a miss (might not have run ANTEATR)
            thisRes.miss = False
            myidxs = np.where(ids==i)[0]
            thisRes.FIDOtimes = FIDOdata[myidxs,1]
            thisRes.FIDOBs    = FIDOdata[myidxs,2]
            thisRes.FIDOBxs   = FIDOdata[myidxs,3]
            thisRes.FIDOBys   = FIDOdata[myidxs,4]
            thisRes.FIDOBzs   = FIDOdata[myidxs,5]
            thisRes.FIDOvs    = FIDOdata[myidxs,6]
            isSheath = FIDOdata[myidxs,7]
            thisRes.FIDOidx = np.where(isSheath==1)[0]
                  
            Bvec = [thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs]
            Kp, BoutGSM   = calcKp(Bvec, DoY, thisRes.FIDOvs) 
            thisRes.FIDOKps   = Kp
            if (OSP.includeSIT) and (i in SITids):  
                thisRes.hasSheath = True
                myID = np.where(SITids == i)[0][0]
                isSheath = FIDOdata[myidxs,7]
                thisRes.SITidx = np.where(isSheath==0)[0]
                thisRes.SITdur = SITdata[myID, 1] #thisRes.FIDOtimes[thisRes.SITidx[-1]]-thisRes.FIDOtimes[0] 
                thisRes.SITcomp = SITdata[myID, 2]
                thisRes.SITMach = SITdata[myID, 3]
                thisRes.SITn    = SITdata[myID, 4]
                thisRes.SITvSheath = SITdata[myID, 5]
                thisRes.SITB    = SITdata[myID, 6]     
                thisRes.SITvShock = SITdata[myID,7]    
                thisRes.SITminBz = np.min(thisRes.FIDOBzs[thisRes.SITidx])
                thisRes.SITmaxB = np.max(thisRes.FIDOBs[thisRes.SITidx])
                thisRes.SITmaxKp = np.max(thisRes.FIDOKps[thisRes.SITidx])
                       
            ResArr[i] = thisRes
            
    # if haven't run FC may have fewer CMEs in ResArr than total runs if have misses
    for j in range(OSP.nRuns):
        if j not in ResArr.keys():
            thisRes = EnsRes(OSP.thisName)
            thisRes.miss = True
            ResArr[j] = thisRes
            
            
            
    # if we ran an ensemble load up the initial parameters for each member        
    if len(ResArr.keys()) > 1:
        ENSfile = OSP.Dir+'/EnsembleParams'+OSP.thisName+'.dat' 
        ENSdata = np.genfromtxt(ENSfile, dtype=None, encoding='utf8')
        global varied
        varied = ENSdata[0][1:] 
        nvar = len(varied) 
        for i in range(len(ENSdata)-1):
            row = ENSdata[i+1].astype(float)
            for j in range(nvar):
                if int(row[0]) in ResArr.keys():
                    ResArr[int(row[0])].EnsVal[varied[j]] = row[j+1]  
        # sort varied according to a nice order
        myOrder = ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEdelCSAx', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB', 'CMEvExp', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWcs', 'SWvA', 'FRBscale', 'FRtau', 'FRCnm', 'CMEvTrans', 'SWBx', 'SWBy', 'SWBz']  
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
    axes[0].plot(ResArr[0].FCrs, ResArr[0].FClats, linewidth=4, color='k')
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FClons, linewidth=4, color='k')
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCtilts, linewidth=4, color='k')
    
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
    plt.subplots_adjust(hspace=0.1,left=0.13,right=0.95,top=0.95,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_CPA.png')
    
def makeADVplot(ResArr):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[1,0], axes[1,0], axes[2,0], axes[2,0], axes[1,1], axes[1,1], axes[2,1]]
    c1   = ['LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue','LightGray', 'lightblue', 'LightGray']
    c2   = ['DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue','DarkGray', 'dodgerblue', 'DarkGray']
    if nEns > 1:
        fakers = np.linspace(1.1,10.05,100, endpoint=True)
        splineVals = np.zeros([nEns, 100, 11])
        means = np.zeros([100, 11])
        stds  = np.zeros([100, 11])
        lims  = np.zeros([100, 2, 11])
    
        i = 0
        # Repackage profiles
        for key in ResArr.keys():
            # Fit a spline to data since may be different lengths since take different times
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCAWs,bc_type='natural')
            splineVals[i,:, 0] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCAWps,bc_type='natural')
            splineVals[i,:, 1] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdelAxs,bc_type='natural')
            splineVals[i,:, 2] = thefit(fakers)   
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdelCSs,bc_type='natural')
            splineVals[i,:, 3] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvFs,bc_type='natural')
            splineVals[i,:, 4] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvEs,bc_type='natural')
            splineVals[i,:, 5] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvCSrs,bc_type='natural')
            splineVals[i,:, 6] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvCSps,bc_type='natural')
            splineVals[i,:, 7] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvAxrs,bc_type='natural')
            splineVals[i,:, 8] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvAxps,bc_type='natural')
            splineVals[i,:, 9] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdefs,bc_type='natural')
            splineVals[i,:, 10] = thefit(fakers)
                         
            i += 1
        for i in range(11):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color=c1[i]) 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color=c2[i]) 
        
    # Plot the seed profile
    axes[0].plot(ResArr[0].FCrs, ResArr[0].FCAWs, linewidth=4, color='k', zorder=3)
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FCAWps, linewidth=4, color='b', zorder=3)
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCdelAxs, linewidth=4, color='k', zorder=3)
    axes[3].plot(ResArr[0].FCrs, ResArr[0].FCdelCSs, linewidth=4, color='b', zorder=3)
    axes[4].plot(ResArr[0].FCrs, ResArr[0].FCvFs, linewidth=4, color='k', zorder=3)
    axes[5].plot(ResArr[0].FCrs, ResArr[0].FCvEs, linewidth=4, color='b', zorder=3)
    axes[6].plot(ResArr[0].FCrs, ResArr[0].FCvCSrs, linewidth=4, color='k', zorder=3)
    axes[7].plot(ResArr[0].FCrs, ResArr[0].FCvCSps, linewidth=4, color='b', zorder=3)
    axes[8].plot(ResArr[0].FCrs, ResArr[0].FCvAxrs, linewidth=4, color='k', zorder=3)
    axes[9].plot(ResArr[0].FCrs, ResArr[0].FCvAxps, linewidth=4, color='b', zorder=3)
    axes[10].plot(ResArr[0].FCrs, ResArr[0].FCdefs, linewidth=4, color='k', zorder=3)
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_AWs, all_AWps, all_delAxs, all_delCSs, all_vFs, all_vEs, all_vCSrs, all_vCSps, all_vAxrs, all_vAxps, all_defs = [], [], [], [], [], [], [], [], [], [], []
        for key in ResArr.keys():
            all_AWs.append(ResArr[key].FCAWs[-1])
            all_AWps.append(ResArr[key].FCAWps[-1])
            all_delAxs.append(ResArr[key].FCdelAxs[-1])
            all_delCSs.append(ResArr[key].FCdelCSs[-1])
            all_vFs.append(ResArr[key].FCvFs[-1])
            all_vEs.append(ResArr[key].FCvEs[-1])
            all_vCSrs.append(ResArr[key].FCvCSrs[-1])
            all_vCSps.append(ResArr[key].FCvCSps[-1])
            all_vAxrs.append(ResArr[key].FCvAxrs[-1])
            all_vAxps.append(ResArr[key].FCvAxps[-1])
            all_defs.append(ResArr[key].FCdefs[-1])
        fitAWs = norm.fit(all_AWs)
        fitAWps = norm.fit(all_AWps)
        fitdelAxs = norm.fit(all_delAxs)
        fitdelCSs = norm.fit(all_delCSs)
        fitvFs = norm.fit(all_vFs)
        fitvEs = norm.fit(all_vEs)
        fitvCSrs = norm.fit(all_vCSrs)
        fitvCSps = norm.fit(all_vCSps)
        fitvAxrs = norm.fit(all_vAxrs)
        fitvAxps = norm.fit(all_vAxps)
        fitdefs = norm.fit(all_defs)
        
        
        axes[0].text(0.97, 0.15, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\pm$'+'{:4.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05,  'AW$_{\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\pm$'+'{:4.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
        axes[2].text(0.97, 0.15, '$\delta_{Ax}$'+'{:4.1f}'.format(fitdelAxs[0])+'$\pm$'+'{:4.2f}'.format(fitdelAxs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.05, '$\delta_{CS}$'+'{:4.1f}'.format(fitdelCSs[0])+'$\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.15, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\pm$'+'{:4.1f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.05,  'v$_E$: '+'{:4.1f}'.format(fitvEs[0])+'$\pm$'+'{:4.1f}'.format(fitvEs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.15, 'v$_{CS,r}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\pm$'+'{:4.1f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.05,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(fitvCSps[0])+'$\pm$'+'{:4.1f}'.format(fitvCSps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')
        axes[8].text(0.97, 0.15, 'v$_{Ax,r}$: '+'{:4.1f}'.format(fitvAxrs[0])+'$\pm$'+'{:4.1f}'.format(fitvAxrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)
        axes[9].text(0.97, 0.05,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(fitvAxps[0])+'$\pm$'+'{:4.1f}'.format(fitvAxps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes, color='b')
        axes[10].text(0.97, 0.15, 'v$_{def}$: '+'{:4.1f}'.format(fitdefs[0])+'$\pm$'+'{:4.1f}'.format(fitdefs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[10].transAxes)              
    else:
        axes[0].text(0.97, 0.15, 'AW: '+'{:4.1f}'.format(ResArr[0].FCAWs[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05,  'AW$_{\perp}$: '+'{:4.1f}'.format(ResArr[0].FCAWps[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].text(0.97, 0.15, '$\delta_{Ax}$'+'{:4.1f}'.format(ResArr[0].FCdelAxs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.05, '$\delta_{CS}$'+'{:4.1f}'.format(ResArr[0].FCdelCSs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.15, 'v$_F$: '+'{:4.1f}'.format(ResArr[0].FCvFs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.05,  'v$_E$: '+'{:4.1f}'.format(ResArr[0].FCvEs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.15, 'v$_{CS,r}$: '+'{:4.1f}'.format(ResArr[0].FCvCSrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.05,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(ResArr[0].FCvCSps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')
        axes[8].text(0.97, 0.15, 'v$_{Ax,r}$: '+'{:4.1f}'.format(ResArr[0].FCvAxrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)
        axes[9].text(0.97, 0.05,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(ResArr[0].FCvAxps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes, color='b')
        axes[10].text(0.97, 0.05, 'v$_{def}$: '+'{:4.1f}'.format(ResArr[0].FCdefs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[10].transAxes)
                  
    # Labels
    axes[0].set_ylabel('AW, AW$_{\perp}$ ('+degree+')')
    axes[2].set_ylabel('$\delta_{Ax}$, $\delta_{\perp}$')
    axes[4].set_ylabel('v$_F$, v$_E$ (km/s)')
    axes[6].set_ylabel('v$_{Ax,r}$, v$_{Ax,\perp}$ (km/s)')
    axes[8].set_ylabel('v$_{CS,r}$, v$_{CS,\perp}$ (km/s)')
    axes[10].set_ylabel('v$_{def}$ (km/s)')
    axes[6].set_xlabel('Distance (R$_S$)')
    axes[10].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([1.01,10.15])
    plt.subplots_adjust(hspace=0.1,left=0.08,right=0.95,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ADV.png')
    
def makeDragplot(ResArr):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[1,0], axes[1,0], axes[2,0], axes[2,0], axes[1,1], axes[1,1], axes[2,1], axes[2,1]]
    c1   = ['LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue','LightGray', 'lightblue', 'LightGray', 'lightblue']
    c2   = ['DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue','DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue']
    # this isn't the exact end for all cases but don't really care in this figure
    # since more showing trend with distance and it flattens
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # get number of impacts, may be less than nEns
    nImp = 0
    hits = []
    for i in range(nEns):
        if not ResArr[i].miss:
            nImp += 1
            hits.append(i)
            
    
    # Arrays to hold spline results
    fakers = np.linspace(rStart,rEnd-5,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, 12])
    means = np.zeros([100, 12])
    stds  = np.zeros([100, 12])
    lims  = np.zeros([100, 2, 12])
    
    if nEns > 1:
        i = 0
        # Repackage profiles
        for key in hits:
            # Fit a spline to data since may be different lengths since take different times
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
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvEs,bc_type='natural')
            splineVals[i,:, 5] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvCSrs,bc_type='natural')
            splineVals[i,:, 6] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvCSps,bc_type='natural')
            splineVals[i,:, 7] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvAxrs,bc_type='natural')
            splineVals[i,:, 8] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvAxps,bc_type='natural')
            splineVals[i,:, 9] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTBtors,bc_type='natural')
            splineVals[i,:, 10] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTBpols,bc_type='natural')
            splineVals[i,:, 11] = thefit(fakers)
                         
            i += 1
        for i in range(11):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color=c1[i]) 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color=c2[i]) 
        
    # Plot the seed profile
    axes[0].plot(ResArr[0].ANTrs, ResArr[0].ANTAWs, linewidth=4, color='k', zorder=3)
    axes[1].plot(ResArr[0].ANTrs, ResArr[0].ANTAWps, linewidth=4, color='b', zorder=3)
    axes[2].plot(ResArr[0].ANTrs, ResArr[0].ANTdelAxs, linewidth=4, color='k', zorder=3)
    axes[3].plot(ResArr[0].ANTrs, ResArr[0].ANTdelCSs, linewidth=4, color='b', zorder=3)
    axes[4].plot(ResArr[0].ANTrs, ResArr[0].ANTvFs, linewidth=4, color='k', zorder=3)
    axes[5].plot(ResArr[0].ANTrs, ResArr[0].ANTvEs, linewidth=4, color='b', zorder=3)
    axes[6].plot(ResArr[0].ANTrs, ResArr[0].ANTvCSrs, linewidth=4, color='k', zorder=3)
    axes[7].plot(ResArr[0].ANTrs, ResArr[0].ANTvCSps, linewidth=4, color='b', zorder=3)
    axes[8].plot(ResArr[0].ANTrs, ResArr[0].ANTvAxrs, linewidth=4, color='k', zorder=3)
    axes[9].plot(ResArr[0].ANTrs, ResArr[0].ANTvAxps, linewidth=4, color='b', zorder=3)
    axes[10].plot(ResArr[0].ANTrs, ResArr[0].ANTBtors, linewidth=4, color='k', zorder=3)
    axes[11].plot(ResArr[0].ANTrs, ResArr[0].ANTBpols, linewidth=4, color='b', zorder=3)
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_AWs, all_AWps, all_delAxs, all_delCSs, all_vFs, all_vEs, all_vCSrs, all_vCSps, all_vAxrs, all_vAxps, all_Btors, all_Bpols = [], [], [], [], [], [], [], [], [], [], [], []
        for key in hits:
            all_AWs.append(ResArr[key].ANTAWs[-1])
            all_AWps.append(ResArr[key].ANTAWps[-1])
            all_delAxs.append(ResArr[key].ANTdelAxs[-1])
            all_delCSs.append(ResArr[key].ANTdelCSs[-1])
            all_vFs.append(ResArr[key].ANTvFs[-1])
            all_vEs.append(ResArr[key].ANTvEs[-1])
            all_vCSrs.append(ResArr[key].ANTvCSrs[-1])
            all_vCSps.append(ResArr[key].ANTvCSps[-1])
            all_vAxrs.append(ResArr[key].ANTvAxrs[-1])
            all_vAxps.append(ResArr[key].ANTvAxps[-1])
            all_Btors.append(ResArr[key].ANTBtors[-1])
            all_Bpols.append(ResArr[key].ANTBpols[-1])
        fitAWs = norm.fit(all_AWs)
        fitAWps = norm.fit(all_AWps)
        fitdelAxs = norm.fit(all_delAxs)
        fitdelCSs = norm.fit(all_delCSs)
        fitvFs = norm.fit(all_vFs)
        fitvEs = norm.fit(all_vEs)
        fitvCSrs = norm.fit(all_vCSrs)
        fitvCSps = norm.fit(all_vCSps)
        fitvAxrs = norm.fit(all_vAxrs)
        fitvAxps = norm.fit(all_vAxps)
        fitBtors = norm.fit(all_Btors)
        fitBpols = norm.fit(all_Bpols)
        
        
        axes[0].text(0.97, 0.95, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\pm$'+'{:4.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.85,  'AW$_{\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\pm$'+'{:4.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
        axes[2].text(0.97, 0.95, '$\delta_{Ax}$'+'{:4.2f}'.format(fitdelAxs[0])+'$\pm$'+'{:4.2f}'.format(fitdelAxs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '$\delta_{CS}$'+'{:4.2f}'.format(fitdelCSs[0])+'$\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.95, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\pm$'+'{:4.1f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.85,  'v$_E$: '+'{:4.1f}'.format(fitvEs[0])+'$\pm$'+'{:4.1f}'.format(fitvEs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.95, 'v$_{CS,r}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\pm$'+'{:4.1f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.85,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(fitvCSps[0])+'$\pm$'+'{:4.1f}'.format(fitvCSps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')
        axes[8].text(0.97, 0.95, 'v$_{Ax,r}$: '+'{:4.1f}'.format(fitvAxrs[0])+'$\pm$'+'{:4.1f}'.format(fitvAxrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)
        axes[9].text(0.97, 0.85,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(fitvAxps[0])+'$\pm$'+'{:4.1f}'.format(fitvAxps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes, color='b')
        axes[10].text(0.97, 0.95, 'B$_{t}$: '+'{:4.1f}'.format(fitBtors[0])+'$\pm$'+'{:4.1f}'.format(fitBtors[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[10].transAxes)
        axes[11].text(0.97, 0.85,  'B$_{p}$: '+'{:4.1f}'.format(fitBpols[0])+'$\pm$'+'{:4.1f}'.format(fitBpols[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[11].transAxes, color='b')
    else:
        axes[0].text(0.97, 0.95, 'AW: '+'{:4.1f}'.format(ResArr[0].ANTAWs[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.85,  'AW$_{\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTAWps[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].text(0.97, 0.95, '$\delta_{Ax}$'+'{:4.1f}'.format(ResArr[0].ANTdelAxs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '$\delta_{CS}$'+'{:4.1f}'.format(ResArr[0].ANTdelCSs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.95, 'v$_F$: '+'{:4.1f}'.format(ResArr[0].ANTvFs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.85,  'v$_E$: '+'{:4.1f}'.format(ResArr[0].ANTvEs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.95, 'v$_{CS,r}$: '+'{:4.1f}'.format(ResArr[0].ANTvCSrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.85,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTvCSps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')
        axes[8].text(0.97, 0.95, 'v$_{Ax,r}$: '+'{:4.1f}'.format(ResArr[0].ANTvAxrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)
        axes[9].text(0.97, 0.85,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTvAxps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes, color='b')
        axes[10].text(0.97, 0.95, 'B$_{t}$: '+'{:4.1f}'.format(ResArr[0].ANTBtors[-1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[10].transAxes)
        axes[11].text(0.97, 0.85,  'B$_{p}$: '+'{:4.1f}'.format(ResArr[0].ANTBpols[-1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[11].transAxes, color='b')
                  
    # Labels
    axes[0].set_ylabel('AW, AW$_{\perp}$ ('+degree+')')
    axes[2].set_ylabel('$\delta_{Ax}$, $\delta_{\perp}$')
    axes[4].set_ylabel('v$_F$, v$_E$ (km/s)')
    axes[6].set_ylabel('v$_{Ax,r}$, v$_{Ax,\perp}$ (km/s)')
    axes[8].set_ylabel('v$_{CS,r}$, v$_{CS,\perp}$ (km/s)')
    axes[10].set_ylabel('B$_t$, B$_p$ (nT)')
    axes[6].set_xlabel('Distance (R$_S$)')
    axes[10].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([rStart, rEnd])
    axes[10].set_yscale('log')
    plt.subplots_adjust(hspace=0.1,left=0.08,right=0.95,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Drag.png')

def makeAThisto(ResArr):
    fig, axes = plt.subplots(4, 2, figsize=(8,12), sharey=True)
    axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1], axes[2,0], axes[2,1], axes[3,0], axes[3,1]]
    all_vFs, all_vExps, all_TTs, all_durs, all_Bfs, all_Bms, all_ns, all_Kps  = [], [], [], [], [], [], [], []
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if not ResArr[key].miss:
            all_vFs.append(ResArr[key].ANTvFs[-1])
            all_vExps.append(ResArr[key].ANTvCSrs[-1])
            all_TTs.append(ResArr[key].ANTtimes[-1])
            all_durs.append(ResArr[key].ANTdur)
            all_Bfs.append(ResArr[key].ANTBpols[-1])
            all_Bms.append(ResArr[key].ANTBtors[-1])
            all_ns.append(ResArr[key].ANTn)
            all_Kps.append(ResArr[key].ANTKp0)
    
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_vFs, bins=10, color='#882255')
    n2, bins, patches = axes[1].hist(all_vExps, bins=10, color='#882255')
    n3, bins, patches = axes[2].hist(all_TTs, bins=10, color='#882255')
    n4, bins, patches = axes[3].hist(all_durs, bins=10, color='#882255')
    n5, bins, patches = axes[4].hist(all_Bfs, bins=10, color='#882255')
    n6, bins, patches = axes[5].hist(all_Bms, bins=10, color='#882255')
    n7, bins, patches = axes[6].hist(all_ns, bins=10, color='#882255')
    n8, bins, patches = axes[7].hist(all_Kps, bins=10, color='#882255')
    maxcount = np.max(np.max([n1, n2, n3, n4, n5, n6, n7, n8]))
    for i in range(8): axes[i].set_ylim(0, maxcount*1.2)
    
    # Add the mean and sigma from a normal fit
    fitvExps = norm.fit(all_vExps)
    fitvFs = norm.fit(all_vFs)
    fitTTs = norm.fit(all_TTs)
    fitdurs = norm.fit(all_durs)
    fitBfs = norm.fit(all_Bfs)
    fitBms = norm.fit(all_Bms)
    fitns = norm.fit(all_ns)
    fitKps = norm.fit(all_Kps)
    
    axes[0].text(0.97, 0.89, '{:4.2f}'.format(fitvFs[0])+'$\pm$'+'{:4.2f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)    
    axes[1].text(0.97, 0.89, str(int(fitvExps[0]))+'$\pm$'+str(int(fitvExps[1]))+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.85, '{:4.2f}'.format(fitTTs[0])+'$\pm$'+'{:4.2f}'.format(fitTTs[1])+' days', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    base = datetime.datetime(yr, 1, 1, 0, 0)
    date = base + datetime.timedelta(days=(DoY+fitTTs[0]))   
    dateLabel = date.strftime('%Y %b %d %H:%M ')
    axes[2].text(0.97, 0.94, dateLabel+'$\pm$'+'{:3.1f}'.format(fitTTs[1]*24)+' hrs', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.89, str(int(fitdurs[0]))+'$\pm$'+str(int(fitdurs[1]))+' hrs', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.89, str(int(fitBfs[0]))+'$\pm$'+str(int(fitBfs[1]))+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
    axes[5].text(0.97, 0.89, str(int(fitBms[0]))+'$\pm$'+str(int(fitBms[1]))+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    axes[6].text(0.97, 0.89, str(int(fitns[0]))+'$\pm$'+str(int(fitns[1]))+' cm$^{-3}$', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
    axes[7].text(0.97, 0.89, str(int(fitKps[0]))+'$\pm$'+str(int(fitKps[1])), horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)

    
    
    # Take out half the ticks for readability
    '''for i in range(8):
        xticks = axes[i].xaxis.get_major_ticks()
        ticks2hide = np.array(range(len(xticks)-1))[::2]+1
        for j in ticks2hide:
            xticks[j].label1.set_visible(False)'''
    
    # Labels
    axes[0].set_xlabel('v$_F$ (km/s)')
    axes[1].set_xlabel('v$_{Exp}$ (km/s)')
    axes[2].set_xlabel('Transit Time (days)')
    axes[3].set_xlabel('Duration (hours)')
    axes[4].set_xlabel('B$_F$ (nT)')
    axes[5].set_xlabel('B$_M$ (nT)')
    axes[6].set_xlabel('n (cm$^{-3}$)')
    axes[7].set_xlabel('Kp')
    for i in range(8): axes[i].set_ylabel('Counts')
    plt.subplots_adjust(hspace=0.35, left=0.1,right=0.95,top=0.98,bottom=0.06)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANT.png')
     
def makeISplot(ResArr):
    fig, axes = plt.subplots(6, 1, sharex=True, figsize=(8,12))
    mindate = None
    maxdate = None
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes is not None: #not ResArr[key].miss:
            #datesNUM = ResArr[key].FIDOtimes+DoY   
            #dates = datetime.datetime(yr, 1, 1) + datetime.timedelta(datesNUM - 1)
            base = datetime.datetime(yr, 1, 1, 1, 0)
            #dates = ResArr[key].FIDOtimes
            if not OSP.doANT:
                dates = np.array([base + datetime.timedelta(days=(i-1)) for i in ResArr[key].FIDOtimes])
            else:
                dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes])
            if mindate is None: 
                mindate = np.min(dates)
                maxdate = np.max(dates)
            axes[0].plot(dates, ResArr[key].FIDOBs, linewidth=2, color='DarkGray')
            axes[1].plot(dates, ResArr[key].FIDOBxs, linewidth=2, color='DarkGray')
            axes[2].plot(dates, ResArr[key].FIDOBys, linewidth=2, color='DarkGray')
            axes[3].plot(dates, ResArr[key].FIDOBzs, linewidth=2, color='DarkGray')
            axes[4].plot(dates, ResArr[key].FIDOKps, linewidth=2, color='DarkGray')
            axes[5].plot(dates, ResArr[key].FIDOvs, linewidth=2, color='DarkGray')
            if np.min(dates) < mindate: mindate = np.min(dates)
            if np.max(dates) > maxdate: maxdate = np.max(dates)        
    
    # Plot the ensemble seed
    dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[0].FIDOtimes])
    axes[0].plot(dates, ResArr[0].FIDOBs, linewidth=4, color='b')
    axes[1].plot(dates, ResArr[0].FIDOBxs, linewidth=4, color='b')
    axes[2].plot(dates, ResArr[0].FIDOBys, linewidth=4, color='b')
    axes[3].plot(dates, ResArr[0].FIDOBzs, linewidth=4, color='b')
    axes[4].plot(dates, ResArr[0].FIDOKps, linewidth=4, color='b')
    axes[5].plot(dates, ResArr[0].FIDOvs, linewidth=4, color='b')
    # Make Kps integers only
    Kpmin, Kpmax = int(np.min(ResArr[0].FIDOKps)), int(np.max(ResArr[0].FIDOKps))+1
    axes[4].set_yticks(range(Kpmin, Kpmax+2))
    
    
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('Kp')
    axes[5].set_ylabel('v (km/s)')
    
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
        # get in situ v data and uncomment....   
        # axes[5].plot(ObsData[0,:], ObsData[7,:], linewidth=4, color='m')

    
    fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.15)
     
    #plt.show()
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_IS.png')    
    
def makeFIDOhistos(ResArr):
    fig, axes = plt.subplots(2, 3, figsize=(9,7), sharey=True)
    axes = [axes[0,0], axes[1,0], axes[1,1], axes[1,2], axes[0,1], axes[0,2]]
    all_dur = []
    all_B   = []
    all_Bz  = []
    all_Kp  = []
    all_vF  = []
    all_vE  = []
    
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if ResArr[key].FIDOtimes is not None:
            all_dur.append(ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])
            all_Bz.append(np.min(ResArr[key].FIDOBzs))
            all_Kp.append(np.max(ResArr[key].FIDOKps))
            all_B.append(np.max(ResArr[key].FIDOBs))
            all_vF.append(ResArr[key].FIDOvs[0])
            all_vE.append(0.5*(ResArr[key].FIDOvs[0] - ResArr[key].FIDOvs[-1]))
            
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color='#882255')
    n2, bins, patches = axes[1].hist(all_Bz, bins=10, color='#882255')
    n3, bins, patches = axes[2].hist(all_B, bins=10, color='#882255')
    n4, bins, patches = axes[3].hist(all_Kp, bins=10, color='#882255')
    n5, bins, patches = axes[4].hist(all_vF, bins=10, color='#882255')
    n6, bins, patches = axes[5].hist(all_vE, bins=10, color='#882255')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6)])
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitDur = norm.fit(all_dur)
    fitBz = norm.fit(all_Bz)
    fitB  = norm.fit(all_B)
    fitKp = norm.fit(all_Kp)
    fitvF  = norm.fit(all_vF)
    fitvE  = norm.fit(all_vE)
    axes[0].text(0.97, 0.95, '{:4.2f}'.format(fitDur[0])+'$\pm$'+'{:4.2f}'.format(fitDur[1])+' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, '{:4.1f}'.format(fitBz[0])+'$\pm$'+'{:4.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.95, '{:4.1f}'.format(fitB[0])+'$\pm$'+'{:4.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.95, '{:4.1f}'.format(fitKp[0])+'$\pm$'+'{:4.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.95, '{:4.1f}'.format(fitvF[0])+'$\pm$'+'{:4.1f}'.format(fitvF[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
    axes[5].text(0.97, 0.95, '{:4.1f}'.format(fitvE[0])+'$\pm$'+'{:4.1f}'.format(fitvE[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    
    # Labels
    axes[0].set_xlabel('Duration (days)')
    axes[1].set_xlabel('Minimum B$_z$ (nT)')
    axes[2].set_xlabel('Maximum B (nT)')
    axes[3].set_xlabel('Maximum Kp')
    axes[4].set_xlabel('v$_F$ (km/s)')
    axes[5].set_xlabel('v$_{Exp}$ (km/s)')
    for i in range(6): axes[i].set_ylabel('Counts')    
    
    plt.subplots_adjust(wspace=0.15, hspace=0.25,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_FIDOhist.png')

def makeSIThistos(ResArr):
    fig, axes = plt.subplots(3, 3, figsize=(8,10), sharey=True)
    axes = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1], axes[1,2], axes[2,0], axes[2,1], axes[2,2]]
    all_dur  = []
    all_comp = []
    all_Mach = []
    all_n    = []
    all_vShock = []
    all_B    = []
    all_Bz   = []
    all_vSheath = []
    all_Kp   = []
        
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if ResArr[key].hasSheath:
            all_dur.append(ResArr[key].SITdur)
            all_comp.append(ResArr[key].SITcomp)
            all_Mach.append(ResArr[key].SITMach)
            all_n.append(ResArr[key].SITn)
            all_vShock.append(ResArr[key].SITvShock)
            all_B.append(ResArr[key].SITB)
            all_Bz.append(ResArr[key].SITminBz)
            all_vSheath.append(ResArr[key].SITvSheath)
            all_Kp.append(ResArr[key].SITmaxKp)
            
                        
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color='#882255')
    n2, bins, patches = axes[1].hist(all_comp, bins=10, color='#882255')
    n3, bins, patches = axes[2].hist(all_n, bins=10, color='#882255')
    n4, bins, patches = axes[3].hist(all_vShock, bins=10, color='#882255')
    n5, bins, patches = axes[4].hist(all_vSheath, bins=10, color='#882255')
    n6, bins, patches = axes[5].hist(all_Mach, bins=10, color='#882255')
    n7, bins, patches = axes[6].hist(all_B, bins=10, color='#882255')
    # Bz might be peaked at 0 if has no neg values
    n8, bins, patches = axes[7].hist(all_Bz, bins=10, color='#882255')
    n9, bins, patches = axes[8].hist(all_Kp, bins=10, color='#882255')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6), np.max(n7), np.max(n8), np.max(n9)])
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitDur = norm.fit(all_dur)
    fitComp = norm.fit(all_comp)
    fitMach = norm.fit(all_Mach)
    fitn = norm.fit(all_n)
    fitvShock = norm.fit(all_vShock)
    fitB = norm.fit(all_B)
    fitBz = norm.fit(all_Bz)
    fitvSheath = norm.fit(all_vSheath)
    fitKp = norm.fit(all_Kp)

    axes[0].text(0.97, 0.95, '{:4.2f}'.format(fitDur[0])+'$\pm$'+'{:4.2f}'.format(fitDur[1])+' hours', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, '{:4.2f}'.format(fitComp[0])+'$\pm$'+'{:4.2f}'.format(fitComp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.95, '{:4.1f}'.format(fitn[0])+'$\pm$'+'{:4.1f}'.format(fitn[1])+' cm$^{-3}$', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.95, '{:4.1f}'.format(fitvShock[0])+'$\pm$'+'{:4.1f}'.format(fitvShock[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.95, '{:4.1f}'.format(fitvSheath[0])+'$\pm$'+'{:4.1f}'.format(fitvSheath[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)    
    axes[5].text(0.97, 0.95, '{:4.1f}'.format(fitMach[0])+'$\pm$'+'{:4.1f}'.format(fitMach[1]), horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    axes[6].text(0.97, 0.95, '{:4.1f}'.format(fitB[0])+'$\pm$'+'{:4.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
    axes[7].text(0.97, 0.95, '{:4.1f}'.format(fitBz[0])+'$\pm$'+'{:4.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)
    axes[8].text(0.97, 0.95, '{:4.1f}'.format(fitKp[0])+'$\pm$'+'{:4.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)    
    
    # Labels
    axes[0].set_xlabel('Duration (hours)')
    axes[1].set_xlabel('Compression Ratio')
    axes[2].set_xlabel('Density (cm$^{-3}$)')
    axes[3].set_xlabel('v$_{Shock}$ (km/s)')
    axes[4].set_xlabel('v$_{Sheath}$ (km/s)')
    axes[5].set_xlabel('Mach Number')
    axes[6].set_xlabel('B (nT)')
    axes[7].set_xlabel('min Bz (nT)')
    axes[8].set_xlabel('Kp')
    
    for i in range(9): axes[i].set_ylabel('Counts')    
    
    plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_SIThist.png')

    
def makeEnsplot(ResArr):
    # At max want to show variation with lat, lon, tilt, AT, v1AU
    # duration, Bz, Kp (8 vals) but depends on what we ran
    deg = '('+'$^\circ$'+')'

    out2outLab = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\perp}$\n'+deg, 'CMEdelAx':'$\delta_{Ax}$', 'CMEdelCS':'$\delta_{CS}$', 'CMEdelCSAx':'$\delta_{CA}$', 'CMEvF':'v$_{F}$\n(km/s)', 'CMEvExp':'v$_{Exp}$\n(km/s)', 'TT':'Transit\nTime\n(days)', 'Dur':'Dur\n(hours)', 'n':'n\n(cm$^{-3}$)',  'B':'max B (nT)', 'Bz':'min Bz\n(nT)', 'Kp':'max Kp'}
    
    myLabs = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEvr':'v$_F$\n(km/s)', 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\perp}$\n'+deg, 'CMEdelAx':'$\delta_{Ax}$', 'CMEdelCS':'$\delta_{CS}$', 'CMEdelCSAx':'$\delta_{CA}$', 'CMEr':'R$_{F0}$ (R$_S$)', 'FCrmax':'FC end R$_{F0}$\n (R$_S$)', 'FCraccel1':'FC R$_{v1}$\n (km/s)', 'FCraccel2':'FC R$_{v2}$\n (km/s)', 'FCvrmin':'FC v$_{0}$\n (km/s)', 'FCAWmin':'FC AW$_{0}$\n'+deg, 'FCAWr':'FC R$_{AW}$\n (R$_S$)', 'CMEM':'M$_{CME}$\n(10$^{15}$ g)', 'FCrmaxM':'FC R$_{M}$\n(R$_S$)', 'FRB':'B$_0$ (nT)', 'CMEvExp':'v$_{Exp}$\n (km/s)', 'SWCd': 'C$_d$', 'SWCdp':'C$_{d,\perp}$', 'SWn':'n$_{SW}$\n(cm$^{-3}$)', 'SWv':'v$_{SW}$\n(km/s)', 'SWB':'B$_{SW}$\n(nT)', 'SWcs':'c$_s$\n(km/s)', 'SWvA':'v$_A$\n(km/s)', 'FRBscale':'B scale', 'FRtau':'$\\tau', 'FRCnm':'C$_{nm}$', 'CMEvTrans':'v$_{Trans}$\n(km/s)', 'SWBx':'SW B$_x$\n(nT)', 'SWBy':'SW B$_y$\n(nT)', 'SWBz':'SW B$_z$\n(nT)'}
    
    nVert = 0
    configID = 0
    if OSP.doFC: configID += 100
    if OSP.doANT: configID += 10
    if OSP.doFIDO: configID += 1
    nVertDict = {100:9, 110:13, 111:15, 11:12, 10:10, 1:4}
    nVert = nVertDict[configID]
    outDict = {100:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp'], 110:['CMElat', 'CMElon',  'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'Kp'], 111:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n',  'B', 'Bz', 'Kp'], 11:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n',  'B', 'Bz', 'Kp'], 10:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'Kp'], 1:['Dur',  'B', 'Bz',  'Kp']}
    # number of vertical plots depends on num params varied
    nHoriz = len(varied)
    
    # get impacts, may be less than nEns
    hits = []
    for i in range(nEns):
        if not ResArr[i].miss:
            hits.append(i)
    
    # group EnsRes once to avoid doing in each plot
    nRuns = len(hits) # might need to change to throw out misses
    EnsVal = np.zeros([nHoriz, nRuns])
    i = 0
    for key in hits:
        j = 0
        for item in varied:
            EnsVal[j,i] = ResArr[key].EnsVal[item]
            j += 1
        i += 1
        
    # group the results
    OSPres = {}#np.zeros([nRuns, nVert])
    for item in outDict[configID]: OSPres[item] = []
    counter = 0
    i = 0
    goodIDs = []
    for key in hits:
        if not ResArr[key].miss: goodIDs.append(key)
        elif configID == 100: goodIDs.append(key)
        
        for item in outDict[configID]:
            if item == 'CMElat':
                OSPres[item].append(ResArr[key].FClats[-1])
            if item == 'CMElon':
                OSPres[item].append(ResArr[key].FClons[-1])
            if item == 'CMEtilt':
                OSPres[item].append(ResArr[key].FCtilts[-1])
            if item == 'CMEAW':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTAWs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCAWs[-1])
            if item == 'CMEAWp':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTAWps[-1])
                else:
                    OSPres[item].append(ResArr[key].FCAWps[-1])
            if item == 'CMEdelAx':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTdelAxs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCdelAxs[-1])
            if item == 'CMEdelCS':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTdelCSs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCdelCSs[-1])
            if item == 'CMEvF':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTvFs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCvFs[-1])
            if item == 'CMEvExp':
                if OSP.doANT:
                    OSPres[item].append(ResArr[key].ANTvCSrs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCvCSrs[-1])
            if not ResArr[key].miss:
                if item == 'TT':
                    OSPres[item].append(ResArr[key].ANTtimes[-1])                    
                if item == 'Dur':
                    if OSP.doFIDO:
                        OSPres[item].append((ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])*24)
                    else:
                        OSPres[item].append(ResArr[key].ANTdur)
                if item == 'n':
                    OSPres[item].append(ResArr[key].ANTn)                    
                if item == 'Kp':
                    if OSP.doFIDO:
                        OSPres[item].append(np.max(ResArr[key].FIDOKps))
                    else:
                        OSPres[item].append(ResArr[key].ANTKp0)
                if item == 'B':
                    OSPres[item].append(np.max(ResArr[key].FIDOBs))                                
                if item == 'Bz':
                    OSPres[item].append(np.min(ResArr[key].FIDOBzs))
            else:                    
                if item == 'TT': None                  
                if item == 'Dur': None
                if item == 'n': None                   
                if item == 'Kp': None
                if item == 'B': None                              
                if item == 'Bz': None
        
                
    print ('Number of hits: ', len(goodIDs)) 
    
    for item in outDict[configID]:
        OSPres[item] = np.array(OSPres[item])
        print (item, np.mean(OSPres[item]), np.std(OSPres[item]))  

    f, a = plt.subplots(1, 1)
    img = a.imshow(np.array([[0,1]]), cmap="cool")
    
    fig, axes = plt.subplots(nVert, nHoriz, figsize=(1.2*nHoriz+1,1.2*nVert+2))
       
    for i in range(nVert-1):
        for j in range(nHoriz):
            axes[i,j].set_xticklabels([])
    for j in range(nHoriz-1):
        for i in range(nVert):
            axes[i,j+1].set_yticklabels([])
    
    for i in range(nHoriz):
        for j in range(nVert):
            col = np.abs(pearsonr(EnsVal[i,:], OSPres[outDict[configID][j]])[0])*np.ones(len(goodIDs))
            axes[j,i].scatter(EnsVal[i,:], OSPres[outDict[configID][j]], c=cm.cool(col))
            
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
    for i in range(nHoriz): axes[-1,i].set_xlabel(myLabs[varied[i]])  
    for j in range(nVert):  axes[j,0].set_ylabel(out2outLab[outDict[configID][j]])  
    
    if configID not in [1,100]:    
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.1, bottom=0.15, top=0.97, right=0.99) 
        cbar_ax = fig.add_axes([0.15, 0.04, 0.79, 0.02])    
    if configID == 1:
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.1, bottom=0.3, top=0.97, right=0.99)
        cbar_ax = fig.add_axes([0.15, 0.09, 0.79, 0.02]) 
    if configID == 100:    
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.1, bottom=0.18, top=0.97, right=0.99) 
        cbar_ax = fig.add_axes([0.15, 0.06, 0.79, 0.02])    
        
    cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')   
    cb.set_label('Correlation') 
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
    KpArr = np.zeros([10, nx])
    # Calculate the values at cell midpoints
    Kptimes = np.array([mindate + 1.5/24 + i*3./24. for i in range(nx)]) 
    
    # fill in KpArr
    counter = 0
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes is not None:
            counter += 1
            tBounds = [ResArr[key].FIDOtimes[0], ResArr[key].FIDOtimes[-1]]
            try:
                thefit = CubicSpline(ResArr[key].FIDOtimes,ResArr[key].FIDOKps,bc_type='natural')
            except:
                # might have duplicate at front
                thefit = CubicSpline(ResArr[key].FIDOtimes[1:],ResArr[key].FIDOKps[1:],bc_type='natural')
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
    
    # get impacts, may be less than nEns
    hits = []
    for i in range(nEns):
        if not ResArr[i].miss:
            hits.append(i)
    
    for key in hits:#ResArr.keys():
        thisLat = ResArr[key].FClats[-1]
        thisLon = ResArr[key].FClons[-1]-OSP.satPos[1]
        thisTilt  = ResArr[key].FCtilts[-1]
        thisAW    = ResArr[key].FCAWs[-1]
        thisAWp   = ResArr[key].FCAWps[-1]
        thisR     = ResArr[key].FCrs[-1]
        thisDelAx = ResArr[key].FCdelAxs[-1]
        thisDelCS = ResArr[key].FCdelCSs[-1]

        # option to replace AW/del with ANT values as may make difference
        if OSP.doANT:
            thisAW    = ResArr[key].ANTAWs[-1]
            thisAWp   = ResArr[key].ANTAWps[-1]
            thisR     = ResArr[key].ANTrs[-1]
            thisDelAx = ResArr[key].ANTdelAxs[-1]
            thisDelCS = ResArr[key].ANTdelCSs[-1]
            

        # calculate widths in Rs
        alpha = np.sqrt(1+16*thisDelAx**2)/4/thisDelAx
        # normal vector dot z
        ndotz = 1./alpha
        Deltabr = thisDelCS * np.tan(thisAWp*dtor) / (1 + thisDelCS * np.tan(thisAWp*dtor))
        CMElens = np.zeros(7)
        CMElens[0] = thisR
        CMElens[3] = Deltabr * CMElens[0]
        CMElens[6]  = (np.tan(thisAW*dtor)*(1-Deltabr) - alpha*Deltabr)/(1+thisDelAx*np.tan(thisAW*dtor)) * CMElens[0]
        CMElens[5]  = thisDelAx * CMElens[6]
        CMElens[4] = CMElens[3] / thisDelCS
        CMElens[2]  = CMElens[0] - CMElens[5] - CMElens[3]
        rCent = CMElens[2] + CMElens[5]
        CMElens[1] = CMElens[6] + CMElens[3]*alpha
        fullWid = CMElens[1]
        crossWid = CMElens[4]
                
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
        
        sns = np.sign(thetas)
        xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(thetas)
        zFR = 0.5 * sns * CMElens[2] * (np.sin(np.abs(thetas)) + np.sqrt(1 - np.cos(np.abs(thetas))))            
        
        points =  CART2SPH(cart2cart([xFR,0.,zFR], thisLat, thisLon, thisTilt))
        
        # check if x increasing or not, reverse if need to
        if points[1][-1] < points[1][0]:
            points[0] = points[0][::-1]
            points[1] = points[1][::-1]
            points[2] = points[2][::-1]
        
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
                #print (x,y)
                if (x >= lons[0]) & (x <= lons[-1]):
                    #print ('here')
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
    if OSP.doANT:
        all_times = []
        for key in ResArr.keys():
            if not ResArr[key].miss:
                all_times.append(ResArr[key].ANTtimes[-1])
        # satellite position at time of impact
        dlon = np.mean(all_times) * OSP.Sat_rot
    else:
        dlon = 0.
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
    
if OSP.doFC:
    # Make CPA plot
    makeCPAplot(ResArr)  
    
    # Make the AW, delta, v plot
    makeADVplot(ResArr)
    

if OSP.doANT:
    # Make drag profile
    makeDragplot(ResArr)

if OSP.doFIDO:
    # Make in situ plot
    makeISplot(ResArr)

# Ensemble plots
if nEns > 1:
    if OSP.doANT:
        # Make arrival time hisogram 
        makeAThisto(ResArr)
    
    if OSP.doFIDO:
        # FIDO histos- duration, minBz
        makeFIDOhistos(ResArr)
        if OSP.includeSIT:
            makeSIThistos(ResArr)
        
        # Kp probability timeline
        makeKpprob(ResArr)

    if OSP.doFC:
        # Make location contour plot
        makeImpContours(ResArr)

    # Ensemble input-output plot
    makeEnsplot(ResArr)


        