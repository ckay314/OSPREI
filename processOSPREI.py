import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm, pearsonr
from scipy import ndimage
import datetime
import matplotlib.dates as mdates

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
from ANT_PUP import lenFun

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
            self.FCdelCAs   = None
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
            self.FIDOns     = None
            self.FIDOtems   = None
            self.FIDOKps    = None
            self.FIDOnormrs = None
            self.isSheath   = None
            self.hasSheath  = False 
            
            # flags for misses and fails-> no ANT/FIDOdata
            self.miss = True
            self.fail = False
            
            # Dictionary for ensemble things
            self.EnsVal = {}
 
def txt2obj(GCStime):
    ResArr = {}
    
    if not OSP.noDate:
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
    else: 
        DoY = 0 # needed for Kp calc

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
            if FCdata[myidxs[-1],2] == FCdata[myidxs[-2],2]:
                myidxs = myidxs[:-1]
            thisRes.FCtimes = FCdata[myidxs,1]
            thisRes.FCrs    = FCdata[myidxs,2]
            thisRes.FClats  = FCdata[myidxs,3]
            thisRes.FClons  = FCdata[myidxs,4]
            thisRes.FClonsS = thisRes.FClons - (OSP.satPos[1] - (360./27.2753) * (GCStime/24.))
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
            thisRes.FCdelCAs = FCdata[myidxs,18]
        
            # Put it in the results array        
            ResArr[i] = thisRes
        
    if OSP.doANT:
        ANTfile = OSP.Dir+'/ANTEATRresults'+OSP.thisName+'.dat'
        ANTdata = np.genfromtxt(ANTfile, dtype=float, encoding='utf8')
        # get the unique ANTEATR ideas (only one row per id here)
        # might have some missing if it misses
        ANTids = ANTdata[:,0].astype(int)
        unANTids = np.unique(ANTdata[:,0].astype(int))
        
        if OSP.doPUP:
            PUPfile = OSP.Dir+'/PUPresults'+OSP.thisName+'.dat'
            PUPdata = np.genfromtxt(PUPfile, dtype=float, encoding='utf8')
            
        global nHits, nFails
        nHits = 0
        nFails = 0
        
        for i in unANTids:
            # Check if we already have set up the objects
            # If not do now
            if not OSP.doFC:  
                thisRes = EnsRes(OSP.thisName)
                thisRes.myID = i
            else:
                thisRes = ResArr[i]
                
            # Set as an impact not a miss
            myidxs = np.where(ANTids==i)[0]  
            if int(ANTdata[myidxs[0],1]) == 8888:
                thisRes.miss = False
                thisRes.fail = True
                nFails += 1
            else:
                thisRes.miss = False
                nHits += 1
                thisRes.ANTtimes = ANTdata[myidxs,1]
                if OSP.doFC:
                    thisRes.ANTtimes = ANTdata[myidxs,1]+thisRes.FCtimes[-1]/60/24.
                else:
                    thisRes.ANTtimes = ANTdata[myidxs,1]
                thisRes.ANTrs    = ANTdata[myidxs,2]
                thisRes.ANTAWs   = ANTdata[myidxs,3]
                thisRes.ANTAWps  = ANTdata[myidxs,4]
                thisRes.ANTdelAxs = ANTdata[myidxs,5]
                thisRes.ANTdelCSs = ANTdata[myidxs,6]
                thisRes.ANTdelCSAxs = ANTdata[myidxs,7]
            
                thisRes.ANTvFs   = ANTdata[myidxs,8]
                thisRes.ANTvEs   = ANTdata[myidxs,9]
                thisRes.ANTvBs   = ANTdata[myidxs,10]
                thisRes.ANTvAxrs = ANTdata[myidxs,13]
                thisRes.ANTvAxps = ANTdata[myidxs,14]
                thisRes.ANTvCSrs = ANTdata[myidxs,11]
                thisRes.ANTvCSps = ANTdata[myidxs,12]
                thisRes.ANTB0s   = ANTdata[myidxs,15]
                thisRes.ANTCnms  = ANTdata[myidxs,16]
                # tau is constant for now
                thisRes.ANTtaus  = ANTdata[myidxs,17]
                thisRes.ANTns    = ANTdata[myidxs,18]
                thisRes.ANTlogTs = ANTdata[myidxs,19] 
                
                # idx of where sheath and/or FR start, will be replaced if FIDOing
                thisRes.ANTshidx = -1
                thisRes.ANTFRidx = -1
                
                # assuming [m,n] = [0,1]
                thisRes.ANTBtors = thisRes.ANTdelCSs * thisRes.ANTB0s * thisRes.ANTtaus
                thisRes.ANTBpols = 2 * thisRes.ANTdelCSs * thisRes.ANTB0s / (thisRes.ANTdelCSs**2+1) / thisRes.ANTCnms
                # calc est (max) duration - for nose impact
                Deltabr = thisRes.ANTdelCSs[-1] * np.tan(thisRes.ANTAWps[-1]*dtor) / (1 + thisRes.ANTdelCSs[-1] * np.tan(thisRes.ANTAWps[-1]*dtor))
                thisRes.ANTrr = Deltabr * thisRes.ANTrs[-1]
                vB = thisRes.ANTvFs[-1]-thisRes.ANTvCSrs[-1]
                thisRes.ANTdur = 4 * thisRes.ANTrr * 7e10 * (2*vB+3*thisRes.ANTvCSrs[-1]) / (2*vB+thisRes.ANTvCSrs[-1])**2 / 1e5 / 3600.
                thisRes.ANTCMEwids = thisRes.ANTdelCSs * np.tan(thisRes.ANTAWps*dtor) / (1 + thisRes.ANTdelCSs* np.tan(thisRes.ANTAWps*dtor)) * thisRes.ANTrs[-1]
                
                # calc Kp
                dphidt = np.power(thisRes.ANTvFs[-1], 4/3.) * np.power(thisRes.ANTBpols[-1], 2./3.) 
                # Mays/Savani expression, best behaved for high Kp
                thisRes.ANTKp0 = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
                # calc density 
                '''thisRes.ANTrp = thisRes.ANTrr / thisRes.ANTdelCSs[-1]
                alpha = np.sqrt(1+16*thisRes.ANTdelAxs[-1]**2)/4/thisRes.ANTdelAxs[-1]
                thisRes.ANTLp = (np.tan(thisRes.ANTAWs[-1]*dtor)*(1-Deltabr) - alpha*Deltabr)/(1+thisRes.ANTdelAxs[-1]*np.tan(np.tan(thisRes.ANTAWs[-1]*dtor))) * thisRes.ANTrs[-1]
                vol = math.pi*thisRes.ANTrr*thisRes.ANTrp *  lenFun(thisRes.ANTdelAxs[-1])*thisRes.ANTrs[-1]
                thisRes.ANTn = OSP.mass*1e15 / vol / 1.67e-24 / (7e10)**3'''
                
                
                if OSP.doPUP:
                    thisRes.PUPvshocks = PUPdata[myidxs,1]
                    thisRes.PUPcomps = PUPdata[myidxs,2]
                    thisRes.PUPMAs = PUPdata[myidxs,3]
                    thisRes.PUPwids = PUPdata[myidxs,4]
                    thisRes.PUPdurs = PUPdata[myidxs,5]
                    thisRes.PUPMs = PUPdata[myidxs,6]
                    thisRes.PUPns = PUPdata[myidxs,7]
                    thisRes.PUPlogTs = PUPdata[myidxs,8]
                    thisRes.PUPBthetas = PUPdata[myidxs,9]
                    thisRes.PUPBs = PUPdata[myidxs,10]
                    thisRes.PUPvts = PUPdata[myidxs,11]
                    thisRes.PUPinit = PUPdata[myidxs,12]
                                        
                ResArr[thisRes.myID] = thisRes

    if OSP.doFIDO:
        FIDOfile = OSP.Dir+'/FIDOresults'+OSP.thisName+'.dat'
        FIDOdata = np.genfromtxt(FIDOfile, dtype=float, encoding='utf8')
        ids = FIDOdata[:,0].astype(int)
        unFIDOids = np.unique(ids)
        
        if OSP.includeSIT:
            SITfile = OSP.Dir+'/SITresults'+OSP.thisName+'.dat'
            SITdata = np.genfromtxt(SITfile, dtype=float, encoding='utf8')
            if len(SITdata.shape) > 1:
                SITids = SITdata[:,0].astype(int)
            else:
                if np.size(SITdata) != 0:
                    SITids = np.array([0]) # single case
                    SITdata = SITdata.reshape([1,-1])
                else:
                    SITids = []
            
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
            thisRes.FIDOns    = FIDOdata[myidxs,7]
            thisRes.FIDOtems  = np.power(10,FIDOdata[myidxs,8])
            thisRes.regions = FIDOdata[myidxs,9]
            regions = FIDOdata[myidxs,9]
            thisRes.FIDO_shidx = np.where(thisRes.regions==0)[0]
            thisRes.FIDO_FRidx = np.where(thisRes.regions==1)[0]
            thisRes.FIDO_SWidx = np.where(np.abs(thisRes.regions-100)<10)[0]
            
            # derived paramters
            thisRes.FIDO_FRdur = (thisRes.FIDOtimes[thisRes.FIDO_FRidx[-1]] - thisRes.FIDOtimes[thisRes.FIDO_FRidx[0]]) * 24
            thisRes.FIDO_shdur = (thisRes.FIDOtimes[thisRes.FIDO_shidx[-1]] - thisRes.FIDOtimes[thisRes.FIDO_shidx[0]]) * 24
            thisRes.FIDO_FRexp = 0.5*(thisRes.FIDOvs[thisRes.FIDO_FRidx[0]] - thisRes.FIDOvs[thisRes.FIDO_FRidx[-1]]) 
            
            # get corresponding idxs for ANT data
            if OSP.doANT:
                if OSP.doPUP:
                    tSH =thisRes.FIDOtimes[thisRes.FIDO_shidx[0]], 
                    thisRes.ANTshidx = np.min(np.where(thisRes.ANTtimes >= tSH))
                tFR = thisRes.FIDOtimes[thisRes.FIDO_FRidx[0]]
                thisRes.ANTFRidx = np.min(np.where(thisRes.ANTtimes >= tFR))
                
                # redo calc Kp with actual front
                dphidt = np.power(thisRes.ANTvFs[thisRes.ANTFRidx], 4/3.) * np.power(thisRes.ANTBpols[thisRes.ANTFRidx], 2./3.) 
                # Mays/Savani expression, best behaved for high Kp
                thisRes.ANTKp0 = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
                
                # reset ANT dur with more accurate version
                thisRes.ANTdur = thisRes.FIDO_FRdur

            Bvec = [thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs]
            Kp, BoutGSM   = calcKp(Bvec, DoY, thisRes.FIDOvs) 
            thisRes.FIDOKps   = Kp
            if (OSP.includeSIT) and (i in SITids):  
                thisRes.hasSheath = True
                myID = np.where(SITids == i)[0][0]
                thisRes.SITidx = np.where(thisRes.regions==0)[0]
                thisRes.SITdur = SITdata[myID, 1] #thisRes.FIDOtimes[thisRes.SITidx[-1]]-thisRes.FIDOtimes[0] 
                thisRes.SITcomp = SITdata[myID, 2]
                thisRes.SITMach = SITdata[myID, 3]
                thisRes.SITn    = SITdata[myID, 4]
                thisRes.SITvSheath = SITdata[myID, 5]
                thisRes.SITB    = SITdata[myID, 6]     
                thisRes.SITvShock = SITdata[myID,7]    
                thisRes.SITtemp = SITdata[myID,8]
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
        myOrder = ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEdelCSAx', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB', 'CMEvExp', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWT', 'SWcs', 'SWvA', 'FRBscale', 'FRtau', 'FRCnm', 'FRTscale', 'Gamma', 'IVDf1', 'IVDf2', 'CMEvTrans', 'SWBx', 'SWBy', 'SWBz', 'MHarea', 'MHdist']  
        varied = sorted(varied, key=lambda x: myOrder.index(x))      
    return ResArr

def readInData():
    dataIn = np.genfromtxt(OSP.ObsDataFile, dtype=float)
    dataIn[np.where(dataIn == -9999)] = math.nan
    
    # Need to check if goes over into new year...
    base = datetime.datetime(int(dataIn[0,0]), 1, 1, 1, 0)
    obsDTs = np.array([base + datetime.timedelta(days=int(dataIn[i,1])-1, seconds=int(dataIn[i,2]*3600)) for i in range(len(dataIn[:,0]))])
    
    nGiven = len(dataIn[0,:])
    global hasv, hasKp
    hasv, hasKp, hasT = True, True, True
    if nGiven == 11:
        # have all the things (yr doy hr, B, Bx, By, Bz, n, v, T, Kp)
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7], dataIn[:,8], dataIn[:,9], dataIn[:,10]/10.])
    elif nGiven == 10:
        # no Kp
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7], dataIn[:,8], dataIn[:,9]])
        hasKp = False
    elif nGiven ==8:
        # no v T or Kp
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7]])
        hasv, hasKp, hasT = False, False, False       
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
    maxr = ResArr[0].FCrs[-1]
    if nEns > 1:
        fakers = np.linspace(1.1,maxr+0.05,100, endpoint=True)
        splineVals = np.zeros([nEns, 100, 3])
        means = np.zeros([100, 3])
        stds  = np.zeros([100, 3])
        lims  = np.zeros([100, 2, 3])
    
        i = 0
        # Repackage profiles
        for key in ResArr.keys():
            # Fit a spline to data since may be different lengths since take different times
            #print (ResArr[key].FCrs)
            #print (ResArr[key].FClats)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClats,bc_type='natural')
            splineVals[i,:, 0] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClonsS,bc_type='natural')
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
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FClonsS, linewidth=4, color='k')
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCtilts, linewidth=4, color='k')
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_latfs, all_lonfs, all_tiltfs = [], [], []
        for key in ResArr.keys():
            all_latfs.append(ResArr[key].FClats[-1])
            all_lonfs.append(ResArr[key].FClonsS[-1])
            all_tiltfs.append(ResArr[key].FCtilts[-1])
        fitLats = norm.fit(all_latfs)
        fitLons = norm.fit(all_lonfs)
        fitTilts = norm.fit(all_tiltfs)
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(fitLats[0])+'$\pm$'+'{:4.1f}'.format(fitLats[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(fitLons[0])+'$\pm$'+'{:4.1f}'.format(fitLons[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(fitTilts[0])+'$\pm$'+'{:4.1f}'.format(fitTilts[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    else:
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClats[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClonsS[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FCtilts[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
                  
    # Labels
    axes[0].set_ylabel('Latitude ('+degree+')')
    axes[1].set_ylabel('Longitude ('+degree+')')
    axes[2].set_ylabel('Tilt ('+degree+')')
    axes[2].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([1.01,maxr+0.15])
    plt.subplots_adjust(hspace=0.1,left=0.13,right=0.95,top=0.95,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_CPA.'+figtag)
    
def makeCPAhist(ResArr):
    fig = plt.figure(constrained_layout=True, figsize=(8,8))
    gs = fig.add_gridspec(3, 3)

    ax1a = fig.add_subplot(gs[0, 0:2])
    ax1b = fig.add_subplot(gs[1, 0:2], sharex=ax1a)
    ax1c = fig.add_subplot(gs[2, 0:2], sharex=ax1a)    
    ax2a = fig.add_subplot(gs[0, 2], sharey=ax1a)
    ax2b = fig.add_subplot(gs[1, 2], sharey=ax1b, sharex=ax2a)
    ax2c = fig.add_subplot(gs[2, 2], sharey=ax1c, sharex=ax2a)
    ax1 = [ax1a, ax1b, ax1c]
    ax2 = [ax2a, ax2b, ax2c]
    
    
    maxr = ResArr[0].FCrs[-1]
    fakers = np.linspace(1.1,maxr+0.05,100, endpoint=True)
    splineVals = np.zeros([nEns, 100, 3])
    means = np.zeros([100, 3])
    stds  = np.zeros([100, 3])
    lims  = np.zeros([100, 2, 3])
    
    i = 0
    # Repackage profiles
    for key in ResArr.keys():
        # Fit a spline to data since may be different lengths since take different times
        #print (ResArr[key].FCrs)
        #print (ResArr[key].FClats)
        thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClats,bc_type='natural')
        splineVals[i,:, 0] = thefit(fakers)
        thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClonsS,bc_type='natural')
        splineVals[i,:, 1] = thefit(fakers)
        thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCtilts,bc_type='natural')
        splineVals[i,:, 2] = thefit(fakers)    
        i += 1
    for i in range(3):
        means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
        stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
        lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
        lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
        ax1[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color='LightGray') 
        ax1[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color='DarkGray')
        # Plot the histos
        ax2[i].hist(splineVals[:,-1,i], orientation='horizontal', color='LightGray', histtype='bar', ec='black')
        
    # Plot the seed profile
    ax1[0].plot(ResArr[0].FCrs, ResArr[0].FClats, linewidth=4, color='k')
    ax1[1].plot(ResArr[0].FCrs, ResArr[0].FClonsS, linewidth=4, color='k')
    ax1[2].plot(ResArr[0].FCrs, ResArr[0].FCtilts, linewidth=4, color='k')

    degree = '$^\circ$'
    
    # Add the final position as text
    all_latfs, all_lonfs, all_tiltfs = [], [], []
    for key in ResArr.keys():
        all_latfs.append(ResArr[key].FClats[-1])
        all_lonfs.append(ResArr[key].FClonsS[-1])
        all_tiltfs.append(ResArr[key].FCtilts[-1])
    fitLats = norm.fit(all_latfs)
    fitLons = norm.fit(all_lonfs)
    fitTilts = norm.fit(all_tiltfs)
    ax1[0].text(0.97, 0.05, '{:4.1f}'.format(fitLats[0])+'$\pm$'+'{:4.1f}'.format(fitLats[1])+degree, horizontalalignment='right', verticalalignment='center', transform=ax1[0].transAxes)
    ax1[1].text(0.97, 0.05, '{:4.1f}'.format(fitLons[0])+'$\pm$'+'{:4.1f}'.format(fitLons[1])+degree, horizontalalignment='right', verticalalignment='center', transform=ax1[1].transAxes)
    ax1[2].text(0.97, 0.05, '{:4.1f}'.format(fitTilts[0])+'$\pm$'+'{:4.1f}'.format(fitTilts[1])+degree, horizontalalignment='right', verticalalignment='center', transform=ax1[2].transAxes)
    
    for i in range(3):
        ax2[i].yaxis.tick_right()
    
    ax1[0].set_ylabel('Latitude ('+degree+')')
    ax1[1].set_ylabel('Longitude ('+degree+')')
    ax1[2].set_ylabel('Tilt ('+degree+')')
    ax1[2].set_xlabel('Distance (R$_S$)')
    ax1[0].set_xlim([1.01,maxr+0.15])
    ax2[2].set_xlabel('Counts')
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_CPAhist.'+figtag)

def makeADVplot(ResArr):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[1,0], axes[1,0], axes[2,0], axes[2,0], axes[1,1], axes[1,1], axes[2,1]]
    c1   = ['LightGray', 'lightblue', 'LightGray', 'LightGray', 'lightblue', 'LightGray', 'lightblue','LightGray', 'lightblue', 'LightGray']
    c2   = ['DarkGray', 'dodgerblue', 'DarkGray', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue','DarkGray', 'dodgerblue', 'DarkGray']
    maxr = ResArr[0].FCrs[-1]

    if nEns > 1:
        fakers = np.linspace(1.1,maxr+0.05,100, endpoint=True)
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
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdelCAs,bc_type='natural')
            splineVals[i,:, 2] = thefit(fakers)   
            #thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdelCSs,bc_type='natural')
            #splineVals[i,:, 3] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvFs,bc_type='natural')
            splineVals[i,:, 3] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvEs,bc_type='natural')
            splineVals[i,:, 4] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvCSrs,bc_type='natural')
            splineVals[i,:, 5] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvCSps,bc_type='natural')
            splineVals[i,:, 6] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvAxrs,bc_type='natural')
            splineVals[i,:, 7] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCvAxps,bc_type='natural')
            splineVals[i,:, 8] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCdefs,bc_type='natural')
            splineVals[i,:, 9] = thefit(fakers)
                         
            i += 1
        for i in range(10):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color=c1[i]) 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color=c2[i]) 
        
    # Plot the seed profile
    axes[0].plot(ResArr[0].FCrs, ResArr[0].FCAWs, linewidth=4, color='k', zorder=3)
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FCAWps, linewidth=4, color='b', zorder=3)
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCdelCAs, linewidth=4, color='k', zorder=3)
    #axes[3].plot(ResArr[0].FCrs, ResArr[0].FCdelCSs, linewidth=4, color='b', zorder=3)
    axes[3].plot(ResArr[0].FCrs, ResArr[0].FCvFs, linewidth=4, color='k', zorder=3)
    axes[4].plot(ResArr[0].FCrs, ResArr[0].FCvEs, linewidth=4, color='b', zorder=3)
    axes[5].plot(ResArr[0].FCrs, ResArr[0].FCvCSrs, linewidth=4, color='k', zorder=3)
    axes[6].plot(ResArr[0].FCrs, ResArr[0].FCvCSps, linewidth=4, color='b', zorder=3)
    axes[7].plot(ResArr[0].FCrs, ResArr[0].FCvAxrs, linewidth=4, color='k', zorder=3)
    axes[8].plot(ResArr[0].FCrs, ResArr[0].FCvAxps, linewidth=4, color='b', zorder=3)
    axes[9].plot(ResArr[0].FCrs, ResArr[0].FCdefs, linewidth=4, color='k', zorder=3)
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_AWs, all_AWps, all_delCAs, all_vFs, all_vEs, all_vCSrs, all_vCSps, all_vAxrs, all_vAxps, all_defs = [], [], [], [], [], [], [], [], [], []
        for key in ResArr.keys():
            all_AWs.append(ResArr[key].FCAWs[-1])
            all_AWps.append(ResArr[key].FCAWps[-1])
            all_delCAs.append(ResArr[key].FCdelCAs[-1])
            #all_delCSs.append(ResArr[key].FCdelCSs[-1])
            all_vFs.append(ResArr[key].FCvFs[-1])
            all_vEs.append(ResArr[key].FCvEs[-1])
            all_vCSrs.append(ResArr[key].FCvCSrs[-1])
            all_vCSps.append(ResArr[key].FCvCSps[-1])
            all_vAxrs.append(ResArr[key].FCvAxrs[-1])
            all_vAxps.append(ResArr[key].FCvAxps[-1])
            all_defs.append(ResArr[key].FCdefs[-1])
        fitAWs = norm.fit(all_AWs)
        fitAWps = norm.fit(all_AWps)
        fitdelCAs = norm.fit(all_delCAs)
        #fitdelCSs = norm.fit(all_delCSs)
        fitvFs = norm.fit(all_vFs)
        fitvEs = norm.fit(all_vEs)
        fitvCSrs = norm.fit(all_vCSrs)
        fitvCSps = norm.fit(all_vCSps)
        fitvAxrs = norm.fit(all_vAxrs)
        fitvAxps = norm.fit(all_vAxps)
        fitdefs = norm.fit(all_defs)
        
        
        axes[0].text(0.97, 0.15, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\pm$'+'{:4.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05,  'AW$_{\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\pm$'+'{:4.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        #axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
        axes[2].text(0.97, 0.05, '$\delta_{CA}: $' + '{:4.2f}'.format(fitdelCAs[0]) + '$\pm$'+'{:4.2f}'.format(fitdelCAs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        #axes[3].text(0.97, 0.05, '$\delta_{CS}$'+'{:4.1f}'.format(fitdelCSs[0])+'$\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[3].text(0.97, 0.15, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\pm$'+'{:4.1f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
        axes[4].text(0.97, 0.05,  'v$_E$: '+'{:4.1f}'.format(fitvEs[0])+'$\pm$'+'{:4.1f}'.format(fitvEs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes, color='b')
        axes[7].text(0.97, 0.15, 'v$_{CS,r}$: '+'{:4.1f}'.format(fitvAxrs[0])+'$\pm$'+'{:4.1f}'.format(fitvAxrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)
        axes[8].text(0.97, 0.05,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(fitvAxps[0])+'$\pm$'+'{:4.1f}'.format(fitvAxps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes, color='b')
        axes[5].text(0.97, 0.15, 'v$_{Ax,r}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\pm$'+'{:4.1f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
        axes[6].text(0.97, 0.05,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(fitvCSps[0])+'$\pm$'+'{:4.1f}'.format(fitvCSps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes, color='b')
        axes[9].text(0.97, 0.05, 'v$_{def}$: '+'{:4.1f}'.format(fitdefs[0])+'$\pm$'+'{:4.1f}'.format(fitdefs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes)              
    else:
        axes[0].text(0.97, 0.15, 'AW: '+'{:4.1f}'.format(ResArr[0].FCAWs[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05,  'AW$_{\perp}$: '+'{:4.1f}'.format(ResArr[0].FCAWps[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].text(0.97, 0.05, '$\delta_{CA}: $'+'{:4.2f}'.format(ResArr[0].FCdelCAs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        #axes[3].text(0.97, 0.05, '$\delta_{CS}$'+'{:4.1f}'.format(ResArr[0].FCdelCSs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[3].text(0.97, 0.15, 'v$_F$: '+'{:4.1f}'.format(ResArr[0].FCvFs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
        axes[4].text(0.97, 0.05,  'v$_E$: '+'{:4.1f}'.format(ResArr[0].FCvEs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes, color='b')
        axes[5].text(0.97, 0.15, 'v$_{CS,r}$: '+'{:4.1f}'.format(ResArr[0].FCvCSrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
        axes[6].text(0.97, 0.05,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(ResArr[0].FCvCSps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes, color='b')
        axes[7].text(0.97, 0.15, 'v$_{Ax,r}$: '+'{:4.1f}'.format(ResArr[0].FCvAxrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)
        axes[8].text(0.97, 0.05,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(ResArr[0].FCvAxps[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes, color='b')
        axes[9].text(0.97, 0.05, 'v$_{def}$: '+'{:4.1f}'.format(ResArr[0].FCdefs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes)
                  
    # Labels
    axes[0].set_ylabel('AW, AW$_{\perp}$ ('+degree+')')
    axes[2].set_ylabel('$\delta_{CA}$')
    axes[3].set_ylabel('v$_F$, v$_E$ (km/s)')
    axes[5].set_ylabel('v$_{Ax,r}$, v$_{Ax,\perp}$ (km/s)')
    axes[7].set_ylabel('v$_{CS,r}$, v$_{CS,\perp}$ (km/s)')
    axes[9].set_ylabel('v$_{def}$ (km/s)')
    axes[5].set_xlabel('Distance (R$_S$)')
    axes[9].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([1.01,maxr+0.15])
    plt.subplots_adjust(hspace=0.1,left=0.08,right=0.95,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ADV.'+figtag)
    
def makeDragless(ResArr):
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(10,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[1,0], axes[1,0], axes[1,1], axes[1,1]]
    c1   = ['LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue']
    c2   = ['DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue']
    # this isn't the exact end for all cases but don't really care in this figure
    # since more showing trend with distance and it flattens
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # get number of impacts, may be less than nEns
    nImp = 0
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            nImp += 1
            hits.append(i)
            
    
    # Arrays to hold spline results
    fakers = np.linspace(rStart,rEnd,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, 8])
    means = np.zeros([100, 8])
    stds  = np.zeros([100, 8])
    lims  = np.zeros([100, 2, 8])
    
    axes[7] = axes[6].twinx()
    
    
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
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTvCSrs,bc_type='natural')
            splineVals[i,:, 5] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTBtors,bc_type='natural')
            splineVals[i,:, 6] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTlogTs,bc_type='natural')
            splineVals[i,:, 7] = thefit(fakers)                         
            i += 1
        for i in range(8):
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
    axes[5].plot(ResArr[0].ANTrs, ResArr[0].ANTvCSrs, linewidth=4, color='b', zorder=3)
    axes[6].plot(ResArr[0].ANTrs, ResArr[0].ANTBtors, linewidth=4, color='k', zorder=3)
    axes[7].plot(ResArr[0].ANTrs, ResArr[0].ANTlogTs, linewidth=4, color='b', zorder=3)    
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_AWs, all_AWps, all_delAxs, all_delCSs, all_vFs, all_vCSrs, all_Btors, all_Ts = [], [], [], [], [], [], [], []
        for key in hits:
            all_AWs.append(ResArr[key].ANTAWs[-1])
            all_AWps.append(ResArr[key].ANTAWps[-1])
            all_delAxs.append(ResArr[key].ANTdelAxs[-1])
            all_delCSs.append(ResArr[key].ANTdelCSs[-1])
            all_vFs.append(ResArr[key].ANTvFs[-1])
            all_vCSrs.append(ResArr[key].ANTvCSrs[-1])
            all_Btors.append(ResArr[key].ANTBtors[-1])
            all_Ts.append(ResArr[key].ANTlogTs[-1])
        fitAWs = norm.fit(all_AWs)
        fitAWps = norm.fit(all_AWps)
        fitdelAxs = norm.fit(all_delAxs)
        fitdelCSs = norm.fit(all_delCSs)
        fitvFs = norm.fit(all_vFs)
        fitvCSrs = norm.fit(all_vCSrs)
        fitBtors = norm.fit(all_Btors)
        fitTs    = norm.fit(all_Ts)
        
        
        axes[0].text(0.97, 0.96, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\pm$'+'{:2.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.9,  'AW$_{\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\pm$'+'{:2.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
        axes[2].text(0.97, 0.96, '$\delta_{Ax}$: '+'{:4.2f}'.format(fitdelAxs[0])+'$\pm$'+'{:4.2f}'.format(fitdelAxs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.9, '$\delta_{CS}$: '+'{:4.2f}'.format(fitdelCSs[0])+'$\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.96, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\pm$'+'{:2.0f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.9, 'v$_{Exp}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\pm$'+'{:2.0f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.96, 'B: '+'{:4.1f}'.format(fitBtors[0])+'$\pm$'+'{:3.1f}'.format(fitBtors[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.9,  'log(T): '+'{:4.1f}'.format(fitTs[0])+'$\pm$'+'{:3.1f}'.format(fitTs[1])+' K', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')        
    else:
        axes[0].text(0.97, 0.96, 'AW: '+'{:4.1f}'.format(ResArr[0].ANTAWs[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.9,  'AW$_{\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTAWps[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].text(0.97, 0.96, '$\delta_{Ax}$: '+'{:4.2f}'.format(ResArr[0].ANTdelAxs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.9, '$\delta_{CS}$: '+'{:4.2f}'.format(ResArr[0].ANTdelCSs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.96, 'v$_F$: '+'{:4.0f}'.format(ResArr[0].ANTvFs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.9, 'v$_{Exp}$: '+'{:4.0f}'.format(ResArr[0].ANTvCSrs[-1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.96, 'B$_{t}$: '+'{:4.1f}'.format(ResArr[0].ANTBtors[-1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.9,  'log(T): '+'{:4.1f}'.format(ResArr[0].ANTlogTs[-1])+' K', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')        

                  
    # Labels
    axes[0].set_ylabel('AW, AW$_{\perp}$ ('+degree+')')
    axes[2].set_ylabel('$\delta_{Ax}$, $\delta_{CS}$')
    axes[4].set_ylabel('v$_F$, v$_{Exp}$ (km/s)')
    axes[6].set_ylabel('B (nT)')
    axes[7].set_ylabel('log(T) (K)')
    axes[7].set_ylim([3,6.5])
    axes[6].set_ylim([1,5e4])
    axes[4].set_xlabel('Distance (R$_S$)')
    axes[6].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([rStart, rEnd])
    axes[6].set_yscale('log')
    plt.subplots_adjust(hspace=0.1,left=0.08,right=0.93,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_DragLess.'+figtag)

def makePUPplot(ResArr):
    fig, axes = plt.subplots(2, 4, sharex=True, figsize=(16,8))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[0,2], axes[0,2], axes[0,3], axes[0,3], axes[1,0], axes[1,0], axes[1,0], axes[1,1], axes[1,1], axes[1,2],  axes[1,2], axes[1,3], axes[1,3]]
    c1 = '#882255'
    c2 = '#332288'
    
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # get number of impacts, may be less than nEns
    nImp = 0
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            nImp += 1
            hits.append(i)
            
    # Arrays to hold spline results
    nParams = 17
    fakers = np.linspace(rStart,rEnd,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, nParams])
    means = np.zeros([100, nParams])
    stds  = np.zeros([100, nParams])
    lims  = np.zeros([100, 2, nParams])
    
    #axes[11] = axes[10].twinx()   
    i = 0     
    for key in hits:
        thisRes = ResArr[key]
        thexs = [thisRes.ANTrs+thisRes.PUPwids,thisRes.ANTrs,  thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, thisRes.ANTrs, ]
        theParams = [thisRes.ANTtimes*24, thisRes.ANTtimes*24, thisRes.ANTAWs, thisRes.ANTAWps, thisRes.ANTdelAxs, thisRes.ANTdelCSs, thisRes.PUPBs, thisRes.ANTBtors,  thisRes.PUPvshocks, thisRes.ANTvFs, thisRes.ANTvCSrs, thisRes.PUPwids, thisRes.ANTCMEwids, thisRes.PUPns, thisRes.ANTns, np.power(10,thisRes.PUPlogTs), np.power(10,thisRes.ANTlogTs)]
        
        if nEns > 1:
            # Fit a spline to data since may be different lengths since take different times
            for k in range(nParams):
                thefit = CubicSpline(thexs[k], theParams[k],bc_type='natural')
                splineVals[i,:, k] = thefit(fakers)
            i += 1
    
    for i in range(nParams):
        col = c1
        if i == 10: 
            col = 'k'
        else:
            calci = i
            if i > 10:
                calci = i - 1
            if calci % 2 == 0: col = c2
        
        if nEns > 1:
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
        
            rmidx = 3
            axes[i].fill_between(fakers[:-rmidx], lims[:-rmidx,0,i], lims[:-rmidx,1,i], color=col, alpha=0.25) 
            axes[i].fill_between(fakers[:-rmidx], means[:-rmidx,i]+stds[:-rmidx,i], means[:-rmidx,i]-stds[:-rmidx,i], color=col, alpha=0.25)
        
        axes[i].plot(thexs[i],theParams[i], linewidth=4, color=col, zorder=3)
        
    # Add the final position as text
    labels = ['AT$_{Sh}$', 'AT$_{CME}$', 'AW', 'AW$_{\perp}$', '$\delta_{Ax}$', '$\delta_{CS}$', 'B$_{sh}$', 'B$_{CME}$', 'v$_{Sh}$', 'v$_{CME}$', 'v$_{Exp}$', 'Wid$_{sh}$', 'Wid$_{CME}$', 'n$_{sh}$', 'n$_{CME}$', 'log(T$_{Sh}$)', 'log(T$_{CME}$)']
    deg = '$^\circ$'
    units = ['hr', 'hr', deg, deg, '', '', 'nT', 'nT', 'km/s', 'km/s', 'km/s', 'R$_S$', 'R$_S$', 'cm$^{-3}$', 'cm$^{-3}$', 'K', 'K']
    
    
    if nEns > 1:
        all_Params = [[] for i in range(nParams)]
        for key in hits:
            thisRes = ResArr[key]
            theParams = [thisRes.ANTtimes*24, thisRes.ANTtimes*24, thisRes.ANTAWs, thisRes.ANTAWps, thisRes.ANTdelAxs, thisRes.ANTdelCSs, thisRes.PUPBs, thisRes.ANTBtors,  thisRes.PUPvshocks, thisRes.ANTvFs, thisRes.ANTvCSrs, thisRes.PUPwids, thisRes.ANTCMEwids, thisRes.PUPns, thisRes.ANTns, thisRes.PUPlogTs, thisRes.ANTlogTs]
        
            for i in range(nParams):
                if i == 0:
                    all_Params[i].append(theParams[i][thisRes.ANTshidx]) 
                elif i == 1:
                    all_Params[i].append(theParams[i][thisRes.ANTFRidx]) 
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
            axes[i].text(0.97, ytext, labels[i]+': '+'{:4.1f}'.format(endMean)+'$\pm$'+'{:.2f}'.format(endSTD)+' '+units[i], transform=axes[i].transAxes, color=col, horizontalalignment='right', verticalalignment='center')
    
    
    ylabels = ['Time (hr)', 'AW ('+deg+')', '$\delta$', 'B (nT)', 'v (km/s)', 'Width (R$_S$)', 'n (cm$^{-3}$)', 'log(T) (T)']

    for i in range(8):
        axes[2*i+1].set_ylabel(ylabels[i])
        if i > 3:
            axes[2*i+1].set_xlabel('R (au)')
    
    for i in [7,13,15]:
        axes[i].set_yscale('log')
    axes[0].set_xlim([rStart, rEnd])
    plt.subplots_adjust(wspace=0.3, hspace=0.01,left=0.06,right=0.99,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANTPUP.'+figtag)

def makeDragplot(ResArr):
    fig, axes = plt.subplots(3, 2, sharex=True, figsize=(14,10))
    axes = [axes[0,0], axes[0,0], axes[0,1], axes[0,1], axes[1,0], axes[1,0], axes[1,1], axes[1,1], axes[2,0], axes[2,0], axes[2,1], axes[2,1], axes[2,1]]
    c1   = ['LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue', 'LightGray', 'lightblue','LightGray', 'lightblue', 'LightGray', 'lightblue', 'Pink']
    c2   = ['DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue','DarkGray', 'dodgerblue', 'DarkGray', 'dodgerblue','Tomato']
    # this isn't the exact end for all cases but don't really care in this figure
    # since more showing trend with distance and it flattens
    rStart = ResArr[0].ANTrs[0]
    rEnd = ResArr[0].ANTrs[-1]
    
    # get number of impacts, may be less than nEns
    nImp = 0
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            nImp += 1
            hits.append(i)
            
    
    # Arrays to hold spline results
    fakers = np.linspace(rStart,rEnd,100, endpoint=True)
    splineVals = np.zeros([nImp, 100, 13])
    means = np.zeros([100, 13])
    stds  = np.zeros([100, 13])
    lims  = np.zeros([100, 2, 13])
    
    axes[12] = axes[11].twinx()
    
    
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
            thefit = CubicSpline(ResArr[key].ANTrs,ResArr[key].ANTlogTs,bc_type='natural')
            splineVals[i,:, 12] = thefit(fakers)
                         
            i += 1
        for i in range(13):
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
    axes[12].plot(ResArr[0].ANTrs, ResArr[0].ANTlogTs, linewidth=4, color='r', zorder=3)
    
    degree = '$^\circ$'
    
    # Add the final position as text
    if nEns > 1:
        all_AWs, all_AWps, all_delAxs, all_delCSs, all_vFs, all_vEs, all_vCSrs, all_vCSps, all_vAxrs, all_vAxps, all_Btors, all_Bpols, all_Ts = [], [], [], [], [], [], [], [], [], [], [], [], []
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
            all_Ts.append(ResArr[key].ANTlogTs[-1])
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
        fitTs    = norm.fit(all_Ts)
        
        
        axes[0].text(0.97, 0.95, 'AW: '+'{:4.1f}'.format(fitAWs[0])+'$\pm$'+'{:4.1f}'.format(fitAWs[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05,  'AW$_{\perp}$: '+'{:4.1f}'.format(fitAWps[0])+'$\pm$'+'{:4.1f}'.format(fitAWps[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].set_ylim(fitdelAxs[0]-fitdelAxs[1]-0.1, 1.05)
        axes[2].text(0.97, 0.95, '$\delta_{Ax}$: '+'{:4.2f}'.format(fitdelAxs[0])+'$\pm$'+'{:4.2f}'.format(fitdelAxs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '$\delta_{CS}$: '+'{:4.2f}'.format(fitdelCSs[0])+'$\pm$'+'{:4.2f}'.format(fitdelCSs[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
        axes[4].text(0.97, 0.95, 'v$_F$: '+'{:4.1f}'.format(fitvFs[0])+'$\pm$'+'{:4.1f}'.format(fitvFs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
        axes[5].text(0.97, 0.85,  'v$_E$: '+'{:4.1f}'.format(fitvEs[0])+'$\pm$'+'{:4.1f}'.format(fitvEs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes, color='b')
        axes[6].text(0.97, 0.95, 'v$_{CS,r}$: '+'{:4.1f}'.format(fitvCSrs[0])+'$\pm$'+'{:4.1f}'.format(fitvCSrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
        axes[7].text(0.97, 0.85,  'v$_{CS,\perp}$: '+'{:4.1f}'.format(fitvCSps[0])+'$\pm$'+'{:4.1f}'.format(fitvCSps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes, color='b')
        axes[8].text(0.97, 0.95, 'v$_{Ax,r}$: '+'{:4.1f}'.format(fitvAxrs[0])+'$\pm$'+'{:4.1f}'.format(fitvAxrs[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)
        axes[9].text(0.97, 0.85,  'v$_{Ax,\perp}$: '+'{:4.1f}'.format(fitvAxps[0])+'$\pm$'+'{:4.1f}'.format(fitvAxps[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[9].transAxes, color='b')
        axes[10].text(0.97, 0.95, 'B$_{t}$: '+'{:4.1f}'.format(fitBtors[0])+'$\pm$'+'{:4.1f}'.format(fitBtors[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[10].transAxes)
        axes[11].text(0.97, 0.85,  'B$_{p}$: '+'{:4.1f}'.format(fitBpols[0])+'$\pm$'+'{:4.1f}'.format(fitBpols[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[11].transAxes, color='b')
        axes[11].text(0.97, 0.75,  'log(T): '+'{:4.1f}'.format(fitTs[0])+'$\pm$'+'{:4.1f}'.format(fitTs[1])+' K', horizontalalignment='right', verticalalignment='center', transform=axes[11].transAxes, color='r')
    else:
        axes[0].text(0.97, 0.95, 'AW: '+'{:4.1f}'.format(ResArr[0].ANTAWs[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.85,  'AW$_{\perp}$: '+'{:4.1f}'.format(ResArr[0].ANTAWps[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes, color='b')
        axes[2].text(0.97, 0.95, '$\delta_{Ax}$: '+'{:4.1f}'.format(ResArr[0].ANTdelAxs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        axes[3].text(0.97, 0.85, '$\delta_{CS}$: '+'{:4.1f}'.format(ResArr[0].ANTdelCSs[-1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes, color='b')
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
    axes[2].set_ylabel('$\delta_{Ax}$, $\delta_{CS}$')
    axes[4].set_ylabel('v$_F$, v$_E$ (km/s)')
    axes[8].set_ylabel('v$_{Ax,r}$, v$_{Ax,\perp}$ (km/s)')
    axes[6].set_ylabel('v$_{CS,r}$, v$_{CS,\perp}$ (km/s)')
    axes[10].set_ylabel('B$_t$, B$_p$ (nT)')
    axes[12].set_ylabel('log(T) (K))')
    axes[12].set_ylim([3,6.5])
    axes[10].set_ylim([1,5e4])
    axes[8].set_xlabel('Distance (R$_S$)')
    axes[10].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([rStart, rEnd])
    axes[10].set_yscale('log')
    plt.subplots_adjust(hspace=0.1,left=0.08,right=0.95,top=0.98,bottom=0.1)
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Drag.'+figtag)

def makeAThisto(ResArr):
    fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True)
    axes[0,0].set_ylabel('Counts')
    axes[1,0].set_ylabel('Counts')
    axes[2,0].set_ylabel('Counts')
    axes = [axes[1,1], axes[1,2], axes[0,0], axes[2,0], axes[2,1], axes[0,1], axes[1,0], axes[0,2], axes[2,2]]
    all_vFs, all_vExps, all_TTs, all_durs, all_Bfs, all_Bms, all_ns, all_Kps, all_Ts, FC_times  = [], [], [], [], [], [], [], [], [], []
    # Collect the ensemble results
    for key in ResArr.keys(): 
        # figure out when hits FR, may not be last pt if doing internal FIDO
        thisidx = ResArr[key].ANTFRidx
        if (not ResArr[key].miss) and (not ResArr[key].fail):
            all_vFs.append(ResArr[key].ANTvFs[thisidx])
            all_vExps.append(ResArr[key].ANTvCSrs[thisidx])
            all_TTs.append(ResArr[key].ANTtimes[thisidx])    
            all_durs.append(ResArr[key].ANTdur)
            all_Bfs.append(ResArr[key].ANTBpols[thisidx])
            all_Bms.append(ResArr[key].ANTBtors[thisidx])
            all_ns.append(ResArr[key].ANTns[thisidx])
            all_Kps.append(ResArr[key].ANTKp0)
            all_Ts.append(ResArr[key].ANTlogTs[thisidx])
    # Ordered Data
    ordData = [all_vFs, all_vExps, all_TTs, all_Bfs, all_Bms, all_durs, all_Ts, all_ns, all_Kps] 
    names = ['v$_F$ (km/s)', 'v$_{Exp}$ (km/s)', 'Transit Time (days)', 'B$_F$ (nT)', 'B$_C$ (nT)', 'Duration (hours)', 'log$_{10}$T (K)','n (cm$^{-3}$)', 'Kp']
    units = ['km/s', 'km/s', 'days', 'nT', 'nT', 'hr', 'log$_{10}$ K','cm$^{-3}$', '']
    fmts = ['{:.0f}','{:.0f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}','{:.1f}']
    #fmtsB = ['{:.0f}','{:.0f}','{:4.2f}','{:4.2f}','{:4.1f}','{:4.1f}','{:4.1f}','{:4.2f}','{:4.2f}']
    maxcount = 0
    for i in range(9):
        theseData = np.array(ordData[i])
        mean, std = np.mean(theseData), np.std(theseData)
        cutoff = 5 *std
        if i in [3,4]: cutoff = 3 * std
        newData = theseData[np.where(np.abs(theseData - mean) < cutoff)[0]]
        n, bins, patches = axes[i].hist(newData, bins=10, color='#882255', histtype='bar', ec='black')
        axes[i].set_xlabel(names[i])
        maxn = np.max(n)
        if maxn > maxcount: maxcount = maxn
        if i != 2:
            axes[i].text(0.97, 0.92, fmts[i].format(mean)+'$\pm$'+fmts[i].format(std)+ ' '+units[i], horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 
        else:
            if not OSP.noDate:
                base = datetime.datetime(yr, 1, 1, 0, 0)
                date = base + datetime.timedelta(days=(DoY+mean))   
                dateLabel = date.strftime('%b %d %H:%M')
                axes[i].text(0.97, 0.92, dateLabel+'$\pm$'+'{:.1f}'.format(std*12)+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 
            axes[i].text(0.97, 0.82, fmts[i].format(mean)+'$\pm$'+'{:.2f}'.format(std)+' days', horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes) 

                
                       
    for i in range(9): axes[i].set_ylim(0, maxcount*1.2)
        
    plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ANT.'+figtag)
     
def makeISplot(ResArr, SWpad=18):
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8,12))
    mindate = None
    maxdate = None
    for key in ResArr.keys():
        if key == 0:
            lw, co, zord = 4, 'b', 11
        else:
            lw, co, zord = 2, 'DarkGray', 2
            
        if ResArr[key].FIDOtimes is not None: #not ResArr[key].miss:
            if OSP.noDate:
                dates = ResArr[key].FIDOtimes
            else:
                base = datetime.datetime(yr, 1, 1, 0, 0)
                if not OSP.doANT:
                    dates = np.array([base + datetime.timedelta(days=(i-1)) for i in ResArr[key].FIDOtimes])
                else:
                    dates = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[key].FIDOtimes])
            # plot the flux rope
            nowIdx = ResArr[key].FIDO_FRidx
            axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[nowIdx], linewidth=lw, color=co, zorder=zord)
            axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[nowIdx], linewidth=lw, color=co, zorder=zord)
            axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[nowIdx], linewidth=lw, color=co, zorder=zord)
            axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[nowIdx], linewidth=lw, color=co, zorder=zord)
            axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[nowIdx], linewidth=lw, color=co, zorder=zord)
            axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[nowIdx], linewidth=lw, color=co, zorder=zord)
            if OSP.isSat:
                axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[nowIdx], linewidth=lw, color=co, zorder=zord)
            else:
                axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[nowIdx], linewidth=lw, color=co, zorder=zord)
            if mindate is None: 
                mindate = dates[nowIdx[0]]
                maxdate = dates[nowIdx[-1]]

            # plot the sheath (and the FR start so connected)
            if len(ResArr[key].FIDO_shidx) != 0:
                nowIdx = ResArr[key].FIDO_shidx
                nowIdx = np.append(nowIdx, ResArr[key].FIDO_FRidx[0])
                axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                if OSP.isSat:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                else:
                    axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[nowIdx], '--', linewidth=lw, color=co, zorder=zord)
                #axes[4].plot(dates[nowIdx], ResArr[key].FIDOKps[nowIdx], '--', linewidth=2, color='DarkGray')
            
            # plot SW outside of sh+FR
            if len(ResArr[key].FIDO_SWidx) > 0:
                if len(ResArr[key].FIDO_shidx) != 0:
                    frontEnd, backStart = dates[ResArr[key].FIDO_shidx[0]], dates[ResArr[key].FIDO_FRidx[-1]]
                    frontStart, backEnd = frontEnd-datetime.timedelta(hours=12), backStart+datetime.timedelta(hours=12)
                    frontIdx = np.where((dates>=frontStart) & (dates <=frontEnd))[0]
                    backIdx = np.where((dates>=backStart) & (dates <=backEnd))[0]
                    for nowIdx in [frontIdx, backIdx]:
                        axes[0].plot(dates[nowIdx], ResArr[key].FIDOBs[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        axes[1].plot(dates[nowIdx], ResArr[key].FIDOBxs[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        axes[2].plot(dates[nowIdx], ResArr[key].FIDOBys[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        axes[3].plot(dates[nowIdx], ResArr[key].FIDOBzs[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        axes[4].plot(dates[nowIdx], ResArr[key].FIDOvs[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        axes[5].plot(dates[nowIdx], ResArr[key].FIDOtems[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        if OSP.isSat:
                            axes[6].plot(dates[nowIdx], ResArr[key].FIDOns[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                        else:
                            axes[6].plot(dates[nowIdx], ResArr[key].FIDOKps[nowIdx], ':', linewidth=lw, color=co, zorder=zord)
                
                
                
            if len(ResArr[key].FIDO_SWidx) > 0:    
                if dates[frontIdx[0]] < mindate: mindate = dates[frontIdx[0]]
                if dates[backIdx[-1]] > maxdate: maxdate = dates[backIdx[-1]]  
            else:
                # will be either sheath or FR as appropriate
                if dates[nowIdx[0]] < mindate: mindate = dates[nowIdx[0]]
                if dates[nowIdx[-1]] > maxdate: maxdate = dates[nowIdx[-1]]
                    
    
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('v (km/s)')
    axes[5].set_ylabel('T (K)')
    if OSP.isSat:
        axes[6].set_ylabel('n (cm$^{-3}$)')
    else:
        axes[6].set_ylabel('Kp')
    
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
    
        # take out ticks if too many
        for i in range(5):
            yticks = axes[i].yaxis.get_major_ticks()
            if len(yticks) > 6:
                ticks2hide = np.array(range(len(yticks)-1))[::2]
                for j in ticks2hide:
                    yticks[j].label1.set_visible(False)
    
    if ObsData is not None:
        axes[0].plot(ObsData[0,:], ObsData[1,:], linewidth=4, color='r')
        axes[1].plot(ObsData[0,:], ObsData[2,:], linewidth=4, color='r')
        axes[2].plot(ObsData[0,:], ObsData[3,:], linewidth=4, color='r')
        axes[3].plot(ObsData[0,:], ObsData[4,:], linewidth=4, color='r')
        axes[4].plot(ObsData[0,:], ObsData[6,:], linewidth=4, color='r')
        axes[5].plot(ObsData[0,:], ObsData[7,:], linewidth=4, color='r')    
        if OSP.isSat:
            axes[6].plot(ObsData[0,:], ObsData[5,:], linewidth=4, color='r')
        elif hasKp:
            axes[6].plot(ObsData[0,:], ObsData[8,:], linewidth=4, color='r')

    
    if not OSP.noDate: fig.autofmt_xdate()
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.15)
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_IS.'+figtag)    
    
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
            all_dur.append(ResArr[key].FIDO_FRdur)
            all_Bz.append(np.min(ResArr[key].FIDOBzs))
            all_Kp.append(np.max(ResArr[key].FIDOKps))
            all_B.append(np.max(ResArr[key].FIDOBs))
            all_vF.append(ResArr[key].FIDOvs[ResArr[key].FIDO_FRidx[0]])
            all_vE.append(ResArr[key].FIDO_FRexp )
            
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color='c', histtype='bar', ec='black')
    n2, bins, patches = axes[1].hist(all_Bz, bins=10, color='c', histtype='bar', ec='black')
    n3, bins, patches = axes[2].hist(all_B, bins=10, color='c', histtype='bar', ec='black')
    n4, bins, patches = axes[3].hist(all_Kp, bins=10, color='c', histtype='bar', ec='black')
    n5, bins, patches = axes[4].hist(all_vF, bins=10, color='c', histtype='bar', ec='black')
    n6, bins, patches = axes[5].hist(all_vE, bins=10, color='c', histtype='bar', ec='black')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6)])
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitDur = norm.fit(all_dur)
    fitBz = norm.fit(all_Bz)
    fitB  = norm.fit(all_B)
    fitKp = norm.fit(all_Kp)
    fitvF  = norm.fit(all_vF)
    fitvE  = norm.fit(all_vE)
    axes[0].text(0.97, 0.95, '{:4.1f}'.format(fitDur[0])+'$\pm$'+'{:4.1f}'.format(fitDur[1])+' hours', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.95, '{:4.1f}'.format(fitBz[0])+'$\pm$'+'{:4.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.95, '{:4.1f}'.format(fitB[0])+'$\pm$'+'{:4.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.95, '{:4.1f}'.format(fitKp[0])+'$\pm$'+'{:4.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.95, '{:4.1f}'.format(fitvF[0])+'$\pm$'+'{:4.1f}'.format(fitvF[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)
    axes[5].text(0.97, 0.95, '{:4.1f}'.format(fitvE[0])+'$\pm$'+'{:4.1f}'.format(fitvE[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    
    # Labels
    axes[0].set_xlabel('Duration (hours)')
    axes[1].set_xlabel('Minimum B$_z$ (nT)')
    axes[2].set_xlabel('Maximum B (nT)')
    axes[3].set_xlabel('Maximum Kp')
    axes[4].set_xlabel('v$_F$ (km/s)')
    axes[5].set_xlabel('v$_{Exp}$ (km/s)')
    for i in range(6): axes[i].set_ylabel('Counts')    
    
    plt.subplots_adjust(wspace=0.15, hspace=0.25,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_FIDOhist.'+figtag)

def makeSIThistos(ResArr):
    fig, axes = plt.subplots(3, 3, figsize=(10,10), sharey=True)
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
    n1, bins, patches = axes[0].hist(all_dur, bins=10, color='b', histtype='bar', ec='black')
    n2, bins, patches = axes[1].hist(all_comp, bins=10, color='b', histtype='bar', ec='black')
    n3, bins, patches = axes[2].hist(all_n, bins=10, color='b', histtype='bar', ec='black')
    n4, bins, patches = axes[3].hist(all_vShock, bins=10, color='b', histtype='bar', ec='black')
    n5, bins, patches = axes[4].hist(all_vSheath, bins=10, color='b', histtype='bar', ec='black')
    n6, bins, patches = axes[5].hist(all_Mach, bins=10, color='b', histtype='bar', ec='black')
    n7, bins, patches = axes[6].hist(all_B, bins=10, color='b', histtype='bar', ec='black')
    # Bz might be peaked at 0 if has no neg values
    n8, bins, patches = axes[7].hist(all_Bz, color='b', histtype='bar', ec='black')
    n9, bins, patches = axes[8].hist(all_Kp, color='b', histtype='bar', ec='black')
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
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_SIThist.'+figtag)

def makeallIShistos(ResArr):
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
        
    # Collect the ensemble results
    for key in ResArr.keys(): 
        if ResArr[key].hasSheath:
            if (not ResArr[key].miss) and (not ResArr[key].fail):
                all_AT.append(ResArr[key].FIDOtimes[ResArr[key].FIDO_shidx[0]])
                all_durS.append(ResArr[key].SITdur)
                all_dur.append(ResArr[key].FIDO_FRdur)
                all_vS.append(ResArr[key].SITvSheath)
                all_vF.append(ResArr[key].FIDOvs[ResArr[key].FIDO_FRidx[0]])
                all_vE.append(ResArr[key].FIDO_FRexp)
                all_Bz.append(np.min(ResArr[key].FIDOBzs))
                all_B.append(np.max(ResArr[key].FIDOBs))
                all_Kp.append(np.max(ResArr[key].FIDOKps))
                       
    # Determine the maximum bin height so we can add extra padding for the 
    # mean and uncertainty
    n1, bins, patches = axes[0].hist(all_AT, bins=10, color='b', histtype='bar', ec='black')
    n2, bins, patches = axes[1].hist(all_durS, bins=10, color='b', histtype='bar', ec='black')
    n3, bins, patches = axes[2].hist(all_dur, bins=10, color='b', histtype='bar', ec='black')
    n4, bins, patches = axes[3].hist(all_vS, bins=10, color='b', histtype='bar', ec='black')
    n5, bins, patches = axes[4].hist(all_vF, bins=10, color='b', histtype='bar', ec='black')
    n6, bins, patches = axes[5].hist(all_vE, bins=10, color='b', histtype='bar', ec='black')
    n7, bins, patches = axes[6].hist(all_B, bins=10, color='b', histtype='bar', ec='black')
    # Bz might be peaked at 0 if has no neg values
    n8, bins, patches = axes[7].hist(all_Bz, color='b', histtype='bar', ec='black')
    n9, bins, patches = axes[8].hist(all_Kp, color='b', histtype='bar', ec='black')
    maxcount = np.max([np.max(n1), np.max(n2), np.max(n3), np.max(n4), np.max(n5), np.max(n6), np.max(n7), np.max(n8), np.max(n9)])
    axes[0].set_ylim(0, maxcount*1.1)
    
    # Add the mean and sigma from a normal fit
    fitAT = norm.fit(all_AT)
    fitDurS = norm.fit(all_durS)
    fitDur = norm.fit(all_dur)
    fitvS = norm.fit(all_vS)
    fitvF = norm.fit(all_vF)
    fitvE = norm.fit(all_vE)
    fitB = norm.fit(all_B)
    fitBz = norm.fit(all_Bz)
    fitKp = norm.fit(all_Kp)
    
    if not OSP.noDate:
        base = datetime.datetime(yr, 1, 1, 0, 0)
        # add in FC time (if desired)
        date = base+datetime.timedelta(days=(fitAT[0]+DoY))
        dateLabel = date.strftime('%b %d %H:%M')
        axes[0].text(0.97, 0.92, dateLabel+'$\pm$'+'{:.1f}'.format(fitAT[1]*24)+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes) 
    axes[0].text(0.97, 0.82, '{:.2f}'.format(fitAT[0])+'$\pm$'+'{:.2f}'.format(fitAT[1]) + ' days', horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)

    #axes[0].text(0.97, 0.95, '{:4.1f}'.format(fitAT[0])+'$\pm$'+'{:4.1f}'.format(fitAT[1]), horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
    axes[1].text(0.97, 0.92, '{:.1f}'.format(fitDurS[0])+'$\pm$'+'{:.1f}'.format(fitDurS[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
    axes[2].text(0.97, 0.92, '{:.1f}'.format(fitDur[0])+'$\pm$'+'{:.1f}'.format(fitDur[1])+' hr', horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
    axes[3].text(0.97, 0.92, '{:.0f}'.format(fitvS[0])+'$\pm$'+'{:.0f}'.format(fitvS[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[3].transAxes)
    axes[4].text(0.97, 0.92, '{:.0f}'.format(fitvF[0])+'$\pm$'+'{:.0f}'.format(fitvF[1])+' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[4].transAxes)    
    axes[5].text(0.97, 0.92, '{:.0f}'.format(fitvE[0])+'$\pm$'+'{:.0f}'.format(fitvE[1])+ ' km/s', horizontalalignment='right', verticalalignment='center', transform=axes[5].transAxes)
    axes[6].text(0.97, 0.92, '{:.1f}'.format(fitB[0])+'$\pm$'+'{:.1f}'.format(fitB[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[6].transAxes)
    axes[7].text(0.97, 0.92, '{:.1f}'.format(fitBz[0])+'$\pm$'+'{:.1f}'.format(fitBz[1])+' nT', horizontalalignment='right', verticalalignment='center', transform=axes[7].transAxes)
    axes[8].text(0.97, 0.92, '{:.1f}'.format(fitKp[0])+'$\pm$'+'{:.1f}'.format(fitKp[1]), horizontalalignment='right', verticalalignment='center', transform=axes[8].transAxes)    
    
    # Labels
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
    
    plt.subplots_adjust(wspace=0.15, hspace=0.3,left=0.12,right=0.95,top=0.95,bottom=0.1)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_allIShist.'+figtag)
    
def makeEnsplot(ResArr, critCorr=0.5):
    deg = '('+'$^\circ$'+')'

    out2outLab = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\perp}$\n'+deg, 'CMEdelAx':'$\delta_{Ax}$', 'CMEdelCS':'$\delta_{CS}$', 'CMEvF':'v$_{F}$\n(km/s)', 'CMEvExp':'v$_{Exp}$\n(km/s)', 'TT':'Transit\nTime\n(days)', 'Dur':'Dur\n(hours)', 'n':'n\n(cm$^{-3}$)',  'B':'max B (nT)', 'Bz':'min Bz\n(nT)', 'Kp':'max Kp', 'logT':'log$_{10}$T\n(K)'}
    
    myLabs = {'CMElat':'Lat\n'+deg, 'CMElon':'Lon\n'+deg, 'CMEtilt':'Tilt\n'+deg, 'CMEvr':'v$_F$\n(km/s)', 'CMEAW':'AW\n'+deg, 'CMEAWp':'AW$_{\perp}$\n'+deg, 'CMEdelAx':'$\delta_{Ax}$', 'CMEdelCS':'$\delta_{CS}$', 'CMEdelCSAx':'$\delta_{CA}$', 'CMEr':'R$_{F0}$ (R$_S$)', 'FCrmax':'FC end R$_{F0}$\n (R$_S$)', 'FCraccel1':'FC R$_{v1}$\n (km/s)', 'FCraccel2':'FC R$_{v2}$\n (km/s)', 'FCvrmin':'FC v$_{0}$\n (km/s)', 'FCAWmin':'FC AW$_{0}$\n'+deg, 'FCAWr':'FC R$_{AW}$\n (R$_S$)', 'CMEM':'M$_{CME}$\n(10$^{15}$ g)', 'FCrmaxM':'FC R$_{M}$\n(R$_S$)', 'FRB':'B$_0$ (nT)', 'CMEvExp':'v$_{Exp}$\n (km/s)', 'SWCd': 'C$_d$', 'SWCdp':'C$_{d,\perp}$', 'SWn':'n$_{SW}$\n(cm$^{-3}$)', 'SWv':'v$_{SW}$\n(km/s)', 'SWB':'B$_{SW}$\n(nT)', 'SWT':'T$_{SW}$\n(K)', 'SWcs':'c$_s$\n(km/s)', 'SWvA':'v$_A$\n(km/s)', 'FRBscale':'B scale', 'FRtau':'$\\tau', 'FRCnm':'C$_{nm}$', 'FRTscale':'T scale',  'Gamma':'$\gamma$', 'IVDf1':'$f_{Exp}$', 'IVDf2':'$f_2$', 'CMEvTrans':'v$_{Trans}$\n(km/s)', 'SWBx':'SW B$_x$\n(nT)', 'SWBy':'SW B$_y$\n(nT)', 'SWBz':'SW B$_z$\n(nT)', 'MHarea':'CH Area (10$^{10}$ km$^2$)', 'MHdist':'HSS Dist. (au)'}
    
    configID = 0
    if OSP.doFC: configID += 100
    if OSP.doANT: configID += 10
    if OSP.doFIDO: configID += 1
    nVertDict = {100:9, 110:14, 111:16, 11:13, 10:11, 1:4}
    nVert = nVertDict[configID]
    outDict = {100:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp'], 110:['CMElat', 'CMElon',  'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT','Kp'], 111:['CMElat', 'CMElon', 'CMEtilt', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT', 'B', 'Bz', 'Kp'], 11:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n',  'logT', 'B', 'Bz', 'Kp'], 10:['CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEvF', 'CMEvExp','TT', 'Dur', 'n', 'logT', 'Kp'], 1:['Dur',  'B', 'Bz',  'Kp']}
    
    # get impacts, may be less than nEns
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            hits.append(i)

    # number of vertical plots depends on num params varied
    nHoriz = len(varied)
            
    # group EnsRes once to avoid doing in each plot
    #nRuns = len(hits) # might need to change to throw out misses
    EnsVal = np.zeros([nHoriz, nEns]) 
    i = 0
    for key in ResArr.keys():
        j = 0
        for item in varied:
            if item != 'CMElon':
                EnsVal[j,key] = ResArr[key].EnsVal[item]
            else:
                EnsVal[j,key] = ResArr[key].EnsVal[item] - OSP.satPos[1] 
            j += 1
    
    # group the results
    OSPres = {}#np.zeros([nRuns, nVert])
    for item in outDict[configID]: OSPres[item] = []
    counter = 0
    i = 0
    goodIDs = []
    failIDs = []
    for key in ResArr.keys():
        if (not ResArr[key].miss):
            if not ResArr[key].fail:
                goodIDs.append(key)
            else:
                failIDs.append(key)
        elif configID == 100: goodIDs.append(key)
        
        for item in outDict[configID]:
            if item == 'CMElat':
                OSPres[item].append(ResArr[key].FClats[-1])
            if item == 'CMElon':
                OSPres[item].append(ResArr[key].FClonsS[-1])
            if item == 'CMEtilt':
                OSPres[item].append(ResArr[key].FCtilts[-1])
            if item == 'CMEAW':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTAWs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCAWs[-1])
            if item == 'CMEAWp':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTAWps[-1])
                else:
                    OSPres[item].append(ResArr[key].FCAWps[-1])
            if item == 'CMEdelAx':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTdelAxs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCdelAxs[-1])
            if item == 'CMEdelCS':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTdelCSs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCdelCSs[-1])
            if item == 'CMEvF':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTvFs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCvFs[-1])
            if item == 'CMEvExp':
                if OSP.doANT and (key in goodIDs):
                    OSPres[item].append(ResArr[key].ANTvCSrs[-1])
                else:
                    OSPres[item].append(ResArr[key].FCvCSrs[-1])
                    
            if key in goodIDs:
                if item == 'TT':
                    if OSP.doFIDO:
                        OSPres[item].append(ResArr[key].FIDOtimes[0])    
                    else:
                        OSPres[item].append(ResArr[key].ANTtimes[-1]+ResArr[key].FCtimes[-1]/60/24.)                    
                if item == 'Dur':
                    if OSP.doFIDO:
                        OSPres[item].append((ResArr[key].FIDOtimes[-1]-ResArr[key].FIDOtimes[0])*24)
                    else:
                        OSPres[item].append(ResArr[key].ANTdur)
                if item == 'n':
                    OSPres[item].append(ResArr[key].ANTns[-1])   
                if item == 'logT':
                    OSPres[item].append(ResArr[key].ANTlogTs[-1])                 
                if item == 'Kp':
                    if OSP.doFIDO:
                        OSPres[item].append(np.max(ResArr[key].FIDOKps))
                    else:
                        OSPres[item].append(ResArr[key].ANTKp0)
                if item == 'B':
                    OSPres[item].append(np.max(ResArr[key].FIDOBs))                                
                if item == 'Bz':
                    OSPres[item].append(np.min(ResArr[key].FIDOBzs))

    print ('Number of hits: ', len(goodIDs)) 
    print ('Mean and Standard Deviation')
    for item in outDict[configID]:
        OSPres[item] = np.array(OSPres[item])
        print (item, np.mean(OSPres[item]), np.std(OSPres[item]), np.min(OSPres[item]), np.max(OSPres[item]))  
    
    # calculate correlations
    corrs = np.zeros([nVert, nHoriz])
    for i in range(nHoriz):
        for j in range(nVert):
            if len(OSPres[outDict[configID][j]]) == nEns:
                col = np.abs(pearsonr(EnsVal[i,:], OSPres[outDict[configID][j]])[0])#*np.ones(nEns)
                #axes[j,i].scatter(EnsVal[i,:], OSPres[outDict[configID][j]], c=cm.turbo(col))            
            else:
                col = np.abs(pearsonr(EnsVal[i,goodIDs], OSPres[outDict[configID][j]])[0])#*np.ones(len(goodIDs))
                #axes[j,i].scatter(EnsVal[i,goodIDs], OSPres[outDict[configID][j]], c=cm.turbo(col))
            corrs[j,i] = col
    # clean out any NaNs (might have for zero variation params)
    corrs[~np.isfinite(corrs)] = 0
    
    # figure out which can be thrown out
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
    newIns  = np.array(varied)[goodHidx]
        
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
            
    f, a = plt.subplots(1, 1)
    img = a.imshow(np.array([[0,1]]), cmap="turbo")
    
    if newnHoriz > 1:
        fig, axes = plt.subplots(newnVert, newnHoriz, figsize=(1.5*newnHoriz,1.5*(newnVert+0.5)))
        for i in range(newnVert-1):
            for j in range(newnHoriz):
                axes[i,j].set_xticklabels([])
        for j in range(newnHoriz-1):
            for i in range(newnVert):
                axes[i,j+1].set_yticklabels([])
    
        for i in range(newnHoriz):
            for j in range(newnVert):
                if len(OSPres[newOuts[j]]) == nEns:
                    axes[j,i].scatter(newEnsVal[i,:], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,i]*np.ones(len(newEnsVal[i,:]))))   
                else:
                    axes[j,i].scatter(newEnsVal[i,goodIDs], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,i]*np.ones(len(newEnsVal[i,goodIDs])))) 
                
        # Rotate bottom axes labels34w34
        for i in range(newnHoriz):
            plt.setp(axes[-1,i].xaxis.get_majorticklabels(), rotation=70 )
    
        # Add labels
        for i in range(newnHoriz): axes[-1,i].set_xlabel(myLabs[newIns[i]])  
        for j in range(newnVert):  axes[j,0].set_ylabel(out2outLab[newOuts[j]])  
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.15, bottom=0.2, top=0.97, right=0.99)
        
    
    else:
        # need to reformat plot if only single significant input parameter    
        fig, axes = plt.subplots(newnVert, newnHoriz, figsize=(6,1.5*(newnVert+0.5)))
        for j in range(newnVert-1):
            axes[j].set_xticklabels([])
        for j in range(newnVert):
            if len(OSPres[newOuts[j]]) == nEns:
                axes[j].scatter(newEnsVal[0,:], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,0]*np.ones(len(newEnsVal[0,:]))))
            else:
                axes[j].scatter(newEnsVal[0,goodIDs], OSPres[newOuts[j]], c=cm.turbo(newCorr[j,0]*np.ones(len(newEnsVal[0,goodIDs]))))
        
        axes[-1].set_xlabel(myLabs[newIns[0]])  
        for j in range(newnVert): 
            axes[j].set_ylabel(out2outLab[newOuts[j]])
        plt.subplots_adjust(hspace=0.01, wspace=0.01, left=0.25, bottom=0.2, top=0.97, right=0.98)
        
       
    cbar_ax = fig.add_axes([0.15, 0.09, 0.79, 0.02])    
    cb = fig.colorbar(img, cax=cbar_ax, orientation='horizontal')   
    cb.set_label('Correlation') 
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_ENS.'+figtag)
                    
def makeAllprob(ResArr):
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
    nx = int(plotlen/3.)#+1
    
    nBins = 10
    allArr = np.zeros([7, nBins, nx])
    
    # want first cell to start at first multiple of 3 hrs before first arrival
    hrs3 = int((mindate+DoY - int(mindate+DoY))*8)
    startPlot = int(mindate+DoY) + hrs3/8.
    # Calculate the values at grid edges
    gridtimes = np.array([startPlot + i*3./24. for i in range(nx+1)])
    centtimes = np.array([startPlot + 1.5/24. + i*3./24. for i in range(nx+1)])
    
    # need min/max for each parameter
    minmax = np.zeros([7,2])
    minmax[4,0] = 400 # set v min to 350 km/s so will probably include steady state vSW range
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes is not None:
            thisRes = ResArr[key]
            if OSP.isSat:
                allParams = [thisRes.FIDOBs, thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs, thisRes.FIDOvs, thisRes.FIDOtems, thisRes.FIDOns]
            else:
                allParams = [thisRes.FIDOBs, thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs, thisRes.FIDOvs, thisRes.FIDOtems, thisRes.FIDOKps]
            for i in range(7):
                thismin, thismax = np.min(allParams[i]), np.max(allParams[i]) 
                if thismin < minmax[i,0]: minmax[i,0] = thismin
                if thismax > minmax[i,1]: minmax[i,1] = thismax 

    # need to compare to min/max of obs data (if given)
    if not OSP.noDate: 
        obsIdx = [1, 2, 3, 4, 6, 7, 5]
        if not OSP.isSat:
            obsIdx[-1] = 8
        for i in range(7):
            thisParam = ObsData[obsIdx[i],:].astype(np.float)
            thisParam =  thisParam[~np.isnan(thisParam.astype(float))]
            thismin, thismax = np.min(thisParam), np.max(thisParam)
            if thismin < minmax[i,0]: minmax[i,0] = thismin
            if thismax > minmax[i,1]: minmax[i,1] = thismax

    # fill in grid
    counter = 0
    for key in ResArr.keys():
        if ResArr[key].FIDOtimes is not None:
            counter += 1
            thisTime = ResArr[key].FIDOtimes + DoY
            thisRes = ResArr[key]
            if OSP.isSat:
                allParams = [thisRes.FIDOBs, thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs, thisRes.FIDOvs, thisRes.FIDOtems, thisRes.FIDOns]
            else:
                allParams = [thisRes.FIDOBs, thisRes.FIDOBxs, thisRes.FIDOBys, thisRes.FIDOBzs, thisRes.FIDOvs, thisRes.FIDOtems, thisRes.FIDOKps]
            
            for j in range(7):
                thismin = minmax[j,0]
                thisrange = minmax[j,1] - minmax[j,0]
                thisbin = thisrange / nBins
                thisParam = allParams[j]
                
                try:
                    thefit = CubicSpline(thisTime, thisParam)
                except:
                    # might have duplicate at front
                    thefit = CubicSpline(thisTime[1:], thisParam[1:])
                tidx = np.where((gridtimes >= thisTime[0]-3./24.) & (gridtimes <= thisTime[-1]))[0]
                thisCentT = centtimes[tidx]
                
                thisRes = ((thefit(thisCentT) -  thismin)/thisbin).astype(int)
                for i in range(len(thisRes)-1):
                    thisCell = thisRes[i]
                    if thisCell < 0: thisCell=0
                    if thisCell >= nBins: thisCell = nBins-1
                    allArr[j,thisCell,tidx[i]] += 1
    if not OSP.noDate:
        # convert x axis to dates
        labelDays = gridtimes[np.where(2*gridtimes%1 == 0)]
        base = datetime.datetime(yr, 1, 1, 0, 0)
        dates = np.array([base + datetime.timedelta(days=i) for i in labelDays])    
        dateLabels = [i.strftime('%Y %b %d %H:%M ') for i in dates]    
        plotStart = base + datetime.timedelta(days=(gridtimes[0]))
    allPerc = allArr/float(counter)*100
    
    cmap1 = cm.get_cmap("turbo",lut=10)
    cmap1.set_bad("w")
    allMasked = np.ma.masked_less(allPerc,0.01)
                     
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(8,12))
    uplim = 0.9*np.max(allMasked)
    for i in range(7):
        ys = np.linspace(minmax[i,0], minmax[i,1],nBins+1)
        XX, YY = np.meshgrid(gridtimes,ys)
        # draw a grid because mask away a lot of it
        for x in gridtimes: axes[i].plot([x,x],[ys[0],ys[-1]], c='LightGrey')
        for y in ys: axes[i].plot([gridtimes[0],gridtimes[-1]],[y,y], c='LightGrey')
        c = axes[i].pcolor(XX,YY,allMasked[i,:,:], cmap=cmap1, edgecolors='k', vmin=0, vmax=100)
        
    # add in observations
    if ObsData is not None:
        # need to convert obsdate in datetime fmt to frac dates
        obsDates =  ObsData[0,:]
        for i in range(len(obsDates)):
             obsDates[i] = (obsDates[i].timestamp()-plotStart.timestamp())/24./3600. +gridtimes[0]
        cs = ['k', 'r']
        lw = [6,3]
        for i in range(2):
            axes[0].plot(obsDates, ObsData[1,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[1].plot(obsDates, ObsData[2,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[2].plot(ObsData[0,:], ObsData[3,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[3].plot(ObsData[0,:], ObsData[4,:], linewidth=lw[i], color=cs[i], zorder=5)
            axes[4].plot(ObsData[0,:], ObsData[6,:], linewidth=lw[i], color=cs[i], zorder=5)    
            axes[5].plot(ObsData[0,:], ObsData[7,:], linewidth=lw[i], color=cs[i], zorder=5)
            if OSP.isSat:
                axes[6].plot(ObsData[0,:], ObsData[5,:], linewidth=lw[i], color=cs[i], zorder=5)
            else:
                axes[6].plot(ObsData[0,:], ObsData[8,:], linewidth=lw[i], color=cs[i], zorder=5)
        
        
    # add in ensemble seed
    if OSP.noDate:
        dates2 = ResArr[0].FIDOtimes
    else:
        dates2 = np.array([base + datetime.timedelta(days=(i+DoY)) for i in ResArr[0].FIDOtimes])
        for i in range(len(dates2)):
             dates2[i] = (dates2[i].timestamp()-plotStart.timestamp())/24./3600. +gridtimes[0]
    thiscol = ['w', 'b']
    for i in range(2):
        axes[0].plot(dates2, ResArr[0].FIDOBs, linewidth=lw[i], color=thiscol[i], zorder=6)
        axes[1].plot(dates2, ResArr[0].FIDOBxs, linewidth=lw[i], color=thiscol[i], zorder=6)
        axes[2].plot(dates2, ResArr[0].FIDOBys, linewidth=lw[i], color=thiscol[i], zorder=6)
        axes[3].plot(dates2, ResArr[0].FIDOBzs, linewidth=lw[i], color=thiscol[i], zorder=6)
        axes[4].plot(dates2, ResArr[0].FIDOvs, linewidth=lw[i], color=thiscol[i], zorder=6)
        axes[5].plot(dates2, ResArr[0].FIDOtems, linewidth=lw[i], color=thiscol[i], zorder=6)
        if OSP.isSat:
            axes[6].plot(dates2, ResArr[0].FIDOns, linewidth=lw[i], color=thiscol[i], zorder=6)
        else:
            axes[6].plot(dates2, ResArr[0].FIDOKps, linewidth=lw[i], color=thiscol[i], zorder=6)
        
    axes[0].set_xlim(gridtimes[0],gridtimes[-1])
        
    axes[0].set_ylabel('B (nT)')
    axes[1].set_ylabel('B$_x$ (nT)')
    axes[2].set_ylabel('B$_y$ (nT)')
    axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('v (km/s)')
    axes[5].set_ylabel('T (K)')
    if OSP.isSat:
        axes[6].set_ylabel('n (cm$^{-3}$)')
    else:
        axes[6].set_ylabel('Kp')
        
        
    
    xlims = list(axes[6].get_xlim())
    xlims[1] = xlims[1]- 3/24.
    axes[6].set_xlim(xlims)
    if not OSP.noDate: 
        plt.xticks(labelDays, dateLabels)
        fig.autofmt_xdate()
    plt.subplots_adjust(left=0.2,right=0.95,top=0.95,bottom=0.15, hspace=0.15)
    
    ax0pos = axes[0].get_position()
    fig.subplots_adjust(top=0.9)
    cbar_ax = fig.add_axes([ax0pos.x0, 0.94, ax0pos.width, 0.02])
    cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal')
    cbar.ax.set_title('Percentage Chance')        
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_allPerc.'+figtag)    
                               
def makeContours(ResArr, calcwid=90, plotwid=40):
    # Start by filling in the area that corresponds to the CME in a convenient frame
    # then rotate it the frame where Earth is at [0,0] at the time of impact for all CMEs
    # using its exact position in "real" space (which will vary slightly between CMEs).
    # Also derive parameters based on Earth's coord system if there were changes in it's
    # position (i.e. vr for rhat corresponding to each potential shifted lat/lon)
            
    # simulation parameters
    ngrid = 2*plotwid+1
    ncalc = 2*calcwid+1
    nThings = 39
    shiftx, shifty = calcwid, calcwid
    counter = 0
    
    # get impacts, may be less than nEns
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            hits.append(i)
    allGrid = np.zeros([len(hits), ngrid, ngrid, nThings])
    
    for key in hits:
        newGrid = np.zeros([ncalc,ncalc,nThings])
        
        # pull in things from ResArr
        if OSP.doFC:
            thisLat = ResArr[key].FClats[-1]
            thisLon = ResArr[key].FClons[-1] % 360.
            thisTilt  = ResArr[key].FCtilts[-1]
        else:
            thisLat = float(OSP.input_values['CMElat'])
            thisLon = float(OSP.input_values['CMElon'])
            thisTilt = float(OSP.input_values['CMEtilt'])
        # use time of first impact
        thisidx   = ResArr[key].ANTFRidx    
        thisAW    = ResArr[key].ANTAWs[thisidx]
        thisAWp   = ResArr[key].ANTAWps[thisidx]
        thisR     = ResArr[key].ANTrs[thisidx]
        thisDelAx = ResArr[key].ANTdelAxs[thisidx]
        thisDelCS = ResArr[key].ANTdelCSs[thisidx]
        thisDelCA = ResArr[key].ANTdelCSAxs[thisidx]
        thesevs = np.array([ResArr[key].ANTvFs[thisidx], ResArr[key].ANTvEs[thisidx], ResArr[key].ANTvBs[thisidx], ResArr[key].ANTvCSrs[thisidx], ResArr[key].ANTvCSps[thisidx], ResArr[key].ANTvAxrs[thisidx], ResArr[key].ANTvAxps[thisidx]])   
        thisB0    = ResArr[key].ANTB0s[thisidx] * np.sign(float(OSP.input_values['FRBscale']))
        thisTau   = ResArr[key].ANTtaus[thisidx]
        thisCnm   = ResArr[key].ANTCnms[thisidx]
        thisPol   = int(float(OSP.input_values['FRpol']))
        thislogT  = ResArr[key].ANTlogTs[thisidx]
        thisn     = ResArr[key].ANTns[thisidx]
        
        # Calculate the CME lengths 
        # CMElens = [CMEnose, rEdge, rCent, rr, rp, Lr, Lp]
        CMElens = np.zeros(7)
        CMElens[0] = thisR
        CMElens[4] = np.tan(thisAWp*dtor) / (1 + thisDelCS * np.tan(thisAWp*dtor)) * CMElens[0]
        CMElens[3] = thisDelCS * CMElens[4]
        CMElens[6] = (np.tan(thisAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + thisDelAx * np.tan(thisAW*dtor))  
        CMElens[5] = thisDelAx * CMElens[6]
        CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
        CMElens[1] = CMElens[2] * np.tan(thisAW*dtor)
              
        # Find the location of the axis
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
        
        # figure out mappings between the latitude and other variables
        lat2theT =  CubicSpline(axSPH[1],thetas/dtor,bc_type='natural')
        lat2AWp = CubicSpline(axSPH[1],varyAWp,bc_type='natural')
        lat2xFR = CubicSpline(axSPH[1],xFR,bc_type='natural')
        minlat, maxlat = int(round(np.min(axSPH[1]))), int(round(np.max(axSPH[1])))
        minlon, maxlon = shiftx-int(round(thisAWp)),shiftx+int(round(thisAWp))
        lat2AWpEd = CubicSpline(edgeSPH[1][::-1],edgeSPH[2][::-1],bc_type='natural')    
        lon2TP = CubicSpline(edgeSPH[2],thetaPs/dtor,bc_type='natural') 
        # check lat limits to make sure we don't go out of range
        maxn = ncalc-1
        if minlat+shifty < 0: minlat = -shifty
        if maxlat+1+shifty > maxn: maxlat = maxn-1-shifty
        minlat2, maxlat2 = int(round(maxlat)), int(round(topSPH[1]))  
        
        if maxlat2+1+shifty > maxn: maxlat2 = maxn-1-shifty
        
        # Loop through in latitude and fill in points that are in the CME
        for i in range(minlat, maxlat+1):
            # Find the range to fill in lon
            idy =  i+shifty
            nowAWp = np.round(lat2AWp(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            # Fill from minlon to maxlon
            newGrid[idy, np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,0] = 1
                       
            # Pad things 2 deg outside "correct" lon range
            newGrid[idy,  np.maximum(0,minlon-2):np.minimum(maxn,maxlon+2)+1,1] = 1
               
            # Calculate the parameteric thetas for each point in the CME
            # ThetaT - start with relation to lat
            thetaT = lat2theT(i)
            newGrid[idy,  np.maximum(0,minlon):np.minimum(maxn,maxlon)+1,2] = thetaT 
            
            # ThetaP - making geometric approx to calc sinThetaP = axis_X tanLon / rperp
            # which ignores that this perp width is not at axis but axis + xCS (which is f(thetaP))
            # (CME is oriented so AWp is in lon direction before we rotate it)
            theselons = np.arange(-nowAWp, nowAWp+1)
            sinTP = lat2xFR(i) * np.tan(theselons*dtor) / CMElens[4]
            # Clean up any places with sinTP > 1 from our geo approx so it just maxes out not blows up
            sinTP[np.where(np.abs(sinTP) > 1)] = np.sign(sinTP[np.where(np.abs(sinTP) > 1)]) * 1
            thetaPs = np.arcsin(sinTP)/dtor
            newGrid[idy,  minlon:maxlon+1,3] = thetaPs
        
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
                         
        # Clean up  angles so sin or cos don't blow up
        fullTT = newGrid[:,:,2]
        fullTT[np.where(np.abs(fullTT)<1e-5)] = 1e-5
        fullTT[np.where(np.abs(fullTT)>0.9*90)] = np.sign(fullTT[np.where(np.abs(fullTT)>.9*90)]) * 0.9 * 90
        newGrid[:,:,2] = fullTT * dtor
        fullTP = newGrid[:,:,3]
        fullTP[np.where(np.abs(fullTP)<1e-5)] = np.sign(fullTP[np.where(np.abs(fullTP)<1e-5)]) * 1e-5
        fullTP[np.where(np.abs(fullTP)>0.9*90)] = np.sign(fullTP[np.where(np.abs(fullTP)>0.9*90)]) * 0.9 * 90
        newGrid[:,:,3] = fullTP * dtor
        
        
        # Vectors ---------------------------------------------------------------------------------
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
                
        # Velocity ----------------------------------------------------------------------------------
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
       
        # Coordinate transformations -------------------------------------------------------------
        # Calculate how far to shift the CME in lat/lon to put Earth at 0,0
        dLon = OSP.satPos[1] + ResArr[key].ANTtimes[-1] * 24 * 3600 * OSP.Sat_rot - thisLon
        dLat = OSP.satPos[0] - thisLat
                        
        # Create the background meshgrid and shift it
        XX, YY = np.meshgrid(range(-shiftx,shiftx+1),range(-shifty,shifty+1))
        
        # Calculate local radial vector, find v component in that dir
        rhatE = np.zeros([ncalc,ncalc,3])
        colat = (90 - YY) * dtor
        rhatE[:,:,0] = np.sin(colat) * np.cos(XX*dtor) 
        rhatE[:,:,1] = np.sin(colat) * np.sin(XX*dtor)
        rhatE[:,:,2] = np.cos(colat)
        newGrid[:,:,24] = rhatE[:,:,0] * newGrid[:,:,18] + rhatE[:,:,1] * newGrid[:,:,19] + rhatE[:,:,2] * newGrid[:,:,20]
        newGrid[:,:,25] = rhatE[:,:,0] * newGrid[:,:,21] + rhatE[:,:,1] * newGrid[:,:,22] + rhatE[:,:,2] * newGrid[:,:,23]
        
        # Shift the meshgrid   
        XX = XX.astype(float) - dLon
        YY = YY.astype(float) - dLat

        # Rotate newGrid array based on CME tilt
        newGrid = ndimage.rotate(newGrid, 90-thisTilt, reshape=False)
        # Force in/out to be 0/1 again
        newGrid[:,:,0] = np.rint(newGrid[:,:,0])
        newGrid[:,:,1] = np.rint(newGrid[:,:,1])
        
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
               
        # local radial unit vector if earth was at a given lat/lon instead of 0,0
        rhatE = np.zeros([ncalc,ncalc,3])
        colat = (90 - YY) * dtor
        rhatE[:,:,0] = np.sin(colat) * np.cos(XX*dtor) 
        rhatE[:,:,1] = np.sin(colat) * np.sin(XX*dtor)
        rhatE[:,:,2] = np.cos(colat)

        # local z unit vector if earth was at a given lat/lon instead of 0,0 (same as colat vec)
        zhatE = np.zeros([ncalc,ncalc,3])
        zhatE[:,:,0] = -np.abs(np.cos(colat)) * np.cos(XX*dtor)
        zhatE[:,:,1] = -np.cos(colat) * np.sin(XX*dtor)
        zhatE[:,:,2] = np.sin(colat)
        
        # local y unit vector, will need for Kp clock angle
        yhatE = np.zeros([ncalc,ncalc,3])
        yhatE[:,:,0] = -np.sin(XX*dtor)
        yhatE[:,:,1] = np.cos(XX*dtor)
        
        
        # Duration --------------------------------------------------------------
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
        
        # Magnetic field ----------------------------------------------------------------
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
               
        # Kp ----------------------------------------------------------------------------
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
        
        # Temperature and density -------------------------------------------------------
        newGrid[:,:,37] = thislogT * newGrid[:,:,1] # uniform temp within CME 
        newGrid[:,:,38] =    thisn * newGrid[:,:,1] # uniform dens within CME 
                      
        # Interpolate and cut out window around Earth/sat to use for summing and plotting
        # Integer shift
        delxi, delyi = -shiftx-int(XX[0,0]), -shifty-int(YY[0,0])
        # Remainder from integer
        delx, dely = int(XX[0,0]) - XX[0,0], int(YY[0,0]) - YY[0,0]
        # Perform shift in x
        startidx = shiftx - plotwid + delxi
        leftX = newGrid[:,startidx:startidx+2*plotwid+1,:]
        rightX = newGrid[:,startidx+1:startidx+2*plotwid+2,:]
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
    fig, axes = plt.subplots(2, 5, figsize=(11,6))
    cmap1 = cm.get_cmap("plasma",lut=10)
    cmap1.set_bad("k")
    # Reorder Axes
    axes = [axes[0,0], axes[0,1], axes[0,2], axes[0,3], axes[0,4], axes[1,0], axes[1,1], axes[1,2], axes[1,3], axes[1,4]]
    labels = ['Chance of Impact (%)', 'B$_z$ Front (nT)', 'v$_r$ Front (km/s)',  'Kp Front', 'n (cm$^{-1}$)', 'Duration (hr)', 'B$_z$ Center (nT)', 'v$_r$ Center (km/s)',  'Kp Center', 'log(T) (K)']
    
    
    # Get the number of CMEs in each grid cell
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
        
    for i in range(len(axes)):
        axes[i].set_facecolor('k')
        axes[i].set_aspect('equal', 'box')
        
        toPlotNow = toPlot[:,:,i]
        cent, rng = np.mean(toPlotNow[nCMEs>0]), 1.5*np.std(toPlotNow[nCMEs>0])
        if i == 0: cent, rng = 50, 50
        toPlotNow[nCMEs==0] = np.inf
        c = axes[i].pcolor(subXX,subYY,toPlotNow,cmap=cmap1,  vmin=cent-rng, vmax=cent+rng, shading='auto')
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
            
        axes[i].plot(0, 0, 'o', ms=15, mfc='#98F5FF')
        if i > 4:
            #axes[i].set_xticklabels([])
            axes[i].xaxis.set_ticks_position('top') 
        else:
            axes[i].tick_params(axis='x', which='major', pad=5)
            axes[i].set_xlabel('Lon ($^{\circ}$)')
        #axes[i].xaxis.set_ticks([-plotwid, -plotwid/2, 0, plotwid/2, plotwid])
        if i not in [0,5]: 
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('Lat ($^{\circ}$)')

    plt.xticks(fontsize=10)    
    plt.subplots_adjust(wspace=0.2, hspace=0.46,left=0.1,right=0.95,top=0.85,bottom=0.12)    
    plt.savefig(OSP.Dir+'/fig'+str(ResArr[0].name)+'_Contours.png')    

    
if __name__ == '__main__':
    # set whether to save the figures as png or pdf
    # set only to 'png' or 'pdf'
    global figtag
    figtag = 'png'
    
    # Get all the parameters from text files and sort out 
    # what we actually ran
    OSP.setupOSPREI()
    
    # check and see if we were given a time for GCS observations 
    # to shift Clon of satellite from start to GCS time and do 
    # Stony lon for CPA plot at time of GCS obs
    GCStime = 0
    if len(sys.argv) > 2:
        GCStime = float(sys.argv[2])
          
    ResArr = txt2obj(GCStime)
     
    global ObsData
    ObsData = None
    if OSP.ObsDataFile is not None:
        ObsData = readInData()
    
    global nEns
    nEns = len(ResArr.keys())

    # Plots we can make for single CME simulations
    if OSP.doFC:
        # Make CPA plot
        makeCPAplot(ResArr)
        # Make the AW, delta, v plot
        # Non-forecast plot
        #makeADVplot(ResArr)    # haven't checked post FIDO integration into ANT

    if OSP.doANT:
        if OSP.doPUP:
            # make IP plot including sheath stuff
            makePUPplot(ResArr)
        else:
            # Make drag profile
            makeDragless(ResArr)
            # Non-forecast version with more params
            #makeDragplot(ResArr)  # haven't checked post FIDO integration into ANT

    if OSP.doFIDO:
        # Make in situ plot
        makeISplot(ResArr)

    # Ensemble plots
    if nEns > 1:
        if OSP.doFC:
            # Make CPA plot
            makeCPAhist(ResArr)
            
        if OSP.doANT:
            # Make arrival time hisogram 
            makeAThisto(ResArr)
                
        if OSP.doFIDO:
            # FIDO histos- duration, minBz
            # Non-forecast version with more params
            # Run it anyway in case we don't have a sheath and 
            # missing params from forecast version
            makeFIDOhistos(ResArr)
            if OSP.includeSIT:
                # Non-forecast version with more params
                #makeSIThistos(ResArr)
                makeallIShistos(ResArr)
            # Kp probability timeline
            makeAllprob(ResArr)

        # Slow plots -- worth commenting out if running quick tests
        # and not looking specifically at these
        
        # Ensemble input-output plot
        makeEnsplot(ResArr,critCorr=0.5)
        
        # Contour plot
        makeContours(ResArr)

        