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
from CME_class import cart2cart
from ANT_PUP import lenFun

# |--------------------------------------------------------------|
# |----- Set up a class to hold all the results from OSPREI -----|
# |--------------------------------------------------------------|
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
            self.FIDOmiss = True
            self.fail = False
            
            # Dictionary for ensemble things
            self.EnsVal = {}
 
# |--------------------------------------------------------------|
# |-------- Read in the text file and fill the object -----------|
# |--------------------------------------------------------------|
def txt2obj(GCStime=0):
    ResArr = {}
    global nHits, nFails
    nHits = 0
    nFails = 0
    
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
            thisRes.FClons  = FCdata[myidxs,4] % 360.
            if (thisRes.FClons[0] < 100.) & (thisRes.FClons[-1] > 250):
                thisRes.FClons[np.where(thisRes.FClons > 250)] -= 360.
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
        if (8888. in ANTdata) & (len(ANTdata.shape) == 1):
            print ('Skipping ANTEATR figures because forces were unstable')
            print ('Try decreasing AWp. Forces tend to break if AWp is too large relative to AW.')
            ANTids = []
            unANTids = []
            OSP.doANT = False
        else:
            ANTids = ANTdata[:,0].astype(int)
            unANTids = np.unique(ANTdata[:,0].astype(int))
        if OSP.doPUP:
            PUPfile = OSP.Dir+'/PUPresults'+OSP.thisName+'.dat'
            PUPdata = np.genfromtxt(PUPfile, dtype=float, encoding='utf8')
            try:
                PUPids = PUPdata[:,0].astype(int)
                unPUPids = np.unique(PUPdata[:,0].astype(int))
            except:
                PUPids = []
                unPUPids = []
            
        
        for i in unANTids:
            # Check if we already have set up the objects
            # If not do now
            if not OSP.doFC:  
                thisRes = EnsRes(OSP.thisName)
                thisRes.myID = i
                # add in CME lat lon tilt
                thisRes.FClats = [float(OSP.input_values['CMElat'])]
                thisRes.FClons = [float(OSP.input_values['CMElon'])]
                thisRes.FCtilts = [float(OSP.input_values['CMEtilt'])]                
            else:
                thisRes = ResArr[i]
                
            # Set as an impact not a miss
            myidxs = np.where(ANTids==i)[0]  
            if int(ANTdata[myidxs[0],1]) == 8888:
                thisRes.fail = True
                nFails += 1
                ResArr[thisRes.myID] = thisRes
            else:
                nHits += 1
                if not OSP.doFIDO:
                    thisRes.FIDOmiss = [False]
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
                # for some reason Vs are passed in terrible order
                thisRes.ANTvFs   = ANTdata[myidxs,8]
                thisRes.ANTvEs   = ANTdata[myidxs,9]
                thisRes.ANTvBs   = ANTdata[myidxs,14]
                thisRes.ANTvAxrs = ANTdata[myidxs,10]
                thisRes.ANTvAxps = ANTdata[myidxs,13]
                thisRes.ANTvCSrs = ANTdata[myidxs,11]
                thisRes.ANTvCSps = ANTdata[myidxs,12]
                thisRes.ANTB0s   = ANTdata[myidxs,15]
                thisRes.ANTCnms  = ANTdata[myidxs,16]
                # tau is constant for now
                thisRes.ANTtaus  = ANTdata[myidxs,17]
                thisRes.ANTns    = ANTdata[myidxs,18]
                thisRes.ANTlogTs = ANTdata[myidxs,19] 
                thisRes.ANTyaws  = ANTdata[myidxs,21] 
                thisRes.ANTyawvs = ANTdata[myidxs,22] 
                thisRes.ANTyawAs = ANTdata[myidxs,23] 
                
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
                dphidt = np.power(thisRes.ANTvFs[-1], 4/3.) * np.power(np.abs(thisRes.ANTBpols[-1]), 2./3.) 
                # Mays/Savani expression, best behaved for high Kp
                thisRes.ANTKp0 = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
                # calc density 
                '''thisRes.ANTrp = thisRes.ANTrr / thisRes.ANTdelCSs[-1]
                alpha = np.sqrt(1+16*thisRes.ANTdelAxs[-1]**2)/4/thisRes.ANTdelAxs[-1]
                thisRes.ANTLp = (np.tan(thisRes.ANTAWs[-1]*dtor)*(1-Deltabr) - alpha*Deltabr)/(1+thisRes.ANTdelAxs[-1]*np.tan(np.tan(thisRes.ANTAWs[-1]*dtor))) * thisRes.ANTrs[-1]
                vol = math.pi*thisRes.ANTrr*thisRes.ANTrp *  lenFun(thisRes.ANTdelAxs[-1])*thisRes.ANTrs[-1]
                thisRes.ANTn = OSP.mass*1e15 / vol / 1.67e-24 / (7e10)**3'''
                
                if OSP.doPUP and (len(PUPdata.shape)!=1):
                    myPUPidxs = np.where(PUPids==i)[0]  
                    
                    if PUPdata[myPUPidxs[0],1] != 8888:
                        thisRes.PUPvshocks = PUPdata[myPUPidxs,1]
                        thisRes.PUPcomps = PUPdata[myPUPidxs,2]
                        thisRes.PUPMAs = PUPdata[myPUPidxs,3]
                        thisRes.PUPwids = PUPdata[myPUPidxs,4]
                        thisRes.PUPdurs = PUPdata[myPUPidxs,5]
                        thisRes.PUPMs = PUPdata[myPUPidxs,6]
                        thisRes.PUPns = PUPdata[myPUPidxs,7]
                        thisRes.PUPlogTs = PUPdata[myPUPidxs,8]
                        thisRes.PUPBthetas = PUPdata[myPUPidxs,9]
                        thisRes.PUPBs = PUPdata[myPUPidxs,10]
                        thisRes.PUPvts = PUPdata[myPUPidxs,11]
                        thisRes.PUPinit = PUPdata[myPUPidxs,12]
                                        
                ResArr[thisRes.myID] = thisRes
               
    if OSP.doFIDO:
        global nSat, satNames, hitsSat 
        satNames = ['']
        nSat = 1
        
        if 'satPath' in OSP.input_values:
            satPath = OSP.input_values['satPath']
            satPaths = OSP.satPathWrapper(satPath, checkSatlen=False)
            satNames = satPaths[-1]
            nSat = len(satNames)
            # assume single sat cases are unnamed
            #if nSat == 1:
            #    satNames = ['']
        satNames = np.array(satNames)
        if (nSat == 1) and (satNames[0]=='sat1'):
            satNames = np.array([''])
        # need to track if we actually have any cases impact that satellite
        # if not, easiest to just turn off plotting
        hitsSat = [False for i in range(nSat)]
        
        FIDOdata = [[] for i in range(nSat)]
        ids = [[] for i in range(nSat)]
        unFIDOids = [[] for i in range(nSat)]
        
        for i in range(nSat):
            satName = satNames[i]
            FIDOfile = OSP.Dir+'/FIDOresults'+OSP.thisName+satName+'.dat'
            file_size = os.path.getsize(FIDOfile)
            if file_size != 0:
                FIDOdata[i] = np.genfromtxt(FIDOfile, dtype=float, encoding='utf8')
                # Check that contact actually happens and prints to a file otherwise
                # don't try to plot FIDO data
                if (len(FIDOdata[i].shape)==1) or (len(FIDOdata[i][:,0]) == int(FIDOdata[i][-1,0]+1)):
                    print('No contact so not plotting FIDO data')
                    OSP.doFIDO = False
                else:
                    ids[i] = FIDOdata[i][:,0].astype(int)
                    unFIDOids[i] = np.unique(ids[i])
                    hitsSat[i] = True
                 
        if OSP.doPUP:
            SITdata = [[] for i in range(nSat)]
            SITids  = [[] for i in range(nSat)]
            for i in range(nSat):
                satName = satNames[i]
                SITfile = OSP.Dir+'/SITresults'+OSP.thisName+satName+'.dat'
                SITdata[i] = np.genfromtxt(SITfile, dtype=float, encoding='utf8')
                if len(SITdata[i].shape) > 1:
                    SITids[i] = SITdata[i][:,0].astype(int)
                else:
                    if np.size(SITdata[i]) != 0:
                        SITids[i] = np.array([0]) # single case
                        SITdata[i] = SITdata[i].reshape([1,-1])
                    else:
                        SITids[i] = []
                        
        if (not OSP.doFC) and (not OSP.doANT):
            for i in unFIDOids[0]:
                thisRes = EnsRes(OSP.thisName)
                thisRes.myID = i
                # add in CME lat lon tilt
                thisRes.FClats = [float(OSP.input_values['CMElat'])]
                thisRes.FClons = [float(OSP.input_values['CMElon'])]
                thisRes.FCtilts = [float(OSP.input_values['CMEtilt'])]
                ResArr[i] = thisRes
                        
        # Reset FIDO data to hold correct number of sats
        for key in ResArr.keys():
            thisRes = ResArr[key]
            thisRes.FIDOtimes  = [None for i in range(nSat)]
            thisRes.FIDOmiss   = [True for i in range(nSat)]
            thisRes.FIDOBxs    = [[] for i in range(nSat)]
            thisRes.FIDOBys    = [[] for i in range(nSat)]
            thisRes.FIDOBzs    = [[] for i in range(nSat)]
            thisRes.FIDOBs     = [[] for i in range(nSat)]
            thisRes.FIDOvs     = [[] for i in range(nSat)]
            thisRes.FIDOns     = [[] for i in range(nSat)]
            thisRes.FIDOtems   = [[] for i in range(nSat)]
            thisRes.FIDOKps    = [[] for i in range(nSat)]
            thisRes.regions    = [[] for i in range(nSat)]
            thisRes.FIDO_shidx = [[] for i in range(nSat)]
            thisRes.FIDO_FRidx = [[] for i in range(nSat)]
            thisRes.FIDO_SWidx = [[] for i in range(nSat)]
            thisRes.FIDO_FRdur = [0 for i in range(nSat)]
            thisRes.FIDO_shdur = [0 for i in range(nSat)]
            thisRes.FIDO_FRexp = [0 for i in range(nSat)]
            thisRes.ANTshidx   = [[] for i in range(nSat)]
            thisRes.ANTFRidx   = [None for i in range(nSat)]
            thisRes.ANTdur = [0 for i in range(nSat)]
            thisRes.ANTKp0 = [0 for i in range(nSat)]
            thisRes.FIDOnormrs = [[] for i in range(nSat)]
            thisRes.isSheath   = [[] for i in range(nSat)]
            thisRes.hasSheath  = [False for i in range(nSat)]   
            thisRes.FIDO_shidx = [[] for i in range(nSat)]
            
            thisRes.SITidx = [[] for i in range(nSat)]
            thisRes.SITdur = [0 for i in range(nSat)]
            thisRes.SITcomp = [0 for i in range(nSat)]
            thisRes.SITMach = [0 for i in range(nSat)]
            thisRes.SITn    = [0 for i in range(nSat)]
            thisRes.SITvSheath = [0 for i in range(nSat)]
            thisRes.SITB    = [0 for i in range(nSat)]   
            thisRes.SITvShock = [0 for i in range(nSat)] 
            thisRes.SITtemp = [0 for i in range(nSat)]
            thisRes.SITminBz = [0 for i in range(nSat)]
            thisRes.SITmaxB = [0 for i in range(nSat)]
            thisRes.SITmaxKp = [0 for i in range(nSat)]
            ResArr[key] = thisRes  
                       
                        
        for k in range(nSat):    
            for i in unFIDOids[k]:  
                thisRes = ResArr[i] 
                myidxs = np.where(ids[k]==i)[0] 
                skipit = False
                # Make sure we actually want to include this case
                if (OSP.doFC) or (OSP.doANT) and (i in ResArr.keys()):
                    thisRes = ResArr[i]
                    if thisRes.fail:
                        skipit = True   
                elif (OSP.doFC) or (OSP.doANT) and (i not in ResArr.keys()):
                    skipit = True   
                elif (FIDOdata[k][myidxs[0],1] == 8888.):
                    skipit = True
                
                if not skipit:
                    # Set as an impact not a miss (might not have run ANTEATR)                    
                    thisRes.FIDOmiss[k] = False
                    myidxs = np.where(ids[k]==i)[0]
                    thisRes.FIDOtimes[k] = FIDOdata[k][myidxs,1]
                    # Check for duplicates in FIDO times, can happen apparently between diff regs
                    uniqtimes, udix = np.unique(thisRes.FIDOtimes[k], return_index=True)
                    myidxs = myidxs[udix]
                    thisRes.FIDOtimes[k] = FIDOdata[k][myidxs,1]
                    thisRes.FIDOBs[k]    = FIDOdata[k][myidxs,2]
                    thisRes.FIDOBxs[k]   = FIDOdata[k][myidxs,3]
                    thisRes.FIDOBys[k]   = FIDOdata[k][myidxs,4]
                    thisRes.FIDOBzs[k]   = FIDOdata[k][myidxs,5]
                    thisRes.FIDOvs[k]    = FIDOdata[k][myidxs,6]
                    thisRes.FIDOns[k]    = FIDOdata[k][myidxs,7]
                    thisRes.FIDOtems[k]  = np.power(10,FIDOdata[k][myidxs,8])
                    thisRes.regions[k] = FIDOdata[k][myidxs,9]
                    #regions = FIDOdata[myidxs,9]
                    thisRes.FIDO_shidx[k] = []
                    if 0 in thisRes.regions[k]:
                        thisRes.FIDO_shidx[k] = np.where(thisRes.regions[k]==0)[0]
                    thisRes.FIDO_FRidx[k] = []
                    if 1 in thisRes.regions[k]:
                        thisRes.FIDO_FRidx[k] = np.where(thisRes.regions[k]==1)[0]
                    thisRes.FIDO_SWidx[k] = np.where(np.abs(thisRes.regions[k]-100)<10)[0]
            
                    # derived paramters
                    thisRes.FIDO_FRdur[k] = (thisRes.FIDOtimes[k][thisRes.FIDO_FRidx[k][-1]] - thisRes.FIDOtimes[k][thisRes.FIDO_FRidx[k][0]]) * 24
                    if len(thisRes.FIDO_shidx[k]) !=0 :
                        thisRes.FIDO_shdur[k] = (thisRes.FIDOtimes[k][thisRes.FIDO_shidx[k][-1]] - thisRes.FIDOtimes[k][thisRes.FIDO_shidx[k][0]]) * 24
                    else:
                        thisRes.FIDO_shdur[k] = 0.
                    thisRes.FIDO_FRexp[k] = 0.5*(thisRes.FIDOvs[k][thisRes.FIDO_FRidx[k][0]] - thisRes.FIDOvs[k][thisRes.FIDO_FRidx[k][-1]]) 
                    
                    # get corresponding idxs for ANT data
                    if OSP.doANT:
                        if OSP.doPUP and (len(thisRes.FIDO_shidx[k]) !=0):
                            tSH =thisRes.FIDOtimes[k][thisRes.FIDO_shidx[k][0]], 
                            thisRes.ANTshidx[k] = np.min(np.where(thisRes.ANTtimes >= tSH))
                        tFR = thisRes.FIDOtimes[k][thisRes.FIDO_FRidx[k][0]]
                        thisRes.ANTFRidx[k] = np.min(np.where(thisRes.ANTtimes >= tFR))
                        
                        # redo calc Kp with actual front
                        dphidt = np.power(thisRes.ANTvFs[thisRes.ANTFRidx[k]], 4/3.) * np.power(np.abs(thisRes.ANTBpols[thisRes.ANTFRidx[k]]), 2./3.) 
                        # Mays/Savani expression, best behaved for high Kp
                        thisRes.ANTKp0[k] = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
                
                        # reset ANT dur with more accurate version
                        thisRes.ANTdur[k] = thisRes.FIDO_FRdur[k]

                    Bvec = [thisRes.FIDOBxs[k], thisRes.FIDOBys[k], thisRes.FIDOBzs[k]]
                    Kp, BoutGSM   = calcKp(Bvec, DoY, thisRes.FIDOvs[k]) 
                    thisRes.FIDOKps[k]   = Kp
                    
                    if (OSP.doPUP) and (i in SITids[k]):  
                        thisRes.hasSheath[k] = True
                        myID = np.where(SITids[k] == i)[0][0]
                        thisRes.SITidx[k] = np.where(thisRes.regions[k]==0)[0]
                        thisRes.SITdur[k] = SITdata[k][myID, 1]
                        thisRes.SITcomp[k] = SITdata[k][myID, 2]
                        thisRes.SITMach[k] = SITdata[k][myID, 3]
                        thisRes.SITn[k]    = SITdata[k][myID, 4]
                        thisRes.SITvSheath[k] = SITdata[k][myID, 5]
                        thisRes.SITB[k]    = SITdata[k][myID, 6]     
                        thisRes.SITvShock[k] = SITdata[k][myID,7]    
                        thisRes.SITtempk = SITdata[k][myID,8]
                        if len(thisRes.FIDO_shidx[k]) !=0:
                            thisRes.SITminBz[k] = np.min(thisRes.FIDOBzs[k][thisRes.SITidx[k]])
                            thisRes.SITmaxB[k] = np.max(thisRes.FIDOBs[k][thisRes.SITidx[k]])
                            thisRes.SITmaxKp[k] = np.max(thisRes.FIDOKps[k][thisRes.SITidx[k]])
                ResArr[i] = thisRes
                

    # if haven't run FC may have fewer CMEs in ResArr than total runs if have misses
    for j in range(OSP.nRuns):
        if j not in ResArr.keys():
            thisRes = EnsRes(OSP.thisName)
            thisRes.FIDOmiss = [True for iii in range(nSat)]
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
        myOrder = ['CMElat', 'CMElon', 'CMEtilt', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEdelCSAx', 'CMEr', 'FCrmax', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'FRB',  'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWT', 'SWcs', 'SWvA', 'FRB', 'FRtau', 'FRCnm', 'FRT', 'Gamma', 'IVDf', 'IVDf1', 'IVDf2', 'MHarea', 'MHdist']  
        varied = sorted(varied, key=lambda x: myOrder.index(x))    
        
    return ResArr, nSat, hitsSat, nFails, DoY, dObj
    
# |----- Mini script to determine Kp from other params -----|
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
    
# |--------------------------------------------------------------|
# |------- Read in the observational data from text files -------|
# |--------------------------------------------------------------|
def readInData(thisfile):
    dataIn = np.genfromtxt(thisfile, dtype=float)
    dataIn[np.where(dataIn == -9999)] = math.nan
    
    # Need to check if goes over into new year...
    base = datetime.datetime(int(dataIn[0,0]), 1, 1, 0, 0)
    obsDTs = np.array([base + datetime.timedelta(days=int(dataIn[i,1])-1, seconds=int(dataIn[i,2]*3600)) for i in range(len(dataIn[:,0]))])
    
    nGiven = len(dataIn[0,:])
    global hasv, hasKp
    hasv, hasKp, hasT = True, True, True
    if nGiven == 11:
        # have all the things (yr doy hr, B, Bx, By, Bz, n, v, T, Kp)
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,8], dataIn[:,9], dataIn[:,7], dataIn[:,10]/10.])
    elif nGiven == 10:
        # no Kp
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7], dataIn[:,8], dataIn[:,9]])
        hasKp = False
    elif nGiven ==8:
        # no v T or Kp
        dataOut = np.array([obsDTs, dataIn[:,3],  dataIn[:,4], dataIn[:,5], dataIn[:,6], dataIn[:,7]])
        hasv, hasKp, hasT = False, False, False       
    return dataOut


# |--------------------------------------------------------------|
# |---------- Wrapper for all obs processing functions ----------|
# |--------------------------------------------------------------|
def processObs(ResArr, nSat):
    # |----- Set up holders -----|
    ObsData = [None]
    satNames = [''] # default for single sat case with no data
    satLoc0 = [] # array of sat locations at start of simulation
    satLocI = [] # array of sat locations at time of impact (for each one)
    satLocAllI = [[] for i in range(nSat)]
    
    # |------Boring single sat case -----------------|
    # |----- Will overwrite if fancy single sat -----|
    if nSat == 1:
        ObsData = [readInData(OSP.ObsDataFile)]
        OSP.obsFRstart, OSP.obsFRend, OSP.obsShstart = [OSP.obsFRstart], [OSP.obsFRend], [OSP.obsShstart]
        satLoc0 = [OSP.satPos]
        satLocI = [OSP.satPos] # this assuming not moving, need to fix
    
    # |------- Pull in details for the fancy case ----------|
    # |------- Either traj path or multi sat (or both) -----|    
    if 'satPath' in OSP.input_values:
        satPath = OSP.input_values['satPath']
        satPaths = OSP.satPathWrapper(satPath,  checkSatlen=False)
        # |----- Multi sat mode -----|
        if (satPath[-4:] == 'sats'):
            satNames = []
            ObsData = [[None] for i in range(nSat)]
            temp = np.genfromtxt(satPath, dtype='unicode', delimiter=' ')
            # |----- Prep single sat using multi sat format -----|
            if nSat == 1:
                temp = [temp] #package in array to make loop happy
            OSP.obsFRstart, OSP.obsFRend, OSP.obsShstart = [[] for i in range(nSat)], [[] for i in range(nSat)], [[] for i in range(nSat)]
            # |----- Actual multi stat details -----|
            for i in range(nSat):
                # |----- Pull in sat names -----|
                if nSat !=1:
                    satNames.append(temp[i][0])
                else:
                    satNames.append(temp[i])
                # |----- Pull in data for each sat -----|    
                if len(temp[0]) >= 6:
                    ObsData[i] = readInData(temp[i][5])
                    hasObs = True
                # |----- Pull in boundary times -----|    
                if len(temp[0]) == 9:
                    OSP.obsFRstart[i] = float(temp[i][7])
                    OSP.obsFRend[i] = float(temp[i][8])
                    OSP.obsShstart[i] = float(temp[i][6])
                    
                # |----- Initial location -----|
                satLoc0.append([float(temp[i][1]), float(temp[i][2]), float(temp[i][3])])
                # |----- Location at time of impact ----|
                myt = ResArr[0].ANTtimes[ResArr[0].ANTFRidx[i]]*24*60 + ResArr[0].FCtimes[-1]
                if OSP.doFIDO:
                    myt = ResArr[0].FIDOtimes[i][ResArr[0].FIDO_shidx[i][0]]*24*60 + ResArr[0].FCtimes[-1]
                else:
                    myt = ResArr[0].FIDOtimes[i][ResArr[0].FIDO_FRidx[i][0]]*24*60 + ResArr[0].FCtimes[-1]
                    
                myImpLoc = [float(satPaths[i][0](myt*60)), float(satPaths[i][1](myt*60)), float(satPaths[i][2](myt*60))]
                satLocI.append(myImpLoc)     
                for j in range(nSat):
                    satLocAllI[i].append([float(satPaths[j][0](myt*60)), float(satPaths[j][1](myt*60)), float(satPaths[j][2](myt*60))])
        
    return ObsData, satNames, satLoc0, satLocI, satLocAllI
    


    