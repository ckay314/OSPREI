import numpy as np
import math
import sys
from scipy.special import jv
from scipy import integrate
import matplotlib
#matplotlib.use("TkAgg")  # OSPREI likes this commented out, need for GUI version
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
from Tkinter import *
from pylab import setp
import random 

global tmax, dt
tmax = 80 * 3600. # maximum time of observations
dt = 1 * 60. # time between spacecraft obs

# useful global variables
global rsun, dtor, radeg, kmRs
rsun  =  7e10		 # convert to cm, 0.34 V374Peg
dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
kmRs  = 1.0e5 / rsun # km (/s) divided by rsun (in cm)

#global rotspeed
#rotspeed = 1./ 3600. / 24 / 365. * 360 

# --------------------------------------------------------------------
# Geometry programs
# --------------------------------------------------------------------

def SPH2CART(sph_in):
    r = sph_in[0]
    colat = (90. - sph_in[1]) * dtor
    lon = sph_in[2] * dtor
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)
    return [x, y, z]

def CART2SPH(x_in):
# calcuate spherical coords from 3D cartesian
# output lat not colat
    r_out = np.sqrt(x_in[0]**2 + x_in[1]**2 + x_in[2]**2)
    colat = np.arccos(x_in[2] / r_out) * 57.29577951
    lon_out = np.arctan(x_in[1] / x_in[0]) * 57.29577951
    if lon_out < 0:
        if x_in[0] < 0:
            lon_out += 180.
        elif x_in[0] > 0:
            lon_out += 360. 
    elif lon_out > 0.:
	    if x_in[0] < 0:  lon_out += 180. 
    return [r_out, 90. - colat, lon_out]

def rotx(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the x-axis
    ang *= dtor
    yout = np.cos(ang) * vec[1] - np.sin(ang) * vec[2]
    zout = np.sin(ang) * vec[1] + np.cos(ang) * vec[2]
    return [vec[0], yout, zout]

def roty(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the y-axis
    ang *= dtor
    xout = np.cos(ang) * vec[0] + np.sin(ang) * vec[2]
    zout =-np.sin(ang) * vec[0] + np.cos(ang) * vec[2]
    return [xout, vec[1], zout]

def rotz(vec, ang):
# Rotate a 3D vector by ang (input in degrees) about the y-axis
	ang *= dtor
	xout = np.cos(ang) * vec[0] - np.sin(ang) * vec[1]
	yout = np.sin(ang) * vec[0] + np.cos(ang) * vec[1]
	return [xout, yout, vec[2]]



# New functions
def isinCME(vec_in, CME_shape):
    # Check and see if the requested point is actually in the CME and return
    # the cylindrical radial distance (from center of FR)
    # Function assumes vec_in is in CME Cartesian coord sys
    thetas = np.linspace(-math.pi/2, math.pi/2, 1001)
    # determine the xz positions of the rope axis
    xFR = CME_shape[0] + CME_shape[1] * np.cos(thetas)
    zFR = CME_shape[3] * np.sin(thetas)
    dists2 = (vec_in[0] - xFR)**2 + vec_in[1]**2 + (vec_in[2] - zFR)**2
    myb2 = np.min(dists2)
    minidxs = np.where(dists2 == myb2)
    # unwrap
    minidx = minidxs[0]
    temp = thetas[np.where(dists2 == myb2)]
    mythetaT = temp[0]
    # add a second iteration to refine B
    # see which side of minidx the actual minimum is on
    if len(minidx) == 1: # if perfectly symmetric can get two equi dists at back edge
        if minidx < len(dists2) - 1: # check to make sure not already at edge
            if dists2[minidx-1] < dists2[minidx+1]: startidx = minidx - 1
            else:  startidx = minidx + 1
    	# repeat the same procedure on the side with the acutal min
            if dists2[minidx-1] != dists2[minidx+1]:
                thetas2 = np.linspace(thetas[startidx], thetas[minidx], 101)
                xFR2 = CME_shape[0] + CME_shape[1] * np.cos(thetas2)
                zFR2 = CME_shape[3] * np.sin(thetas2)
                dists2 = (vec_in[0] - xFR2)**2 + vec_in[1]**2 + (vec_in[2] - zFR2)**2
                myb2 = np.min(dists2)
                minidxs = np.where(dists2 == myb2)
                minidx = minidxs[0]
                temp = thetas2[np.where(dists2 == myb2)]
                mythetaT = temp[0]
    myb = np.sqrt(myb2)
    CME_crossrad = CME_shape[2]
    if (myb > CME_crossrad):
        #print CME_shape[2]+CME_shape[1]+CME_shape[0], myb/CME_crossrad, vec_in
        myb = -9999.
        return myb, -9999, -9999., -9999, -9999
    else:
	# thetaP should swing quickly at closest approach to FR
        mythetaP = np.arcsin(vec_in[1] / myb)
        origTP = mythetaP
        
	# doesn't run through second loop sometimes
        if ('xFR2' not in locals()) and ('xFR2' not in globals()):
            xFR2 = xFR
	# convert thetaP to complement
        # had a random case that blew up here -> added try/except
        try: 
            if vec_in[0] < (xFR2[minidx] ):	
                if vec_in[1] > 0: mythetaP = math.pi - mythetaP
                else: mythetaP = -math.pi - mythetaP
            # shut down if more than 90 from nose (behind what normal consider to be the torus)
            # with out this, will add rounded hemispheres at back of torus legs essentially
            if np.abs(mythetaT) > 3.14159/2.:
                return -9999, -9999, -9999., -9999, -9999    
        except:
            pass
    return myb, mythetaT, mythetaP, 0, CME_crossrad

def getBvector(CME_shape, minb, thetaT, thetaP):
    tdir = np.array([-(CME_shape[1] + minb * np.cos(thetaP)) * np.sin(thetaT), 0., (CME_shape[3] + minb * np.cos(thetaP)) * np.cos(thetaT)])
    pdir = np.array([-minb * np.sin(thetaP) * np.cos(thetaT), minb * np.cos(thetaP), -minb * np.sin(thetaP) * np.sin(thetaT)])
    tmag = np.sqrt(np.sum(tdir**2))
    pmag = np.sqrt(np.sum(pdir**2))
    tdir = tdir / tmag
    pdir = pdir / pmag
    return tdir, pdir

def update_insitu(inps):
    # unpack the params
    FFlat, FFlon0, CMElat, CMElon, CMEtilt, CMEAW, CMESRA, CMESRB, CMEvr, CMEB0, CMEH, tshift, CMEstart, CMEend, vExp, FFr, rotspeed = inps[0], inps[1], inps[2], inps[3], inps[4], inps[5], inps[6], inps[7], inps[8], inps[9], inps[10], inps[11], inps[12], inps[13], inps[14], inps[15], inps[16]
    
    # determine the CME shape
    CME_shape = np.zeros(4)
    # set up CME shape as [d, a, b, c]
    shapeC = np.tan(CMEAW*dtor) / (1. + CMESRB + np.tan(CMEAW*dtor) * (CMESRA + CMESRB)) 
    
    #if np.abs(CMEtilt) < 1e-3: 
    #    CMEtilt = 1e-3
    dtorang = (CMElon - FFlon0) / np.sin((90.-CMEtilt) * dtor) * dtor
    CMEnose = 0.999*FFr # start just in front of sat dist, can't hit any closer 
    CME_shape[3] = CMEnose * shapeC
    CME_shape[1] = CME_shape[3] * CMESRA
    CME_shape[2] = CME_shape[3] * CMESRB 
    CME_shape[0] = CMEnose - CME_shape[1] - CME_shape[2]
    
    # set up arrays to hold values over the spacecraft path
    obsBx = []
    obsBy = []
    obsBz = []
    tARR = []
    rCME = []
    radfrac = []
    
    # reset time and other vars to initial values (necessary anymore?)   
    t = 0.
    CMEB = CMEB0
    FFlon = FFlon0
    
    # set various switches/flags
    thetaPprev = -42
    switch = 0
    flagExp = False
    # Imp Param can be used to keep track of closest approach
    ImpParam = 9999.
    # start simulation
    while t < tmax:  
	    # convert to FIDO position to Cartesian in loop to include Elon change
	    # get Sun xyz position
        FF_sunxyz = SPH2CART([FFr, FFlat, FFlon])
        # rotate to CMEcoord system
        temp = rotz(FF_sunxyz, -CMElon)
        temp2 = roty(temp, CMElat)
        FF_CMExyz = rotx(temp2, (90.-CMEtilt))
        
        # calculate CME shape (self-simlar unless flagged off)
        if flagExp == False:
            CME_shape[3] = CMEnose * np.tan(CMEAW*dtor) / (1. + CMESRB + np.tan(CMEAW*dtor) * (CMESRA + CMESRB))
            CME_shape[1] = CME_shape[3] * CMESRA
            CME_shape[2] = CME_shape[3] * CMESRB 
        elif expansion_model == 'vExp':
            CMEnose += vExp * dt/7e5
            CME_shape[2] += vExp * dt/7e5
        CME_shape[0] = CMEnose - CME_shape[1] - CME_shape[2]
	    
        # check to see if the flyer is currently in the CME
	    # if so, get its position in CME torus coords 
        minb, thetaT, thetaP, flagit, CME_crossrad = isinCME(FF_CMExyz, CME_shape)
        if np.abs(minb/CME_crossrad) < ImpParam: ImpParam = np.abs(minb/CME_crossrad)
        #print CMEnose, CME_crossrad, minb/CME_crossrad, CMEB
        # need to check that thetaP doesn't jump as it occasionally will
        if flagit != -9999:
            #print CMEnose, FFlat, FFlon, CME_shape
	        # if not expanding self similar flag it to stop after first contact
            if expansion_model != 'Self-Similar': flagExp = True
            # get the toroidal and poloidal magnetic field
            Btor = CMEB * jv(0, 2.4 * minb / CME_crossrad)
            Bpol = CMEB * CMEH * jv(1, 2.4 * minb / CME_crossrad)
	        # convert this into CME Cartesian coordinates
            tdir, pdir = getBvector(CME_shape, minb, thetaT, thetaP)
            Bt_vec = Btor * tdir
            Bp_vec = Bpol * pdir
            Btot = Bt_vec + Bp_vec
            # convert to spacecraft coord system
            temp = rotx(Btot, -(90.-CMEtilt))
            temp2 = roty(temp, CMElat - FFlat) 
            BSC = rotz(temp2, CMElon - FFlon)
            obsBx.append(-BSC[0])
            obsBy.append(-BSC[1])
            obsBz.append(BSC[2])
            tARR.append(t/3600.)
            rCME.append(CMEnose)
            radfrac.append(minb/CME_crossrad)
                
        else:
            # stop checking if exit CME
            if thetaPprev != -42: t = tmax+1

        # move to next step in simulation
        t += dt
	    # CME nose moves to new position
        CMEnose += CMEvr * dt / 7e5 # change position in Rsun
	    # update the total magnetic field
        if flagExp == False:
            CMEB *= ((CMEnose - CMEvr * dt / 7e5) / CMEnose)**2
        elif expansion_model == 'vExp':
            CMEB *= ((CME_shape[2] - vExp * dt / 7e5) / CME_shape[2])**2
	    # determine new lon of observer
        FFlon += dt * rotspeed
        
    # clean up the result and package to return
    obsBx, obsBy, obsBz, tARR = np.array(obsBx), np.array(obsBy), np.array(obsBz), np.array(tARR)
    obsB = np.sqrt(obsBx**2 + obsBy**2 + obsBz**2)
    zerotime = (CMEstart + tshift/24.) # in days
    # try to set tARR to start at time 0, if fails then know a miss
    try:
        tARR = (tARR-tARR[0])/24. # also in days
        tARR += zerotime
        isHit = True
    except:
        isHit = False
    Bout = np.array([obsBx, obsBy, obsBz, obsB])   
    return Bout, tARR, isHit, ImpParam, np.array(radfrac)

def radfrac2vprofile(radfrac, vAvg, vExp):
    # take the position within CME profile and add expansion to vAvg
    # this is correct if path through is diameter, not exact for shorter
    # cut through, how to use angle btwn trajectory and impact?
    newfrac = radfrac - np.min(radfrac)
    newfrac = newfrac/np.max(newfrac)
    centID = np.where(newfrac == 0)[0]
    centID = centID[0]
    vProf = vAvg + vExp*newfrac
    vProf[centID:] = vAvg - vExp*newfrac[centID:]
    return vProf

def reScale(Bout, tARR, CMEstart, CMEend):  
    global d_Btot     
    obsB = Bout[3] 
    CMEmid    = 0.5 * (CMEstart + CMEend)
    avg_obs_B = np.mean(d_Btot[np.where(np.abs(d_tUN - CMEmid) < 2./24.)])
    cent_idx = np.where(np.abs(tARR - CMEmid) < 2./24.)[0] 
    #scale = 1 
    if len(cent_idx) > 0: 
        avg_FIDO_B = np.mean(obsB[cent_idx])
        scale = avg_obs_B / avg_FIDO_B
        Bout = Bout * scale
    else:
        print('CME too short to autonormalize, reverting to B0')
    
    #obsBy *= scale
    #obsBz *= scale
    #obsB = np.sqrt(obsBx**2 + obsBy**2 + obsBz**2)                                       
    return Bout

def pullGUIvals():
    # set equal to the original values
    inps = inps0
    
    # check if expansion changed
    global expansion_model
    expansionToggle = expansionToggleVAR.get()
    expansion_model = 'None'
    if expansionToggle == 0: expansion_model = 'Self-Similar'
    elif expansionToggle ==2: expansion_model = 'vExp'
    
    # check if Autonorm changed
    global Autonormalize
    Autonormalize = False
    if autonormVAR.get() == 1: Autonormalize = True
    
    # pull inp params
    for i in range(14):
        try: 
            inps[i] = float(wids[i].get())
        except:
            pass
    # need to check that CME range isn't outside data range
    # can still plot, but no longer score
    flagged = False
    if ISfilename != False:
        if inps[12] < minISdate:
            print('CME_start before first in situ time')
            flagged = True
        if inps[13] > maxISdate:
            print('CME_end after last in situ time')
            flagged = True
    if doAsym:
            global DiP
            DiP = float(wids[15].get())
    if expansion_model == 'vExp':
            inps[14] = float(wids[14].get())
                    
    return inps, flagged
        
def hourify(tARR, vecin):
    # assume input is in days, will spit out results in hourly averages
    # can conveniently change resolution by multiplying input tARR
    # ex. 10*tARR -> 1/10 hr resolution
    newt = (tARR-tARR[0])*24.
    maxt = int((newt[-1]))
    vecout = np.zeros(maxt+1)
    for i in range(maxt):
        ishere = len(newt[abs(newt - (i+.5))  <= 0.5]) > 0
        if ishere:
            vecout[i+1] =  np.mean(vecin[abs(newt - (i+.5))  <= 0.5])
    # keep the first point, important for precise timing
    vecout[0] = vecin[0]
    return vecout

def calc_score(Bout, tARR, CMEstart, CMEend):

    #ACE_t = (d_tUN - CMEstart) * 24
    FIDO_hr = (tARR - tARR[0]) *24
    # unpack Bs
    obsBx = Bout[0]
    obsBy = Bout[1]
    obsBz = Bout[2]
    
    # covert to hourly averages
    FIDO_hrt = hourify(tARR-CMEstart, FIDO_hr)
    FIDO_hrBx = hourify(tARR-CMEstart, obsBx)
    FIDO_hrBy = hourify(tARR-CMEstart, obsBy)
    FIDO_hrBz = hourify(tARR-CMEstart, obsBz)
    ACE_hrBx = hourify(d_tUN-CMEstart, d_Bx)
    ACE_hrBy = hourify(d_tUN-CMEstart, d_By)
    ACE_hrBz = hourify(d_tUN-CMEstart, d_Bz)
    
    # take same portion of ACE as have for CME
    ACE_hrBx = ACE_hrBx[:len(FIDO_hrBx)]
    ACE_hrBy = ACE_hrBy[:len(FIDO_hrBx)]
    ACE_hrBz = ACE_hrBz[:len(FIDO_hrBx)]    
    
    # determine total avg B at hourly intervals
    ACE_hrB = np.sqrt(ACE_hrBx**2 + ACE_hrBy**2 + ACE_hrBz**2)
    errX = np.abs((FIDO_hrBx[np.where(ACE_hrBx!=0)] - ACE_hrBx[np.where(ACE_hrBx!=0)])) / np.mean(ACE_hrB[np.where(ACE_hrBx!=0)])
    errY = np.abs((FIDO_hrBy[np.where(ACE_hrBx!=0)] - ACE_hrBy[np.where(ACE_hrBx!=0)])) / np.mean(ACE_hrB[np.where(ACE_hrBx!=0)])
    errZ = np.abs((FIDO_hrBz[np.where(ACE_hrBx!=0)] - ACE_hrBz[np.where(ACE_hrBx!=0)])) / np.mean(ACE_hrB[np.where(ACE_hrBx!=0)])
    scoreBx = np.mean(errX)
    scoreBy = np.mean(errY)
    scoreBz = np.mean(errZ)   

    totalscore = np.mean(np.sqrt(errX**2+errY**2+errZ**2))

    if tARR[-1] < CMEend - .5/24.: totalscore += 5.
    overamt = tARR[-1] - CMEend
    if overamt > 1/24.: totalscore += 0.1 * (24 * overamt - 1) 
    
    scores = [scoreBx, scoreBy, scoreBz, totalscore]
    
    return scores

def calc_indices(Bout, CMEstart, CMEv):
    fracyear = CMEstart / 365.
    rotang = 23.856 * np.sin(6.289 * fracyear + 0.181) + 8.848
    GSMBx = []
    GSMBy = []
    GSMBz = []
    #print('Rotating by '+str(rotang)+' to GSM')
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
    
def update_fig(Bout, tARR, scores, axes, hasData, isHit, CMEstart, CMEend):
    #Bobs = [d_Btot, d_Bx, d_By, d_Bz]
    #Bsim = [Bout[3], -Bout[0], -Bout[1], Bout[2]]
    for ax in axes: ax.clear()
    mins = [9999,9999,9999,9999]
    maxs = [-9999,-9999,-9999,-9999]
    if hasData:
        Bobs = [d_Btot, d_Bx, d_By, d_Bz]
        for i in range(4):
            axes[i].plot(d_tUN, Bobs[i], '#696969', linewidth=4)
            mins[i] = np.min(Bobs[i])
            maxs[i] = np.max(Bobs[i])
            plotstart, plotend = np.min(d_tUN), np.max(d_tUN)
    cols = ['k', 'k', 'b','r']
    if isHit:
        Bsim = [Bout[3], Bout[0], Bout[1], Bout[2]]
        for i in range(4):
            axes[i].plot(tARR, Bsim[i], color=cols[i], linewidth=4)
            if mins[i] > np.min(Bsim[i]): mins[i] = np.min(Bsim[i])
            if maxs[i] < np.max(Bsim[i]): maxs[i] = np.max(Bsim[i])
            if not hasData:
                plotstart, plotend = np.min(tARR)-0.2, np.max(tARR)+0.2
    # put B mag min back at zero and set ranges to sym about zero
    mins[0] = 0 
    for i in range(3):
        biggest = np.max([np.abs(mins[i+1]), maxs[i+1]])
        mins[i+1], maxs[i+1] = -biggest, biggest
    
    # add start/stop lines and set figure limits
    scl = 1.25
    for i in range(4):  
        if (ISfilename != False):
            axes[i].plot([CMEstart, CMEstart], [1.3*mins[i], 1.3*maxs[i]], 'k--', linewidth=2)
            axes[i].plot([CMEend, CMEend], [1.3*mins[i], 1.3*maxs[i]], 'k--', linewidth=2)
        axes[i].set_ylim([scl*mins[i], scl*maxs[i]])
    try:
        axes[0].set_xlim([plotstart, plotend])
    except:
        axes[0].set_xlim([0,1]) # will fail if have no CME or background data
    
    # add scores on the figure  
    if (scores[0] != 9999) and (plotScores):  
        axes[0].annotate('%0.2f'%(scores[3]), xy=(1, 0), color='r', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')    
        axes[1].annotate('%0.2f'%(scores[0]), xy=(1, 0), color='r', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')
        axes[2].annotate('%0.2f'%(scores[1]), xy=(1, 0), color='r', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')    
        axes[3].annotate('%0.2f'%(scores[2]), xy=(1, 0), color='r', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')    
        
    # Labels
    axes[0].set_ylabel('B (nT)')
    setp(axes[0].get_xticklabels(), visible=False)    
    axes[1].set_ylabel('B$_x$ (nT)')
    setp(axes[1].get_xticklabels(), visible=False)
    axes[2].set_ylabel('B$_y$ (nT)')
    setp(axes[2].get_xticklabels(), visible=False)
    axes[3].set_ylabel('B$_z$ (nT)')
    plt.subplots_adjust(right=0.8, wspace=0.001, hspace=0.25, bottom=0.12)
    if not show_indices: axes[3].set_xlabel('Day of Year')
    
def plot_Kp(BoutGSM, Kp, tARR, axes, CMEstart, CMEend):
    #axes[1].plot(tARR, BoutGSM[0], 'b--', linewidth=4, zorder=0)
    #axes[2].plot(tARR, BoutGSM[1], 'b--', linewidth=4, zorder=0)
    #axes[3].plot(tARR, BoutGSM[2], 'b--', linewidth=4, zorder=0)
    axes[4].plot(tARR, Kp, 'g', linewidth=4)
    axes[4].plot([CMEstart, CMEstart], [0, 9999], 'k--', linewidth=2)
    axes[4].plot([CMEend, CMEend], [0, 9999], 'k--', linewidth=2)
    axes[4].set_ylim([0, 10])
    plt.subplots_adjust(hspace=0.05)
    setp(axes[3].get_xticklabels(), visible=False)
    axes[4].set_xlabel('Day of Year')
    axes[4].set_ylabel('Kp Index')
    
def plotObsKp(Kpdate, obsKp, axes):
    axes[4].plot(Kpdate, obsKp, 'k', linewidth=3)

def finish_plot():
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.gcf().canvas.draw()
    
def calc_jump(CMEstart, shinps):
    tobs = shinps[0]
    tSh = tobs#CMEstart-shinps[1]/24.
    # determine the upstream magnitude
    global upDur
    upDur = 2/24. # how long upstream to used to determine B?
    sheathidxs = np.where(np.abs(d_tUN - tSh+upDur/2)<upDur/2.)
    upB = np.array([np.mean(d_Btot[sheathidxs]), np.mean(d_Bx[sheathidxs]), np.mean(d_By[sheathidxs]), np.mean(d_Bz[sheathidxs])])
    compB = upB * shinps[2]
    #compB[1] = upB[1] # set Bx to no increase
    # reset full mag b/c get slight diffs from using means
    compB[0] = np.sqrt(compB[1]**2 + compB[2]**2 + compB[3]**2)
    return [[upB[i],compB[i]] for i in range(4)]

def make_sheathOLD(jumpvec, Bout, CMEstart, shinps):
    # determine initial flux rope vector
    iFRvec = [Bout[3][0], Bout[0][0], Bout[1][0], Bout[2][0]]
    tSheath = np.linspace(CMEstart-shinps[1]/24.,CMEstart,20)
    Bsheath = [[],[],[],[]]
    #print iFRvec/iFRvec[0]
    #print jumpvec[1][1]/jumpvec[0][1], jumpvec[2][1]/jumpvec[0][1], jumpvec[3][1]/jumpvec[0][1]
    for i in range(4):
        slope = (iFRvec[i]-jumpvec[i][1])/(tSheath[-1]-tSheath[0])
        Bsheath[i] = jumpvec[i][1]+slope*(tSheath-tSheath[0]) 
    Bsheath[0] = np.sqrt(Bsheath[1]**2+Bsheath[2]**2+Bsheath[3]**2)
    return tSheath, Bsheath

def make_sheath(jumpvec, Bout, CMEstart, shinps):
    tSheath = np.linspace(CMEstart-shinps[1]/24.,CMEstart,20)
    BUsheath = [[],[],[],[]]
    iFRvec = [Bout[3][0], Bout[0][0], Bout[1][0], Bout[2][0]]
    iShvec = np.array([jumpvec[0][1],jumpvec[1][1],jumpvec[2][1],jumpvec[3][1]])
    # calculate unit vectors
    iShUvec = iShvec / iShvec[0]
    iFRUvec = iFRvec / iFRvec[0]
    for i in range(4):
        slope = (iFRUvec[i]-iShUvec[i])/(tSheath[-1]-tSheath[0])
        BUsheath[i] = iShUvec[i]+slope*(tSheath-tSheath[0]) 
    BUsheath[0] = np.sqrt(BUsheath[1]**2 + BUsheath[2]**2 + BUsheath[3]**2)
    BUsheath = np.array(BUsheath)
    # want the unit vec to maintain mag of 1, need to scale it
    scl = 1./ BUsheath[0]
    Bsheath = BUsheath*scl
    # get the total B vector
    Bslope = (iFRvec[0]-iShvec[0])/(tSheath[-1]-tSheath[0])
    Bmag = iShvec[0] + Bslope*(tSheath-tSheath[0]) 
    Bsheath *= Bmag
    #print Bsheath
    return tSheath, Bsheath
     
def plot_sheath(shinps, tsheath, sheathB, axes, scores):
    for i in range(4):
        # plot arbitrary big enough y values since use lim elsewhere
        axes[i].plot([shinps[0],shinps[0]],[-300,300],'k--', linewidth=2)
        axes[i].plot(tsheath,sheathB[i],'b', linewidth=4)
    if show_indices:
        axes[4].plot([shinps[0],shinps[0]],[-300,300],'k--', linewidth=2)
    if plotScores:
        axes[0].annotate('%0.2f'%(scores[3]), xy=(.1, 0), color='b', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')    
        axes[1].annotate('%0.2f'%(scores[0]), xy=(.1, 0), color='b', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')
        axes[2].annotate('%0.2f'%(scores[1]), xy=(.1, 0), color='b', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')    
        axes[3].annotate('%0.2f'%(scores[2]), xy=(.1, 0), color='b', xycoords='axes fraction', fontsize=16, horizontalalignment='right', verticalalignment='bottom')
    
def addAsym(Bout, tARR, ImpParam):
    # impact paramter
    b = ImpParam # for now, need to take as input
    
    # define flex point in 0 to 1 coords
    tc = 0.4 # assuming this for now
    
    # determine where flex is in tARR
    trange = tARR[-1] - tARR[0]
    flext = tARR[0] + tc * trange
    flexID = np.max(np.where(tARR < flext))
    
    # function for B/B0 as function of t in 0 to 1
    alpha = lambda x: np.sqrt(jv(0,2.4*np.sqrt(4*(1-b**2)*(x-0.5)**2+b**2))**2 + jv(1,2.4*np.sqrt(4*(1-b**2)*(x-0.5)**2+b**2))**2)
    
    # calculate slope of linear front portion
    atc = alpha(tc)
    intatc1 = integrate.quad(alpha,tc,1)[0]
    if DiP <= tc:
        mB = (-4*atc*DiP + 2*atc*tc + 2*intatc1)/(4*tc*DiP - 2*DiP**2 -tc**2)
    else:
        intatcDiP = integrate.quad(alpha,tc,DiP)[0]
        mB = (2./tc**2) * (intatc1 -2*intatcDiP - tc*atc)

    # get estimate of B0
    estB0 = Bout[3][flexID]/atc

    # get unit vector for front of CME
    unit_vec = np.transpose(np.array([Bout[0],Bout[1],Bout[2]])/Bout[3])
    
    # readjust magnitude of front of CME
    Bmag = Bout[3]
    Bmag[:flexID+1] = (tc-(tARR[:flexID+1]-tARR[0])/trange) * mB * estB0 + Bmag[flexID]
    Bout[3] = Bmag
    # adjust vector components to new mag, no change in dir
    Bout[:-1] = np.transpose(unit_vec * Bmag.reshape([-1,1]))
        
    return Bout

def run_case(inps, shinps):
    CMEstart = inps[12]
    CMEend = inps[13]
    
    global Bout, tARR, Kp, tsheath, sheathB, sheathKp
    # set up dummys in case not actually calculated
    Bout = []
    Kp = []
    tsheath = []
    sheathB = []
    sheathKp = []
    scores = [9999, 9999, 9999, 9999]
    # run the simulation    
    Bout, tARR, isHit, ImpParam, radfrac = update_insitu(inps)
    # execute extra functions as needed
    if isHit:
        if doAsym: 
                Bout = addAsym(Bout, tARR, ImpParam)            
        
        if ISfilename !=False:
            # Autonormalize
            if Autonormalize==True:
                Bout = reScale(Bout, tARR, inps[12], inps[13])
            # Calculate score
            scores = calc_score(Bout, tARR, CMEstart, CMEend)
            if canprint: print('score '+ str(scores[3])) 
            # measure DiP
            measure_DiP(Bout[3])
        # add sheath if desired
        if hasSheath:
            jumpvec = calc_jump(CMEstart, shinps)
            tsheath, sheathB = make_sheath(jumpvec, Bout, CMEstart, shinps)  
            # add tshift to sheath
            tsheath += inps[11]/24.
            sheath_scores = calc_score([sheathB[1],sheathB[2],sheathB[3]], tsheath, CMEstart-shinps[1]/24.,CMEstart)  
            print sheath_scores
        # Calculate Kp
        if show_indices:
            if hasSheath:
                sheathKp, sheathBoutGSM = calc_indices([sheathB[1],sheathB[2],sheathB[3],sheathB[0]], CMEstart-shinps[1]/24., shinps[3])
            Kp, BoutGSM = calc_indices(Bout, CMEstart, inps[8])
    else:
        if canprint: print ('No impact expected')
        scores = [9999, 9999, 9999, 9999]
    
    if not NoPlot:
        update_fig(Bout, tARR, scores, axes, canScore, isHit, CMEstart, CMEend)
        if isHit:
            if show_indices:
                if Kpfilename != False:
                    plotObsKp(Kpdate, obsKp, axes)
                plot_Kp(BoutGSM, Kp, tARR, axes, CMEstart, CMEend) 
                if hasSheath:
                      plot_Kp(sheathBoutGSM, sheathKp, tsheath, axes, shinps[0], CMEstart) 
            if hasSheath:
                plot_sheath(shinps, tsheath, sheathB, axes, sheath_scores) 
        finish_plot()  
        
    return Bout, tARR, radfrac

def save_plot(inps, Bout, tARR, shinps, tsheath, Bsheath, sheathKp):
    # unpack the params
    FFlat, FFlon0, CMElat, CMElon, origCMEtilt, CMEAW, CMESRA, CMESRB, CMEvr, CMEB0, CMEH, tshift, CMEstart, CMEend, vExp, Sat_rad, Sat_rot = inps[0], inps[1], inps[2], inps[3], inps[4], inps[5], inps[6], inps[7], inps[8], inps[9], inps[10], inps[11], inps[12], inps[13], inps[14], inps[15], inps[16]
    if canprint: print('saving as '+my_name)
    plt.savefig(my_name+'.png')
    f1 = open(my_name+'.txt', 'w')
    if ISfilename != False: 
        f1.write('insitufile: '+ISfilename+' \n')
    f1.write('%-13s %8.2f \n' % ('CME_lat: ', CMElat))
    f1.write('%-13s %8.2f \n' % ('CME_lon: ', CMElon))
    f1.write('%-13s %8.2f \n' % ('CME_tilt: ', origCMEtilt))
    f1.write('%-13s %8.2f \n' % ('CME_AW: ', CMEAW))
    f1.write('%-13s %8.2f \n' % ('CME_Ashape: ', CMESRA))
    f1.write('%-13s %8.4f \n' % ('CME_Bshape: ', CMESRB))
    f1.write('%-13s %8.2f \n' % ('CME_v1AU: ', CMEvr))
    f1.write('%-13s %8.2f \n' % ('FR_B0: ', CMEB0))
    f1.write('%-13s %8.2f \n' % ('FR_pol: ', CMEH))
    f1.write('%-13s %8.2f \n' % ('Sat_lat: ', FFlat))
    f1.write('%-13s %8.2f \n' % ('Sat_lon: ', FFlon0))
    f1.write('%-13s %8.2f \n' % ('Sat_rad: ', Sat_rad))
    f1.write('%-13s %8.2f \n' % ('Sat_rot: ', Sat_rot))
     # don't save the default values
    if (CMEstart != 0) and (CMEend !=1):
        f1.write('%-13s %8.2f \n' % ('CME_start: ', CMEstart))
        f1.write('%-13s %8.2f \n' % ('CME_stop: ', CMEend))
    f1.write('Launch_GUI: '+ str(Launch_GUI)+  '\n')
    f1.write('Autonormalize: '+ str(Autonormalize)+  '\n')
    f1.write('Save_Profile: '+ str(Save_Profile)+  '\n')
    f1.write('Expansion_Model: '+ expansion_model+  '\n')
    if expansion_model == 'vExp':
        f1.write('vExp: ' + str(vExp)+'\n')
    # don't print settings for extra features if not used
    # to avoid confusion about unnecessary things
    if 'tshift' in input_values:
        f1.write('%-13s %8.2f \n' % ('tshift: ', tshift))
    if 'PlotScores' in input_values:
        f1.write('PlotScores: ' + str(plotScores)+'\n')
    if 'No_Plot' in input_values:
        f1.write('No_Plot: '+ str(NoPlot)+  '\n')
    if 'Silent' in input_values:
        f1.write('Silent: '+ str(not canprint)+ '\n')
    if 'Indices' in input_values:
        f1.write('Indices: '+ str(show_indices)+'\n')
    if hasSheath:
        f1.write('Add_Sheath: True \n')
        f1.write('Sheath_start: ' + str(shinps[0]) + '\n')
        f1.write('Sheath_time: ' + str(shinps[1]) + '\n')
        f1.write('Compression: ' + str(shinps[2])+ '\n')
        f1.write('Sheath_v: ' + str(shinps[3])+ '\n')
        if Kpfilename != False:
            f1.write('ObsKpFile: ' + Kpfilename + '\n')
    else:
        if 'Add_Sheath' in input_values:
            f1.write('Add_Sheath: False' + '\n')
    if doAsym:
        f1.write('DiP: '+ str(DiP) +'\n')
    f1.close()
    if Save_Profile == True:
        if canprint: print('saving profile') 
        f1 = open(my_name+'.dat', 'w')
        if not show_indices:
            if hasSheath:
                for i in range(len(tsheath)):
                    f1.write('%10.5f %10.4f %10.4f %10.4f \n' % (tsheath[i], sheathB[1][i], sheathB[2][i], sheathB[3][i]))
            for i in range(len(tARR)):
                f1.write('%10.5f %10.4f %10.4f %10.4f \n' % (tARR[i], Bout[0][i], Bout[1][i], Bout[2][i]))
        else:
            if hasSheath:
                for i in range(len(tsheath)):
                    f1.write('%10.5f %10.4f %10.4f %10.4f  %10.4f \n' % (tsheath[i], sheathB[1][i], sheathB[2][i], sheathB[3][i], sheathKp[i]))    
            for i in range(len(tARR)):
                f1.write('%10.5f %10.4f %10.4f %10.4f  %10.4f \n' % (tARR[i], Bout[0][i], Bout[1][i], Bout[2][i], Kp[i]))
        f1.close()

def measure_DiP(Btot):
    totalB = np.sum(Btot)
    critB = 0.5 * totalB
    mylen = len(Btot)
    sumB = 0
    counter = 0
    while sumB <= critB:
        sumB += Btot[counter]
        counter += 1
    print 'DiP: ', float(counter)/mylen

def get_inputs(inputs):
    # take a file with unsorted input values and return a dictionary.
    # variable names have to match their names below
    possible_vars = ['insitufile', 'Sat_lat', 'Sat_lon', 'CME_lat', 'CME_lon', 'CME_tilt', 'CME_AW', 'CME_v1AU', 'tshift', 'CME_Ashape', 'CME_Bshape', 'FR_B0', 'FR_pol', 'CME_start', 'CME_stop', 'Autonormalize', 'Launch_GUI', 'Save_Profile', 'Expansion_Model', 'No_Plot', 'Silent', 'Indices', 'Add_Sheath', 'Sheath_start', 'Sheath_time', 'Compression', 'Sheath_v', 'ObsKpFile', 'PlotScores', 'DiP', 'CME_vExp', 'Sat_rad', 'Sat_rot']
    
    # if matches add to dictionary
    input_values = {}
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
    return input_values

def readinputfile():
    # Get the CME number
    global my_name
    if len(sys.argv) < 2: 
        #sys.exit("Need an input file")
        print('No input file, running without in situ data and starting with defaults')
        input_values = []
        my_name = 'temp'
    else:
        input_file = sys.argv[1]
        inputs = np.genfromtxt(input_file, dtype=str)
        my_name = input_file[:-4]
        input_values = get_inputs(inputs)
    return input_values
    

def read_more_inputs(inputs, input_values):
    possible_vars = ['FR_B0', 'FR_pol', 'CME_start', 'CME_stop', 'Expansion_Model', 'CME_vExp', 'CME_v1AU']
    # if matches add to dictionary
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
    return input_values
    

def setupOptions(input_values, silent=False):
    # these are all the things that set up how the simulation is run
    # but not specific inputs for an individual run
    
    # Print things to command line?
    global canprint 
    canprint = not silent
    if 'Silent' in input_values: 
        if input_values['Silent']=='True': canprint = False
    if canprint: print('Files will be saved as '), my_name

    # Pop up a GUI or just save a file
    global Launch_GUI
    Launch_GUI = True
    if 'Launch_GUI' in input_values:
       if input_values['Launch_GUI'] == 'False': Launch_GUI = False
       
    global NoPlot
    NoPlot = False
    if 'No_Plot' in input_values:
        if input_values['No_Plot'] == 'True': NoPlot = True
    # not allowed to launch a GUI without a plot
    if Launch_GUI == True: NoPlot = False
   
    # Option to print a file with simulation results
    global Save_Profile
    Save_Profile = False
    if 'Save_Profile' in input_values:
        if input_values['Save_Profile'] == 'True': Save_Profile = True
      
    # Autonormalizing magnitude
    global Autonormalize
    Autonormalize = False
    if 'Autonormalize' in input_values:
        if input_values['Autonormalize'] == 'True': Autonormalize = True
    
    # Pick expansion mode
    global expansion_model
    expansion_model = 'None'
    if 'Expansion_Model' in input_values:
        if input_values['Expansion_Model'] == 'None':
            expansion_model = 'None'
        elif input_values['Expansion_Model'] == 'Self-Similar':
            expansion_model = 'Self-Similar'        
        elif input_values['Expansion_Model'] == 'vExp':
            expansion_model = 'vExp'        
        else:
            sys.exit('Expansion_Model should be None, Self-Similar, or vExp')
        
    # Determine if we are plotting kp index
    global show_indices
    show_indices = False
    if 'Indices' in input_values:
        if input_values['Indices'] == 'True':
            show_indices = True
            
    # Determine if we are adding a sheath
    global hasSheath
    hasSheath = False
    if 'Add_Sheath' in input_values:
        if input_values['Add_Sheath'] == 'True':
            hasSheath = True
    
    # Determine if we want to add scores on the figure itself
    global plotScores
    plotScores = True
    if 'PlotScores' in input_values:
        if input_values['PlotScores'] == 'False':
            plotScores = False        
                
    # Get in situ filename
    global ISfilename, canScore
    ISfilename = False
    canScore = False
    if 'insitufile' in input_values:
        if input_values['insitufile'] != 'NONE':
            ISfilename = input_values['insitufile']
            canScore = True
            
    # Get Kp filename  
    global Kpfilename      
    Kpfilename = False
    if show_indices:
        if 'ObsKpFile' in input_values:
            Kpfilename = input_values['ObsKpFile']
            
    # Check if given a distortion parameter value
    # If so include asym
    global doAsym, DiP
    doAsym = False
    DiP = 0.5
    if 'DiP' in input_values:
        doAsym = True
        DiP = float(input_values['DiP'])
        # check the DiP in [0,1]
        if np.abs(DiP-0.5) > 0.5:
            doAsym = False
            if canprint: print('DiP must be between 0 and 1, cannot add asymmetry. ')
    
def getInps(input_values):
    # Set sim params to default, replace with any given values
    # order is FFlat [0], FFlon0 [1], CMElat [2], CMElon [3], CMEtilt [4], CMEAW [5]
    # CMESRA [6], CMESRB [7], CMEvr [8], CMEB0 [9], CMEH [10], tshift [11], start [12], end [13], vexp[14]
    # Sat_rad [15], Sat_rot[16]
    inps = [0.,0.,0.,0.,0.,45.,0.75,0.35,440.,25.,1,0.,0.,0, 0, 213, 360/365/24./3600.]
    print inps
    if 'Sat_lat' in input_values: inps[0] = float(input_values['Sat_lat'])
    if 'Sat_lon' in input_values: inps[1] = float(input_values['Sat_lon'])
    if 'CME_lat' in input_values: inps[2] = float(input_values['CME_lat']) 
    if 'CME_lon' in input_values: inps[3] = float(input_values['CME_lon'])
    if 'CME_tilt' in input_values: inps[4] = float(input_values['CME_tilt']) 
    if 'CME_AW' in input_values: inps[5] = float(input_values['CME_AW'])
    if 'CME_Ashape' in input_values: inps[6] = float(input_values['CME_Ashape'])
    if 'CME_Bshape' in input_values: inps[7] = float(input_values['CME_Bshape'])   
    if 'CME_v1AU' in input_values:inps[8] = float(input_values['CME_v1AU'])
    if 'tshift' in input_values: inps[11] = float(input_values['tshift'])
    if 'FR_B0' in input_values: inps[9] =float(input_values['FR_B0'])
    if "FR_pol" in input_values: 
        if input_values['FR_pol'][0] == '-': inps[10] = -1
    if 'CME_start' in input_values: inps[12] =float(input_values['CME_start'])
    if 'CME_stop' in input_values: inps[13] = float(input_values['CME_stop'])
    if 'CME_vExp' in input_values: inps[14] = float(input_values['CME_vExp'])
    if 'Sat_rad' in input_values: inps[15] = float(input_values['Sat_rad'])
    if 'Sat_rot' in input_values: inps[16] = float(input_values['Sat_rot'])/60.
    return inps
    

def getSheathInps(input_values):
    sheathTime = 11.9 # default in hours, avg of paper vales
    compression = 2.0 # no justifcation for this default right now
    sheathv     = 500. # again a random default
    sheathStart = float(input_values['CME_start']) - sheathTime/24.
    if 'Sheath_start' in input_values:
        sheathStart = float(input_values['Sheath_start'])
    if 'Sheath_time' in input_values:
        sheathTime = float(input_values['Sheath_time'])
    if 'Compression' in input_values:
        compression = float(input_values['Compression'])
    if 'Sheath_v' in input_values:
        sheathv = float(input_values['Sheath_v'])
    return [sheathStart, sheathTime, compression, sheathv]

def setupFigure():
    fig2 = plt.figure()
    # set up the panels of the figure depending on whether we 
    # want to show Kp or not
    if show_indices == False:
        ax2  = fig2.add_subplot(411)
        ax3  = fig2.add_subplot(412, sharex=ax2)
        ax4  = fig2.add_subplot(413, sharex=ax2)
        ax5  = fig2.add_subplot(414, sharex=ax2)
        axes = [ax2,ax3,ax4,ax5]
    else:
        ax2  = fig2.add_subplot(511)
        ax3  = fig2.add_subplot(512, sharex=ax2)
        ax4  = fig2.add_subplot(513, sharex=ax2)
        ax5  = fig2.add_subplot(514, sharex=ax2)
        ax6  = fig2.add_subplot(515, sharex=ax2)
        axes = [ax2,ax3,ax4,ax5,ax6]
        setp(ax5.get_xticklabels(), visible=False)
    
    setp(ax2.get_xticklabels(), visible=False)
    setp(ax3.get_xticklabels(), visible=False)
    setp(ax4.get_xticklabels(), visible=False)
    
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.subplots_adjust(right=0.8, wspace=0.001, hspace=0.0001)
    plt.tight_layout()
    return fig2, axes

def setupGUI(root, fig, axes, inps, shinps):
    # Add fig to GUI canvas
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=2, rowspan=30) #.grid(row=0,column=0)
    
    # need array to hold all the widgets so can pull from them later
    global wids
    wids = []

    # CME parameters
    Label(root, text='CME Parameters', bg='gray75').grid(column=0,row=0, columnspan=2)
    Label(root, text='CME Lat (deg):', bg='gray75').grid(column=0, row=1)
    e1 = Entry(root, width=10)
    e1.grid(column=1,row=1)
    Label(root, text='CME Lon (deg):', bg='gray75').grid(column=0, row=2)
    e2 = Entry(root, width=10)
    e2.grid(column=1, row=2)
    Label(root, text='Tilt from W (deg):', bg='gray75').grid(column=0, row=3)
    e3 = Entry(root, width=10)
    e3.grid(column=1, row=3)
    Label(root, text='Angular Width (deg):', bg='gray75').grid(column=0, row=4)
    e3b = Entry(root, width=10)
    e3b.grid(column=1, row=4)
    Label(root, text='CME vr (km/s):', bg='gray75').grid(column=0, row=5)
    e8 = Entry(root, width=10)
    e8.grid(column=1, row=5)


    Label(root, text='Time Shift (hr):', bg='gray75').grid(column=0, row=17)
    e9 = Entry(root, width=10)
    e9.grid(column=1, row=17)


    # Torus Parameters
    Label(root, text="Torus Shape Parameters", bg='gray75').grid(column=0, row=6, columnspan=2)
    Label(root, text='A:', bg='gray75').grid(column=0, row=7)
    e4 = Entry(root, width=7)
    e4.grid(column=1, row=7)
    Label(root, text='B:', bg='gray75').grid(column=0, row=8)
    e5 = Entry(root, width=7)
    e5.grid(column=1, row=8)

    # Flux rope params
    Label(root, text='Force Free Parameters', bg='gray75').grid(column=0, row=11, columnspan=2)
    Label(root, text='B0:', bg='gray75').grid(column=0, row=12)
    e6 = Entry(root, width=10)
    e6.grid(column=1, row=12)
    Label(root, text='Pol. Direction:', bg='gray75').grid(column=0, row=13)
    e7 = Entry(root, width=10)
    e7.grid(column=1, row=13)
    
    # add in dip control if using
    if doAsym:
        Label(root, text='DiP:', bg='gray75').grid(column=0, row=15)
        e7b = Entry(root, width=10)
        e7b.grid(column=1, row=15)
        

    # check button for autonormalizing magnitude
    global autonormVAR  
    autonormVAR = IntVar()
    autonormVAR.set(0)
    if Autonormalize: autonormVAR.set(1)
    Label(root, text='Autonormalize', bg='gray75').grid(column=3, row=0, columnspan=2)
    normCheck = Checkbutton(root, bg='gray75', var=autonormVAR).grid(column=3, row=1, columnspan=2)

    global expansionToggleVAR
    expansionToggleVAR = IntVar()
    Label(root, text='Expansion Profile:', bg='gray75').grid(row=2, column=3,columnspan=2)
    Label(root, text='Self-Similar', bg='gray75').grid(row=3, column=3,columnspan=1)
    Radiobutton(root, variable=expansionToggleVAR, value=0, bg='gray75').grid(column=3,row=4)
    Label(root, text='None', bg='gray75').grid(row=3, column=4,columnspan=1)
    Radiobutton(root, variable=expansionToggleVAR, value=1, bg='gray75').grid(column=4,row=4)
    Label(root, text='vExp', bg='gray75').grid(row=3, column=5,columnspan=1)
    Radiobutton(root, variable=expansionToggleVAR, value=2, bg='gray75').grid(column=5,row=4)
    expansionToggleVAR.set(2)
    if expansion_model == "Self-Similar": expansionToggleVAR.set(0)
    elif expansion_model == "vExp": expansionToggleVAR.set(2)
    elif expansion_model == 'None': expansionToggleVAR.set(1)

    e7c = Entry(root, width=10)
    if expansion_model == "vExp":   
        Label(root, text='vExp:', bg='gray75').grid(column=0, row=16)
        e7c.grid(column=1, row=16)
        e7c.insert(0, inps[14])


    # CME start stop time
    Label(root, text='Observed CME Boundaries', bg='gray75').grid(column=3,row=5, columnspan=2)
    Label(root, text='Start (DOY)', bg='gray75').grid(column=3,row=6)
    eS1 = Entry(root, width=10)
    eS1.grid(column=4, row=6)
    Label(root, text='Stop (DOY)', bg='gray75').grid(column=3,row=7)
    eS2 = Entry(root, width=10)
    eS2.grid(column=4, row=7)


    # FIDO parameters
    Label(root, text='FIDO position', bg='gray75').grid(column=3,row=10, columnspan=2)
    Label(root, text='FIDO Lat:', bg='gray75').grid(column=3, row=11)
    eR1 = Entry(root, width=10)
    eR1.grid(column=4, row=11)
    Label(root, text='FIDO Lon:', bg='gray75').grid(column=3, row=12)
    eR2 = Entry(root, width=10)
    eR2.grid(column=4, row=12)

    Label(root, text='Save Plot', bg='gray75').grid(column=3,row=16, columnspan=1)
    print_button = Button(root, bg='black', command = lambda: save_plot(inps, Bout, tARR, shinps, tsheath, sheathB, sheathKp))
    print_button.grid(row=17,column=3, columnspan=1)

    Label(root, text='Quit', bg='gray75').grid(column=4,row=16, columnspan=1)
    quit_button = Button(root, command = root.quit)
    quit_button.grid(row=17, column=4, columnspan=2)
    
    eR1.insert(0, inps[0])
    eR2.insert(0, inps[1])
    e1.insert(0, inps[2]) 
    e2.insert(0, inps[3])
    e3.insert(0, inps[4]) 
    e3b.insert(0, inps[5])
    e4.insert(0, inps[6]) 
    e5.insert(0, inps[7])   
    e8.insert(0, inps[8])
    e9.insert(0, inps[11])
    e6.insert(0, inps[9])
    e7.insert(0, inps[10])
    eS1.insert(0, inps[12])
    eS2.insert(0, inps[13])
    wids = [eR1, eR2, e1, e2, e3, e3b, e4, e5, e8, e6, e7, e9, eS1, eS2, e7c ]
    if doAsym:
        e7b.insert(0, DiP)
        wids = [eR1, eR2, e1, e2, e3, e3b, e4, e5, e8, e6, e7, e9, eS1, eS2, e7c, e7b]
         

    Label(root, text='Update Plot', bg='gray75').grid(column=3,row=14, columnspan=2)
    draw_button = Button(root, command = rerun, bg='gray75')
    draw_button.grid(row=15,column=3, columnspan=2)

def rerun():
    newinps, flagged = pullGUIvals()
    if not flagged: 
        checkStartStop(newinps, shinps)
        run_case(newinps, shinps)
    else:
        print('Fix CME start/stop to run')
    
def checkStartStop(inps, shinps):
    global CMEmid, plotstart, plotend
    CMEstart, CMEend = inps[12], inps[13]
    pad = 3
    plotstart = 0
    if CMEstart != 0:
        plotstart = CMEstart - pad/24. 
    if hasSheath:
        plotstart = np.min([shinps[0] - pad/24., CMEstart-shinps[1]/24.-pad/24.])
    if (CMEend == 0) and  canprint:
        print('!!!Have CME start but not stop!!!')
        print('!!!Defaulting to duration of a day!!!')
        print('!!!May cause error if insufficient in situ data!!!')
        print('!!!Should not autonormalize or use score unless fix CME stop!!!')
        CMEend = CMEstart + 1
        inps[13] = CMEend
    plotend   = CMEend + pad/24.
    
def setupObsData(inps):
    # see if we have IS and start/stop -> can calc score
    # otherwise fix so will plot and flag not to score
    data = np.genfromtxt(ISfilename, dtype=np.float, skip_header=44)
    # if have no given start/stop, set equal to begining 
    global plotstart, plotend
    if plotstart == 0:
        plotstart += data[0,0] 
        inps[12] += data[0,0]
        plotend  += data[0,0]
        inps[13]   += data[0,0]
    i_date = int(plotstart)
    i_hour = int(plotstart % 1 * 24)
    f_date = int(plotend) 
    f_hour = int(plotend % 1 % 1 * 24)
    # determine the initial and final index of the desired time period
    try:
        iidx = np.min(np.where(data[:,0] >= plotstart))
    except:
        sys.exit('CME_start outside in situ data range')
    try:
        fidx = np.max(np.where(data[:,0] <= plotend))
    except:
        sys.exit('CME_stop outside in situ data range')
    global d_Bx, d_By, d_Bz, d_Btot
    d_fracdays = data[iidx:fidx+1,0]
    d_Bx = data[iidx:fidx+1,1]
    d_By = data[iidx:fidx+1,2]
    d_Bz = data[iidx:fidx+1,3]
    d_Btot = np.sqrt(d_Bx**2 + d_By**2 + d_Bz**2)
    d_t = (d_fracdays - d_fracdays[0]) * 24
    global d_tUN
    d_tUN = d_fracdays
    global minISdate, maxISdate
    minISdate, maxISdate = np.min(d_tUN), np.max(d_tUN)
    # calculate the average magnetic field in the middle, used to normalize out B0
    CMEmid    = 0.5 * (inps[12] + inps[13])
    global avg_obs_B
    avg_obs_B = np.mean(d_Btot[np.where(np.abs(d_tUN - CMEmid) < 2./24.)])

def setupKpData(inps,Kpfilename, plotstart, plotend):
    data2 = np.genfromtxt(Kpfilename, dtype=float)
    Kpdate = data2[:,1]+data2[:,2]/24.
    if Kpdate[0] > plotstart:
        print('Plot start before Kp data start')
    if Kpdate[-1] < plotend:
        print('Plot end after Kp data end')
    obsKp = data2[:,3]/10. # bc OMNI gives Kp*10
    return Kpdate, obsKp

def runFIDO():
    # read in the text file and grab labeled inputs
    global input_values
    input_values = readinputfile()
    # set up the general properties (how will FIDO be ran)
    setupOptions(input_values)   

    # set up the actual simulation input params
    inps = getInps(input_values)
    # get sheath values if we are including
    global shinps
    shinps = []
    if hasSheath:
        shinps = getSheathInps(input_values)
     
    # set a global with the initial inps values, will default
    # to these if GUI values get changes to bad 
    global inps0
    inps0 = inps
    
    # Establish a GUI root if needed
    if Launch_GUI:
        root = Tk()
        root.configure(background='gray75')
        
    # Establish the figure and axes
    if not NoPlot:
        global axes
        fig, axes = setupFigure()
    global Kpdate, obsKp
    Kpdate, obsKp = [], []
    if hasSheath:
        checkStartStop(inps, shinps)
    else:
        checkStartStop(inps, [0.])
    if ISfilename!=False: 
        setupObsData(inps)
    if Kpfilename!=False:    
        Kpdate, obsKp = setupKpData(inps,Kpfilename, plotstart, plotend)
    
    # set up GUI with figure
    if Launch_GUI:
        setupGUI(root, fig, axes, inps, shinps)
    
    # run the initial conditions
    Bout, tARR = run_case(inps, shinps)

    if Launch_GUI != False:
    	root.mainloop()
    	root.quit()
    else:
    	save_plot(inps, Bout, tARR, 0, 0, 0, 0)
#runFIDO()
