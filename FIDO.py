import numpy as np
import math
import sys
from scipy.special import jv
from scipy import integrate
import matplotlib
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

# Empirical function to calculate length of axis
global lenFun
lenCoeffs = [0.61618, 0.47539022, 1.95157615]
lenFun = np.poly1d(lenCoeffs)

# variables that carry all the simulation params
global inps, shinps, calcSheathParams, moreShinps, vExps, CMEvr
# set up with default values that will get replaced when 
# set by user
inps = np.zeros(19)
inps[0]  = 0.       # Satellite Latitude
inps[1]  = 0.       # Satellite Longitude
inps[2]  = 0.       # CME Latitude
inps[3]  = 0.       # CME Longitude
inps[4]  = 0.       # CME Tilt
inps[5]  = 45.      # Angular Width
inps[6]  = 15.
inps[7]  = 0.7     # deltaAx
inps[8]  = 0.5     # deltaCS
inps[9] = 0.3
inps[10]  = 25.      # CME B0
inps[11] = 1.       # CME Polarity (+ = RH, - = LH)
inps[12] = 0.       # t Shift (rel. to CME start)
inps[13] = 0.       # CME start
vExp = 0.       # CME expansion velocity
inps[14] = 213.     # Satellite radius
inps[15] = 1.141e-5 # Satellite orbital rate
inps[16] = 212      # CME starting radius (Rs)
inps[17] = 1.927    # cnm
inps[18] = 1        # tau

# default to no expansion, vfront is same as vbulk
vExps = np.array([440., 0, 440, 0, 0, 0, 0])
CMEvr = 440.

shinps = np.zeros(7)
shinps[0] =  -12    # Sheath Start Time
shinps[1] =  12     # Sheath Duration
shinps[2] =  2.0    # Compression
shinps[3] =  500.   # Sheath Velocity
shinps[4] =  -4.    # SW Bx
shinps[5] =  4.     # SW By
shinps[6] =  0.     # SW Bz
moreShinps = np.zeros(5)
moreShinps[0] = 5.   # nSW
moreShinps[1] = 400. # vSW
moreShinps[2] = 49.5 # cs
moreShinps[3] = 55.4 # vA
moreShinps[4] = 600  # transit time
    

# settings for the model and their defaults
global expansion_model, ISfilename, hasSheath, calcSheath, canPrint, ObsData
expansion_model = 'None'
ISfilename = False
hasSheath  = False
calcSheath = False
canPrint   = False
ObsData    = None


# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# Geometry programs
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #

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


# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# FIDO programs
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
def isinCME(vec_in, CME_shape): 
    # Check and see if the requested point is actually in the CME and return
    # the cylindrical radial distance (from center of FR)
    # Function assumes vec_in is in CME Cartesian coord sys
    
    # Make a flux rope axis in xz plane
    thetas = np.linspace(-math.pi/2, math.pi/2, 1001)
    xFR = CME_shape[0] + CME_shape[1] * np.cos(thetas)
    zFR = CME_shape[3] * np.sin(thetas)
    # Calc the distance from all axis points
    dists2 = (vec_in[0] - xFR)**2 + vec_in[1]**2 + (vec_in[2] - zFR)**2
    # Find the closest point on the axis
    myb2 = np.min(dists2)
    minidx = np.where(dists2 == myb2)[0]
    temp = thetas[np.where(dists2 == myb2)]
    mythetaT = temp[0]
    # Do a second iteration to refine B by looking near the closest point
    # First check which side of minidx the actual minimum is on
    if minidx < len(dists2) - 1: # check to make sure not already at edge
        if dists2[minidx-1] < dists2[minidx+1]: startidx = minidx - 1
        else:  startidx = minidx + 1
    	# Repeat the min distance procedure on the correct side
        if dists2[minidx-1] != dists2[minidx+1]:
            thetas2 = np.linspace(thetas[startidx], thetas[minidx], 101)
            xFR2 = CME_shape[0] + CME_shape[1] * np.cos(thetas2)
            zFR2 = CME_shape[3] * np.sin(thetas2)
            dists2 = (vec_in[0] - xFR2)**2 + vec_in[1]**2 + (vec_in[2] - zFR2)**2
            myb2 = np.min(dists2)
            minidx = np.where(dists2 == myb2)[0]
            mythetaT = thetas2[np.where(dists2 == myb2)][0] 
    # Don't run again if at edge 
    else: 
        xFR2 = xFR
    # Determine if the closest distance is less than the cross section radius 
    myb = np.sqrt(myb2)
    CME_crossrad = CME_shape[2]
    # Return if outside -> misses the satellite
    if (myb > CME_crossrad):
        myb = -9999.
        return myb, -9999, -9999., -9999, -9999    
    # If hits, find the theta/phi at point of impact    
    else:
	# thetaP should swing quickly at closest approach to FR
        mythetaP = np.arcsin(vec_in[1] / myb)
        origTP = mythetaP        
	    # removed a check to make sure did second loop
	# convert thetaP to complement (used to have a try/except here)
        if vec_in[0] < (xFR2[minidx] ):	
            if vec_in[1] > 0: mythetaP = math.pi - mythetaP
            else: mythetaP = -math.pi - mythetaP
        # shut down if more than 90 from nose (behind what normal consider to be the torus)
        # with out this, will add rounded hemispheres at back of torus legs essentially
        if np.abs(mythetaT) > 3.14159/2.:
            return -9999, -9999, -9999., -9999, -9999    
    return myb, mythetaT, mythetaP, 0, CME_crossrad

def new_isinCME(vec_in, CMElens, deltaAx, deltaCS): 
    # Check and see if the requested point is actually in the CME and return
    # the cylindrical radial distance (from center of FR)
    # Function assumes vec_in is in CME Cartesian coord sys
    # CMElens = [CMEnose, rEdge, d, br, bp, a, c]
    
    # Make a flux rope axis in xz plane
    thetas = np.linspace(-math.pi/2, math.pi/2, 1001)
    xFR = CMElens[2] + deltaAx * CMElens[6] * np.cos(thetas)
    zFR = 0.5 * CMElens[6] * np.sign(thetas)*(np.sin(np.abs(thetas)) + np.sqrt(1 - np.cos(np.abs(thetas))))
    # Calc the distance from all axis points
    dists2 = (vec_in[0] - xFR)**2 + vec_in[1]**2 + (vec_in[2] - zFR)**2
    # Find the closest point on the axis
    myb2 = np.min(dists2)
    
    minidx = np.where(dists2 == myb2)[0]
    temp = thetas[np.where(dists2 == myb2)]
    mythetaT = temp[0]
    myb = np.sqrt(myb2)
    CME_crossrad = CMElens[4] 
    # Return if outside -> misses the satellite
    if (myb > CME_crossrad):
        myb = -9999.
        return myb, -9999, -9999., -9999, -9999    
    # If hits, find the theta/phi at point of impact    
    else:
    # Need to check if less than actual r for that part of cross section
        # Get the normal vector
        if mythetaT == 0: mythetaT = 0.00001
        sn = np.sign(mythetaT)
        mythetaT = np.abs(mythetaT)
        Nvec = [0.5 * sn*(np.cos(mythetaT) + 0.5*np.sin(mythetaT)/np.sqrt(1-np.cos(mythetaT))), 0., deltaAx * np.sin(mythetaT)] 
        Nmag = np.sqrt(Nvec[0]**2+Nvec[2]**2)
        norm = np.array(Nvec) / Nmag
        
        # Find the sat position within the CS
        thisAx = [CMElens[2] + deltaAx * CMElens[6] * np.cos(mythetaT), 0.,  0.5 * sn * CMElens[6] * (np.sin(mythetaT) + np.sqrt(1 - np.cos(mythetaT)))]
        vp = np.array(vec_in) - np.array(thisAx)
        vpmag = np.sqrt(np.sum(vp**2))
        vpn = vp / vpmag
        dotIt = np.dot(vp, norm)
        CSpol = np.abs(np.arccos(dotIt / vpmag))
        #CSxy = np.array([vpmag*np.cos(CSpol), vpmag*np.sin(CSpol)])
        parat = np.abs(np.arctan(np.tan(CSpol)*deltaCS))
        ogParat = np.arctan(np.tan(CSpol)*deltaCS)
        # need to make sure parat is in right quadrant
        zone = 1
        if (vp[1] < 0):
            if vp[0] <0:
                parat = -(math.pi - parat)
                zone = 4
            else:
                parat = -parat
                zone = 3
        elif (vp[0]<0): # y pos, x neg
            parat = math.pi - parat
            zone=2
        # Get the max R for that parametric t
        maxr = np.sqrt(deltaCS**2 * np.cos(parat)**2 + np.sin(parat)**2) * CMElens[4]  
        if myb < maxr:
            return myb, sn*mythetaT, parat, 0, maxr
        else:
            return myb, -9999, -9999, -9999, -9999


# -------------------------------------------------------------------------------------- #
def getBvector(CME_shape, minb, thetaT, thetaP):
    # Takes in the CME shape and point within CME and gets the poloidal and
    # toroidal direction at that point
    tdir = np.array([-(CME_shape[1] + minb * np.cos(thetaP)) * np.sin(thetaT), 0., (CME_shape[3] + minb * np.cos(thetaP)) * np.cos(thetaT)])
    pdir = np.array([-minb * np.sin(thetaP) * np.cos(thetaT), minb * np.cos(thetaP), -minb * np.sin(thetaP) * np.sin(thetaT)])
    # Convert to unit vector
    tmag = np.sqrt(np.sum(tdir**2))
    pmag = np.sqrt(np.sum(pdir**2))
    tdir = tdir / tmag
    pdir = pdir / pmag
    return tdir, pdir

def new_getBvector(CMElens, minb, mythetaT, thetaP, deltaAx, deltaCS):
    # Takes in the CME shape and point within CME and gets the poloidal and
    # toroidal direction at that point
    # Normal direction for that theta along axis
    sn = np.sign(mythetaT)
    mythetaT = np.abs(mythetaT)
    # took out sn from Nvec[0], pretty sure this fixes weird jumps
    Nvec = [0.5 * (np.cos(mythetaT) + 0.5*np.sin(mythetaT)/np.sqrt(1-np.cos(mythetaT))), 0., deltaAx * np.sin(mythetaT)] 
    Nmag = np.sqrt(Nvec[0]**2+Nvec[2]**2)
    norm = np.array(Nvec) / Nmag
    axAng = np.arccos(np.abs(norm[0]))
    # tangent direction
    tan  = np.zeros(3)
    tan[0], tan[2] = -norm[2], norm[0] 
    # CS angular direction
    pol = -np.array([-deltaCS * np.sin(thetaP), np.cos(thetaP), 0.])
    return tan, pol
      
    

# -------------------------------------------------------------------------------------- #
def getFRprofile(satfs=None,  FRmass=None, FRtemp=None, FRgamma=5./3):
    # Main function for getting the flux rope profile by propagating a CME outward, 
    # determining when/where it is within the CME, and getting the magnetic field vector
    # Unpack the parameters from inps
    FFlat, FFlon, CMElat, CMElon, CMEtilt, CMEAW, CMEAWp, deltaAx, deltaCS, deltaCSAx, CMEB, CMEH, tshift, CMEstart, FFr, rotspeed, CMEr, cnm, tau = inps[0], inps[1], inps[2], inps[3], inps[4], inps[5], inps[6], inps[7], inps[8], inps[9], inps[10], inps[11], inps[12], inps[13], inps[14], inps[15], inps[16], inps[17], inps[18]
                
    # new shape things
    #CMElens = [CMEnose, rEdge, d, br, bp, a, c]
    CMElens = np.zeros(7)
    CMElens[0] = CMEr#0.999*FFr # start just in front of sat dist, can't hit any closer 
    CMElens[4] = np.tan(CMEAWp*dtor) / (1 + deltaCS * np.tan(CMEAWp*dtor)) * CMElens[0]
    CMElens[3] = deltaCS * CMElens[4]
    CMElens[6] = (np.tan(CMEAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + deltaAx * np.tan(CMEAW*dtor))    
    CMElens[5] = deltaAx * CMElens[6]
    CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
    CMElens[1] = CMElens[2] * np.tan(CMEAW*dtor)
    
    # things for scaling the magnetic field as CME changes shape/size
    B0 = CMEB / deltaCS / tau
    B0scaler = B0 * deltaCS**2 * CMElens[4]**2 
    initlen = lenFun(CMElens[5]/CMElens[6]) * CMElens[6]
    cnmscaler = cnm / initlen * CMElens[4] * (deltaCS**2+1)    
    
    # calc n/T
    if FRmass != None:
        vol = math.pi*CMElens[3]*CMElens[4] *  lenFun(CMElens[5]/CMElens[6])*CMElens[6] * (7e10)**3
        n = FRmass/ vol / 1.67e-24
    if FRtemp != None:
        gm1 = FRgamma - 1    
        tempScaler = np.power(CMElens[3] * CMElens[4] * initlen , gm1) * FRtemp
        
    # Set up arrays to hold values over the spacecraft path
    obsBx = []
    obsBy = []
    obsBz = []
    tARR = []
    rCME = []
    nCME = []
    tempCME = []
    radfrac = []
    vIS  = []
    
    # Initialize time
    t = 0
    
    # Set various switches/flags
    # Track ThetaP to make sure doesn't jump -> need previous value
    thetaPprev = -42
    enteredCME = False
    # If exp = None then is self-similar til impact (so AW = given value
    # at time of impact) then switches to none
    flagExp = False
    # Variable to keep track of closest approach
    ImpParam = 9999.
    ImpThetaT = 9999.
    
    # set up whether using path functions or simple orbit
    if satfs == None:
        doPath = False
    else:
        doPath = True
        fLat = satfs[0]
        fLon = satfs[1]
        fR   = satfs[2]
    
    # Start simulation, run until reach some tmax so doesn't loop forever
    while t < tmax:  
	    # Converting FIDO position to CME Cartesian 
        # Done in loop so includes Elon change
	    # Get Sun xyz position
        FF_sunxyz = SPH2CART([FFr, FFlat, FFlon])
        # Rotate to CME coord system
        temp = rotz(FF_sunxyz, -CMElon)
        temp2 = roty(temp, CMElat)
        FF_CMExyz = rotx(temp2, (90.-CMEtilt))
        
        minb, thetaT, thetaP, flagit, CME_crossrad = new_isinCME(FF_CMExyz, CMElens, deltaAx, deltaCS)    
               
        # Check if currently in the CME and get position within
        # Track the minimum distance from center -> Impact Parameter
        if np.abs(minb/CME_crossrad) < ImpParam: ImpParam = np.abs(minb/CME_crossrad)
        # If hasn't been flagged determine the magnetic field vector
        if flagit != -9999:
            enteredCME = True
            # If not set to expand flag it to stop after first contact
            if expansion_model != 'Self-Similar': flagExp = True
            # Get the toroidal and poloidal magnetic field based on distance
            Btor = B0 * deltaCS * (tau - (minb/CME_crossrad)**2)
            ecH  = np.sqrt(deltaCS**2 * np.sin(thetaP)**2 + np.cos(thetaP)**2)
            Bpol = - 2 * deltaCS * B0 * CMEH * ecH / (deltaCS**2 + 1) / cnm * (minb/CME_crossrad) 
            #print (np.abs(minb/CME_crossrad),thetaT,thetaT*radeg)
            # Convert this into CME Cartesian coordinates
            tdir, pdir = new_getBvector(CMElens, minb, thetaT, thetaP, deltaAx, deltaCS) 
            
            Bt_vec = Btor * tdir
            Bp_vec = Bpol * pdir
            Btot = Bt_vec + Bp_vec
            # Convert to spacecraft coord system
            temp = rotx(Btot, -(90.-CMEtilt))
            temp2 = roty(temp, CMElat - FFlat) 
            BSC = rotz(temp2, CMElon - FFlon)
            
            # get accurate velocity vector
            vCMEframe, vExpCME = getvCMEframe(minb/CME_crossrad, thetaT, thetaP, CMElens[5] / CMElens[6], CMElens[3] / CMElens[4], vExps)

            # rot B to s/c frame -> take - of x&y but this correct for x away from sun
            temp = rotx(vCMEframe, -(90.-CMEtilt))
            temp2 = roty(temp, CMElat - FFlat) 
            vInSitu = rotz(temp2, CMElon - FFlon)
            
            # Update n/T
            if FRmass != None:
                vol = math.pi*CMElens[3]*CMElens[4] *  lenFun(CMElens[5]/CMElens[6])*CMElens[6] * (7e10)**3
                n = FRmass/ vol / 1.67e-24
            else:
                n = 0
                
            if FRtemp !=None:
                lenNow = lenFun(CMElens[5]/CMElens[6])*CMElens[6] 
                tem = tempScaler / np.power(CMElens[3] * CMElens[4] * lenNow, gm1)
            else:
                tem = 0
                
                
            # Append to output arrays
            obsBx.append(-BSC[0])
            obsBy.append(-BSC[1])
            obsBz.append(BSC[2])
            tARR.append(t/3600.)
            rCME.append(CMElens[0])
            nCME.append(n)
            tempCME.append(tem)
            radfrac.append(minb/CME_crossrad)
            vIS.append(vInSitu[0])   
            
        else:
            # stop checking if exit CME
            if enteredCME: t = tmax+1
        # Move to next step in simulation
        t += dt
	    # CME nose moves to new position, have vr in vexp vector
        CMElens += vExps * dt / 7e5
        # Update B0/cnm
        lenNow = lenFun(CMElens[5]/CMElens[6])*CMElens[6]
        B0 = B0scaler / deltaCS**2 / CMElens[4]**2 
        cnm = cnmscaler * lenNow  / CMElens[4] / (deltaCS**2+1)
                             
	    # Include the orbit of the satellite/Earth
        if not doPath:
            FFlon += dt * rotspeed
        else:
            FFlat = fLat(t)
            FFlon = fLon(t)
            FFr   = fR(t)
            
        
    # Clean up the result and package to return
    obsBx, obsBy, obsBz, tARR = np.array(obsBx), np.array(obsBy), np.array(obsBz), np.array(tARR)
    obsB = np.sqrt(obsBx**2 + obsBy**2 + obsBz**2)
    zerotime = (CMEstart + tshift/24.) # in days
    # Try to set tARR to start at time 0, if fails then know a miss
    try:
        tARR = (tARR-tARR[0])/24. # also in days
        tARR += zerotime
        isHit = True
    except:
        isHit = False
    Bout = np.array([obsBx, obsBy, obsBz, obsB])   
    return Bout, tARR, isHit, ImpParam, np.array(radfrac), np.array(vIS), np.array(nCME), np.array(tempCME)
    
# -------------------------------------------------------------------------------------- #
def run_case(inpsIn, shinpsIn, vExpIn, satfs=None,  FRmass=None, FRtemp=None, FRgamma=None):
    # Take the values passed in and set to globals for convenience
    global inps, shinps, vExps
    inps = inpsIn
    shinps = shinpsIn
    vExps = vExpIn
    
    # use to set sheath end
    CMEstart = inps[13] + inps[12]/24.
    
    # run the simulation    
    Bout, tARR, isHit, ImpParam, radfrac, vIS, nCME, tempCME = getFRprofile(satfs=satfs, FRmass=FRmass, FRtemp=FRtemp, FRgamma=FRgamma)
    # execute extra functions as needed
    Bsheath = []
    tsheath = []
    if isHit:
        # Add sheath if desired
        if hasSheath:
            # Given Br and Bphi in shinps-> change to taking in [B, Bx, By, Bz]?
            upB = np.array([np.sqrt(shinps[4]**2+shinps[5]**2+shinps[6]**2),shinps[4],shinps[5], shinps[6]])   
            # Calculate compressed magnetic field            
            jumpvec = upB * shinps[2] 
            # Create a sheath by connecting jumpvec to front of flux rope
            tsheath, sheathB = make_sheath(jumpvec, Bout, CMEstart, shinps)  
            Bsheath = np.array([sheathB[1], sheathB[2], sheathB[3], sheathB[0]])
    # No impact case
    else:
        if canPrint: print ('No impact expected')
    return Bout, tARR, Bsheath, tsheath, radfrac, isHit, vIS, nCME, tempCME 

# -------------------------------------------------------------------------------------- #
def radfrac2vprofile(radfrac, vAvg, vExp):
    # Take the position within CME profile and add expansion to vAvg.
    # this is correct if path through is diameter, not exact for shorter
    # cut through, how to use angle btwn trajectory and impact?
    newfrac = radfrac - np.min(radfrac)
    newfrac = newfrac/np.max(newfrac)
    centID = np.where(newfrac == 0)[0]
    centID = centID[0]
    vProf = vAvg + vExp*newfrac
    vProf[centID:] = vAvg - vExp*newfrac[centID:]
    return vProf
    
# -------------------------------------------------------------------------------------- #
def getvCMEframe(rbar, thetaT, thetaP, delAx, delCS, vExps):
    # CMElens/vExps = [CMEnose, rEdge, d, br, bp, a, c]
    aTT = np.abs(thetaT)
    pmTT = np.sign(thetaT)    
    # can take out the Lp part in calc xAx and zAx
    xAx = delAx * np.cos(aTT)
    zAx = 0.5 * (np.sin(aTT) + np.sqrt(1 - np.cos(aTT))) * pmTT
    thisL = np.sqrt(xAx**2 + zAx**2)
    vAx = thisL * vExps[5] / delAx
    nAx = np.array([0.5 * (np.cos(aTT) + 0.5 * np.sin(aTT)/np.sqrt(1-np.cos(aTT))), 0, delAx * np.sin(thetaT)])
    normN = np.sqrt(np.sum(nAx**2))
    nAx = nAx / normN
    vAxVec = vAx * nAx
    
    xCS = delCS * rbar * np.cos(thetaP)
    yCS = rbar * np.sin(thetaP)
    thisr = np.sqrt(xCS**2 + yCS**2) 
    vCS = thisr * vExps[3] / delCS
    nCS = np.array([np.cos(thetaP)/delCS, np.sin(thetaP), 0.])
    normN2 = np.sqrt(np.sum(nCS**2))
    nCS = nCS / normN2
    nCSatAx = np.array([nCS[0] * np.cos(thetaT), nCS[1], nCS[0] * np.sin(thetaT)])
    vCSVec = vCS * nCSatAx
    vCMEframe = np.array([vExps[2], 0., 0.]) + vAxVec + vCSVec
    return vCMEframe, vCSVec
    

# -------------------------------------------------------------------------------------- #
def hourify(tARR, vecin):
    # Assume input is in days, will spit out results in hourly averages
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
    
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
# Sheath Things 
# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #
def make_sheath(jumpvec, Bout, CMEstart, shinps):
    # Function that takes the B vectors at the front and back of the sheath
    # and pretty much makes a straight line between.
    # Place the sheath at the appropriate time before the CME
    tSheath = np.linspace(CMEstart-shinps[1]/24.,CMEstart,100)
    BUsheath = [[],[],[],[]]
    # Flux rope vector
    iFRvec = [Bout[3][0], Bout[0][0], Bout[1][0], Bout[2][0]]
    # Sheath vector
    iShvec = jumpvec    
    # Convert to unit vectors
    iShUvec = iShvec / iShvec[0]
    iFRUvec = iFRvec / iFRvec[0]
    # Calculate the slope between front and back for each component
    # of unit vectors then make a profile with that slope
    for i in range(4):
        slope = (iFRUvec[i]-iShUvec[i])/(tSheath[-1]-tSheath[0])
        BUsheath[i] = iShUvec[i]+slope*(tSheath-tSheath[0]) 
    # Determine the magnitude over time based on components    
    BUsheath[0] = np.sqrt(BUsheath[1]**2 + BUsheath[2]**2 + BUsheath[3]**2)
    BUsheath = np.array(BUsheath)
    # The linear profiles won't guarantee a unit vector -> figure out how much
    # we need to scale each time step to make unit magnitude and scale it
    scl = 1./ BUsheath[0]
    Bsheath = BUsheath*scl
    # Get the linear total B vector and multiply by scaled unit vector
    Bslope = (iFRvec[0]-iShvec[0])/(tSheath[-1]-tSheath[0])
    Bmag = iShvec[0] + Bslope*(tSheath-tSheath[0]) 
    Bsheath *= Bmag
    return tSheath, Bsheath

# -------------------------------------------------------------------------------------- #
def calcSheathInps(CMEstart, vels, nSW, BSW, satr, B=None, cs=49.5, vA=55.4):
    # Given basic SW and CME properties calculate the sheath parameters using the
    # methods from the 2020 FIDO-SIT paper
    # vels should be [vbulk, vexp, vtransit (avg), vSW]
    # Assume B in is Br, was calc at 213 so scale if need be
    # Use a simple Parker model for mag field
    # BSW can be positive for radial out or neg for radial in 
    if B == None:
        # scale to sat distance if not 1 AU
        BphiBr = (2.7e-6) * satr * 7e5 / vels[3]
        Br = BSW/np.sqrt(1+BphiBr**2) * (213./satr)**2
        Bphi = -Br * BphiBr 
        BSW = np.sqrt(Br**2+Bphi**2)
        B = [-Br, -Bphi, 0.]
    # Calculate Alfven speed
    vA = BSW *1e-5 / np.sqrt(4*3.14159 * nSW * 1.67e-24) / 1e5
    # Velocity at front of flux rope (expansion+bulk)
    vfront = vels[0]+vels[1]
    # Calc sheath v using MLR from paper
    sheathv = 0.297*(vfront)+0.112*vels[2]+0.779*vels[3]-37.6
    # Use perpendicular shock model for compression and vshock
    compression, vs =getPerpSheath(vfront, vels[3], nSW, BSW, cs, vA)
    # Sheath duration from MLR
    sheathTime = (-16.1*(vfront) + 11.9*(vs- vfront) + 6.8*vels[2] +7776)/sheathv
    return [CMEstart-sheathTime/24., sheathTime, compression, sheathv, B[0], B[1], B[2], vs]
    
# -------------------------------------------------------------------------------------- #
def getPerpSheath(vSheath, vSW, nSW, BSW, csin=49.5, vAin=55.4):
    # Use perp RH equations to derive the sheath properties given upstream conditions
    # and sheath velocity
    # Iterate through until find values that satisfy both equations since can't solve
    # analytically
    # avg SW T of 1.5e5 at 1AU -> cs =49.5 if not given
    # avg SW B and n -> vA = 1e-5 B (nT) / sqrt(4pi n (cm-3) * mp) / 1e5 km/s
    vA = vAin
    cs = csin
    ms = np.sqrt(vA**2 + cs**2)
    
    # Assume a value for delta (offset between downstream and shock speed)
    delta = 0    
    vS = vSheath + delta
    r = determineR(vS, vSheath, vSW, cs, vA)
    vu = vSW - vS
    # Calculate the downstream velocity (in shock frame) two ways
    vd1 = vSheath - vS
    vd2 = vu/r
    diff = vd1 - vd2
    olddiff = vd1 - vd2
    counter = 0
    # Keep changing delta as the diff decreases
    while diff*olddiff > 0:
        olddiff = diff
        delta+=1.
        counter+=1
        vS = vSheath + delta
        r = determineR(vS, vSheath, vSW, cs, vA)
        vu = vSW - vS
        vd1 = vSheath - vS
        vd2 = vu/r
        diff = vd1 - vd2
        if counter > 5000: break
    # Take a step back and repeat the process with finer resolution
    delta -= 1.
    diff = olddiff
    if counter <4999:
        while diff*olddiff > 0:
            olddiff = diff
            delta+=0.1
            vS = vSheath + delta
            r = determineR(vS, vSheath, vSW, cs, vA)
            vu = vSW - vS
            vd1 = vSheath - vS
            vd2 = vu/r
            diff = vd1 - vd2
            #print vS, r, diff, vu/r
        nd = nSW*r
        Bd = BSW*r
        # return the compression and shock speed
        return r, vS

def determineR(vS, vSheath, vSW, cs, vA): 
    # Use the shock adiabatic to calculate the compression given
    # the shock speed (and other params)
    gam = 5/3.
    vu = vSW - vS
    vd = vSheath - vS
    beta = (2./gam) * (cs/vA)**2
    Ma2 = (vu/vA)**2
    # from utex perp shock page with math (their M is sonic only)
    coeffs = [2-gam, gam*(1+beta)+(gam-1)*Ma2, -(gam+1)*Ma2]
    roots = np.roots(coeffs)
    goodroots = roots[np.where((roots>=1))]
    roots = goodroots[np.where((goodroots<=4))]
    if len(roots) == 1:
        return roots[0]   
    else:
        return 1.

# -------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------- #

def startFromText():
    input_values = readInputFile()
    assignInps(input_values)
    
def readInputFile():
    # Takes in a file from the command line and puts parameters in a dictionary if
    # they are given with keyword from the list
    # Don't simulation if no file is passed but warn (will use defaults)
    if len(sys.argv) < 2: 
        #sys.exit("Need an input file")
        print('No input file given!')
        sys.exit()
    # Read in the file   
    global input_file 
    input_file = sys.argv[1]
    inputs = np.genfromtxt(input_file, dtype=str, encoding='utf8')
    # Include some duplicates between OSPREI names and more natural FIDO only names
    # (ie CME_lat and ilat both work to set CME lat) so can reuse OSPREI .txt files
    possible_vars = ['CME_lat', 'CME_lon', 'CME_tilt', 'CME_AW', 'CME_AWp', 'AW', 'AWp', 'ilat', 'ilon', 'tilt', 'AWmax', 'shapeA', 'shapeB', 'FR_B0', 'FR_pol', 'CME_start', 'CME_stop', 'Expansion_Model', 'CME_vExp', 'CME_v1AU', 'CME_vr', 'vTrans', 'Sat_lat', 'Sat_lon', 'Sat_rad', 'includeSIT', 'hasSheath', 'sheathTime', 'Compression', 'vSheath', 'SWBr', 'SWBphi', 'calcSheath', 'BSW', 'nSW', 'vSW', 'vTransit', 'cs', 'vA', 'SWBx', 'SWBy', 'SWBz', 'ObsDataFile', 'ImpCMEr', 'Sat_rot', 'deltaAx', 'deltaCS', 'cnm', 'Cnm', 'tau']
    # Check if name is in list and add to dictionary if so
    input_values = {}
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
    return input_values
        
def assignInps(input_values):
    global inps, shinps, hasSheath, calcSheath, expansion_model, ObsData, vExps, CMEvr
    input_names = input_values.keys()
    if 'ObsDataFile' in input_names:
        ObsData = input_values['ObsDataFile']
    if 'Sat_lat' in input_names:
        inps[0] = float(input_values['Sat_lat'])
    if 'Sat_lon' in input_names:
        inps[1] = float(input_values['Sat_lon'])
    if 'ilat' in input_names:
        inps[2] = float(input_values['ilat'])
    elif 'CME_lat' in input_names:
        inps[2] = float(input_values['CME_lat'])
    if 'ilon' in input_names:
        inps[3] = float(input_values['ilon'])
    elif 'CME_lon' in input_names:
        inps[3] = float(input_values['CME_lon'])
    if 'tilt' in input_names:
        inps[4] = float(input_values['tilt'])
    elif 'CME_tilt' in input_names:
        inps[4] = float(input_values['CME_tilt'])
    if 'CME_AW' in input_names:
        inps[5] = float(input_values['CME_AW'])
    elif 'AWmax' in input_names:
        inps[5] = float(input_values['AWmax'])
    elif 'AW' in input_names:
        inps[5] = float(input_values['AW'])
    if 'CME_AWp' in input_names:
        inps[6] = float(input_values['CME_AWp'])
    elif 'AWp' in input_names:
        inps[6] = float(input_values['AWp'])        
    if 'deltaAx' in input_names:
        inps[7] = float(input_values['deltaAx'])
    if 'deltaCS' in input_names:
        inps[8] = float(input_values['deltaCS'])
    if 'deltaCSAx' in input_names:
        inps[9] = float(input_values['deltaCSAx'])
    if 'FR_B0' in input_names:
        inps[10] = float(input_values['FR_B0'])
    if 'FR_pol' in input_names:
        inps[11] = float(input_values['FR_pol'])
    if 'tshift' in input_names:
        inps[12] = float(input_values['tshift'])
    if 'CME_start' in input_names:
        inps[13] = float(input_values['CME_start'])
    if 'Sat_rad' in input_names:
        inps[14] = float(input_values['Sat_rad'])
    if 'Sat_rot' in input_names:
        inps[15] = float(input_values['Sat_rot'])  
    if 'ImpCMEr' in input_names:
        inps[16] = float(input_values['ImpCMEr'])
    if 'cnm' in input_names:
        inps[17] = float(input_values['cnm'])
    elif 'Cnm' in input_names:
        inps[17] = float(input_values['Cnm'])
    if 'tau' in input_names:
        inps[18] = float(input_values['tau'])
                
    if 'CME_vr' in input_names:
        CMEvr = float(input_values['CME_vr'])
    elif 'CME_v1AU' in input_names:
        CMEvr = float(input_values['CME_v1AU'])        
    if 'CME_vExp' in input_names:
        vExp = float(input_values['CME_vExp']) / inps[8] # vExp_rr is vExp * deltaCS so convert to just vExp
        # vExps = [CMEnose, rEdge, d, br, bp, a, c]
    vExps = np.array([CMEvr, 0, 0., inps[8]*vExp, vExp, vExp/inps[9]*inps[7], vExp/inps[9]])    
    vExps[2] = vExps[0] - vExps[3] - vExps[5]
    vExps[1] = vExps[2]*np.tan(inps[5]*dtor)
            
    # Can either take sheath inputs from file or calculate from 
    # given parameters
    if 'hasSheath' in input_names:
        if input_values['hasSheath'] == 'True':
            hasSheath = True
    if 'includeSIT' in input_names:
        if input_values['includeSIT'] == 'True':
            hasSheath = True
    if 'sheathTime' in input_names:
        shinps[1] = float(input_values['sheathTime'])  
    if 'Compression' in input_names:
        shinps[2] = float(input_values['Compression'])  
    if 'vSheath' in input_names:
        shinps[3] = float(input_values['vSheath']) 
    countBsIn = 0    
    if 'SWBx' in input_names:
        shinps[4] = float(input_values['SWBx'])  
        countBsIn += 1
    if 'SWBy' in input_names:
        shinps[5] = float(input_values['SWBy']) 
        countBsIn += 1 
    if 'SWBz' in input_names:
        shinps[6] = float(input_values['SWBz'])
        countBsIn += 1
    if 'calcSheath' in input_names:
        if input_values['calcSheath'] == 'True':
            calcSheath = True
    # set start of sheath based on CME_start and sheathTime        
    shinps[0] = inps[13]-shinps[1]/24.
    # Pull in parameters needed to calculate the sheath if needed
    global moreShinps
    if calcSheath:
        BSW = np.sqrt(shinps[4]**2 + shinps[5]**2) # use other values if given
        nSW = 5.   # Solar wind number density
        vSW = 400. # Solar wind velocity
        vTransit = 600. # Average transit velocity
        cs = 49.5  # Sound speed
        vA = 55.4  # Alfven speed
        if 'BSW' in input_names:
            BSW = float(input_values['BSW'])
        if 'nSW' in input_names:
            nSW = float(input_values['nSW'])
        if 'vSW' in input_names:
            vSW = float(input_values['vSW'])
        if 'vTransit' in input_names:
            vTransit = float(input_values['vTransit'])
        if 'cs' in input_names:
            cs = float(input_values['cs'])
        if 'vA' in input_names:
            vA = float(input_values['vA'])
        moreShinps = [nSW, vSW, cs, vA, vTransit]
            
        vels =  [CMEvr, vExp, vTransit, vSW]
        if countBsIn == 3:
            Bvec = [shinps[4], shinps[5], shinps[6]]
            shinps = calcSheathInps(inps[13], vels, nSW, BSW, inps[14], B=Bvec, cs=cs, vA=vA)    
        else:
            shinps = calcSheathInps(inps[13], vels, nSW, BSW, inps[14], cs=cs, vA=vA)    
            
def saveResults(Bout, tARR, Bsheath, tsheath, radfrac, t_res=1):
    # Save a file with the data
    try:
        FIDOfile = open('FIDOresults'+input_file[:-4]+'.dat', 'w')
        print ('Saving results in ', 'FIDOresults'+input_file[:-4]+'.dat')
    except:
        FIDOfile = open('FIDOresults.dat', 'w')
        print ('Saving results in FIDOresults.dat')
        
    # Down sample B resolution
    # resolution = 60 mins/ t_res -> # of points per hour
    tARRDS = hourify(t_res*tARR, tARR)
    BvecDS = [hourify(t_res*tARR,Bout[0][:]), hourify(t_res*tARR,Bout[1][:]), hourify(t_res*tARR,Bout[2][:]), hourify(t_res*tARR,Bout[3][:])]
    vProf = radfrac2vprofile(radfrac, vs[0], vExp)
    vProfDS = hourify(t_res*tARR, vProf)
    
    # Write sheath stuff first if needed
    if hasSheath:
        tsheathDS = hourify(t_res*tsheath, tsheath)
        BsheathDS = [hourify(t_res*tsheath,Bsheath[0][:]), hourify(t_res*tsheath,Bsheath[1][:]), hourify(t_res*tsheath,Bsheath[2][:]), hourify(t_res*tsheath,Bsheath[3][:])]
        for j in range(len(BsheathDS[0])):
            outprint = ''
            outstuff = [tsheathDS[j], BsheathDS[3][j], BsheathDS[0][j], BsheathDS[1][j], BsheathDS[2][j], shinps[3]]
            for iii in outstuff:
                outprint = outprint +'{:6.3f}'.format(iii) + ' '
            FIDOfile.write(outprint+'\n')
    # Print the flux rope field        
    for j in range(len(BvecDS[0])):
        outprint = ''
        outstuff = [tARRDS[j], BvecDS[3][j], BvecDS[0][j], BvecDS[1][j], BvecDS[2][j], vProfDS[j]]
        for iii in outstuff:
            outprint = outprint +'{:6.3f}'.format(iii) + ' '
        FIDOfile.write(outprint+'\n')
        
def saveRestartFile():
    # Save a new file FIDO.inp with the current values of everything
    print('Saving current parameters in FIDO.inp')
    FIDOfile = open('FIDO.inp', 'w')
    if ObsData != None:
        FIDOfile.write('ObsDataFile: '+ObsData+'\n')
    FIDOfile.write('Sat_lat: '+str(inps[0])+'\n')
    FIDOfile.write('Sat_lon: '+str(inps[1])+'\n')
    FIDOfile.write('CME_lat: '+str(inps[2])+'\n')
    FIDOfile.write('CME_lon: '+str(inps[3])+'\n')
    FIDOfile.write('CME_tilt: '+str(inps[4])+'\n')
    FIDOfile.write('CME_AW: '+str(inps[5])+'\n')
    FIDOfile.write('shapeA: '+str(inps[6])+'\n')
    FIDOfile.write('shapeB: '+str(inps[8])+'\n')
    FIDOfile.write('CME_vr: '+str(vs[0])+'\n')
    FIDOfile.write('FR_B0: '+str(inps[10])+'\n')
    FIDOfile.write('FR_pol: '+str(inps[11])+'\n')
    FIDOfile.write('tshift: '+str(inps[12])+'\n')
    FIDOfile.write('CME_start: '+str(inps[13])+'\n')
    FIDOfile.write('CME_vExp: '+str(vExp)+'\n')
    FIDOfile.write('Sat_rad: '+str(inps[14])+'\n')
    FIDOfile.write('Sat_rot: '+str(inps[15])+'\n')
    FIDOfile.write('Expansion_Model: '+expansion_model+'\n')
    FIDOfile.write('hasSheath: ' + str(hasSheath)+'\n')
    if hasSheath:
        FIDOfile.write('calcSheath: '+str(calcSheath)+'\n')
        FIDOfile.write('sheathTime: '+'{:6.2f}'.format(shinps[1])+'\n')
        FIDOfile.write('Compression: ''{:6.2f}'.format(shinps[2])+'\n')
        FIDOfile.write('vSheath: '+'{:6.2f}'.format(shinps[3])+'\n')
        FIDOfile.write('SWBx: '+str(shinps[4])+'\n')
        FIDOfile.write('SWBy: '+str(shinps[5])+'\n')
        FIDOfile.write('SWBz: '+str(shinps[6])+'\n')
        if calcSheath:
            FIDOfile.write('nSW: '+str(moreShinps[0])+'\n')
            FIDOfile.write('vSW: '+str(moreShinps[1])+'\n')
            FIDOfile.write('cs: '+str(moreShinps[2])+'\n')
            FIDOfile.write('vA: '+str(moreShinps[3])+'\n')
            FIDOfile.write('vTransit: '+str(moreShinps[4])+'\n')            
            
    FIDOfile.close()

if __name__ == '__main__':
    # order is sat_lat [0], Sat_lon [1], CMElat [2], CMElon [3], CMEtilt [4], CMEAW [5]
    # CMEAWp[6], CMEdeltaAx [7], CMEdeltaCS [8], CMEdeltaCSAx [9], CMEvr [10], CMEB0 [11], CMEH [12],  
    # tshift [13], tstart [14], Sat_rad [15], Sat_rot [16], CMEr[17], cnm [18], tau [19]
    
    #startFromText()    
            
    inps = np.array([ 2.41913440e+00,  3.34644031e+02, -1.98498080e+00,  3.36247187e+02,      1.13041108e+01 , 2.56405640e+01 , 1.11406022e+01 , 3.44152732e-01 ,     4.46205290e-01,  2.66446241e-01, -2.53424676e+01, -1.00000000e+00  ,    0.00000000e+00 , 3.77881944e+00 , 1.10770431e+02,  2e-5 ,     1.11344490e+02  ,2.81156186e+00 , 1.00000000e+00])
    shinps = []
    vExps = np.array([251.36686641,  97.23989514, 226.15032099,   4.95635432,  72.12133565, 20.26019111,  88.34122117])

    
    Bout, tARR, Bsheath, tsheath, radfrac, isHit, vIS = run_case(inps, shinps, vExps)
    print (np.mean(Bout))
    #print(Bout)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(tARR, Bout[-1], 'k')
    plt.plot(tARR, Bout[0], 'r')
    plt.plot(tARR, Bout[1], 'b')
    plt.plot(tARR, Bout[2], 'g')
    plt.show()
    #print(Bsheath)