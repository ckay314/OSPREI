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


# variables that carry all the simulation params
global inps, shinps

# settings for the model and their defaults
global expansion_model, ISfilename, hasSheath, canPrint
expansion_model = 'None'
ISfilename = False
hasSheath = False
canPrint = False


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
        #print CME_shape[2]+CME_shape[1]+CME_shape[0], myb/CME_crossrad, vec_in
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

# -------------------------------------------------------------------------------------- #
def getFRprofile():
    # Main function for getting the flux rope profile by propagating a CME outward, 
    # determining when/where it is within the CME, and getting the magnetic field vector
    
    # Unpack the parameters from inps
    FFlat, FFlon, CMElat, CMElon, CMEtilt, CMEAW, CMESRA, CMESRB, CMEvr, CMEB, CMEH, tshift, CMEstart, vExp, FFr, rotspeed = inps[0], inps[1], inps[2], inps[3], inps[4], inps[5], inps[6], inps[7], inps[8], inps[9], inps[10], inps[11], inps[12], inps[13], inps[14], inps[15]
    
    # Convert from shape parameters to actual CME shape
    # Set up CME shape as [d, a, b, c]
    CME_shape = np.zeros(4)
    shapeC = np.tan(CMEAW*dtor) / (1. + CMESRB + np.tan(CMEAW*dtor) * (CMESRA + CMESRB)) 
    dtorang = (CMElon - FFlon) / np.sin((90.-CMEtilt) * dtor) * dtor
    CMEnose = 0.999*FFr # start just in front of sat dist, can't hit any closer 
    CME_shape[3] = CMEnose * shapeC
    CME_shape[1] = CME_shape[3] * CMESRA
    CME_shape[2] = CME_shape[3] * CMESRB 
    CME_shape[0] = CMEnose - CME_shape[1] - CME_shape[2]
    
    # Set up arrays to hold values over the spacecraft path
    obsBx = []
    obsBy = []
    obsBz = []
    tARR = []
    rCME = []
    radfrac = []
    
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
        
        # Calculate CME shape (self-simlar unless flagged off)
        # Self-similar
        if (enteredCME == False) or (expansion_model == 'Self-Similar'):
            CME_shape[3] = CMEnose * np.tan(CMEAW*dtor) / (1. + CMESRB + np.tan(CMEAW*dtor) * (CMESRA + CMESRB))
            CME_shape[1] = CME_shape[3] * CMESRA
            CME_shape[2] = CME_shape[3] * CMESRB 
        # Using expansion velocity
        elif expansion_model == 'vExp':
            CMEnose += vExp * dt/7e5
            CME_shape[2] += vExp * dt/7e5
        # Update front position within CME_shape regardless of expansion mode
        CME_shape[0] = CMEnose - CME_shape[1] - CME_shape[2]
	    
        # Check if currently in the CME and get position within
        minb, thetaT, thetaP, flagit, CME_crossrad = isinCME(FF_CMExyz, CME_shape)
        # Track the minimum distance from center -> Impact Parameter
        if np.abs(minb/CME_crossrad) < ImpParam: ImpParam = np.abs(minb/CME_crossrad)
        # If hasn't been flagged determine the magnetic field vector
        if flagit != -9999:
            enteredCME = True
            # If not set to expand flag it to stop after first contact
            if expansion_model != 'Self-Similar': flagExp = True
            # Get the toroidal and poloidal magnetic field based on distance
            Btor = CMEB * jv(0, 2.4 * minb / CME_crossrad)
            Bpol = CMEB * CMEH * jv(1, 2.4 * minb / CME_crossrad)
	        # Convert this into CME Cartesian coordinates
            tdir, pdir = getBvector(CME_shape, minb, thetaT, thetaP)
            Bt_vec = Btor * tdir
            Bp_vec = Bpol * pdir
            Btot = Bt_vec + Bp_vec
            # Convert to spacecraft coord system
            temp = rotx(Btot, -(90.-CMEtilt))
            temp2 = roty(temp, CMElat - FFlat) 
            BSC = rotz(temp2, CMElon - FFlon)
            # Append to output arrays
            obsBx.append(-BSC[0])
            obsBy.append(-BSC[1])
            obsBz.append(BSC[2])
            tARR.append(t/3600.)
            rCME.append(CMEnose)
            radfrac.append(minb/CME_crossrad)                
        else:
            # stop checking if exit CME
            if enteredCME: t = tmax+1
        # Move to next step in simulation
        t += dt
	    # CME nose moves to new position
        CMEnose += CMEvr * dt / 7e5 # change position in Rsun
	    # Update the total magnetic field -> should decrease as expands
        if (enteredCME == False) or (expansion_model == 'Self-Similar'):
            CMEB *= ((CMEnose - CMEvr * dt / 7e5) / CMEnose)**2
        elif expansion_model == 'vExp':
            CMEB *= ((CME_shape[2] - vExp * dt / 7e5) / CME_shape[2])**2
	    # Include the orbit of the satellite/Earth
        FFlon += dt * rotspeed
        
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
    return Bout, tARR, isHit, ImpParam, np.array(radfrac)
    
# -------------------------------------------------------------------------------------- #
def run_case(inpsIn, shinpsIn):
    # Take the values passed in and set to globals for convenience
    global inps, shinps
    inps = inpsIn
    shinps = shinpsIn
    
    # use to set sheath end
    CMEstart = inps[12]
    
    # run the simulation    
    Bout, tARR, isHit, ImpParam, radfrac = getFRprofile()
    # execute extra functions as needed
    if isHit:
        # Add sheath if desired
        if hasSheath:
            # Given Br and Bphi in shinps-> change to taking in [B, Bx, By, Bz]?
            upB = np.array([np.sqrt(shinps[4]**2+shinps[5]**2),-shinps[4],-shinps[5], 0.])   
            # Calculate compressed magnetic field            
            jumpvec = upB * shinps[2] 
            # Create a sheath by connecting jumpvec to front of flux rope
            tsheath, sheathB = make_sheath(jumpvec, Bout, CMEstart, shinps)  
            Bsheath = np.array([sheathB[1], sheathB[2], sheathB[3], sheathB[0]])
        else:
            Bsheath = []
            tsheath = []
    # No impact case
    else:
        if canPrint: print ('No impact expected')
      
    return Bout, tARR, Bsheath, tsheath, radfrac 

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
    tSheath = np.linspace(CMEstart-shinps[1]/24.,CMEstart,20)
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
def calcSheathInps(CMEstart, vels, nSW, BSW, satr, cs=49.5, vA=55.4):
    # Given basic SW and CME properties calculate the sheath parameters using the
    # methods from the 2020 FIDO-SIT paper
    # vels should be [vCME at FIDO, vtransit (avg), vSW]
    # Assume B in is Br, was calc at 213 so scale if need be
    # Use a simple Parker model for mag field
    Br = BSW * (213./satr)**2
    Bphi = -Br * (2.7e-6) * satr * 7e5 / vels[3]
    BSW = np.sqrt(Br**2+Bphi**2)
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
    return [CMEstart-sheathTime/24., sheathTime, compression, sheathv, Br, Bphi]
    
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
def read_more_inputs(inputs, input_values):
    possible_vars = ['FR_B0', 'FR_pol', 'CME_start', 'CME_stop', 'Expansion_Model', 'CME_vExp', 'CME_v1AU', 'vTrans']
    # if matches add to dictionary
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
    return input_values


if __name__ == '__main__':
    inps = [4.131, 80.83, -12.280000102840006, 90.99999979431998, 38.37, 64.99999992222988, 0.7, 0.6, 500.0, -70.0, 1.0, 0.0, 2.45, 0.0, 213.0, 1.1407524220455427e-05]    
    Bout, tARR, isHit, ImpParam, radfrac = run_case(inps, [])
