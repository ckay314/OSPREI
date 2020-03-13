import numpy as np
import math
import sys

global rsun, dtor, radeg, kmRs
rsun  =  7e10		 # convert to cm, 0.34 V374Peg
dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
kmRs  = 1.0e5 / rsun # km (/s) divided by rsun (in cm)


#-----------geometry functions ------------------------------------

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
    
def processANTinputs(input_values):
    Epos = [0.,0.,213.,360/365.256/24./60.] # sat_lat, sat_lon, sat_rad, sat_rod
    try: 
        Epos[0] = float(input_values['Sat_lat'])
    except:
        print('Assuming satellite at 0 lat')
    try: 
        Epos[1] = float(input_values['Sat_lon'])
    except:
        print('Assuming satellite at 0 lon')
    try: 
        Epos[2] = float(input_values['Sat_rad'])
    except:
        print('Assuming satellite at 213 Rs (L1)')
    try: 
        Epos[3] = float(input_values['Sat_rot'])
    except:
        print('Assuming satellite orbits at Earth orbital speed')
    try: 
        Cd = float(input_values['Cd'])
    except:
        print('Assuming radial Cd = 1')
        Cd = 1.
    try: 
        nSW = float(input_values['nSW'])
    except:
        nSW = -9999
    try: 
        vSW = float(input_values['vSW'])
    except:
        vSW = -9999
    try: 
        BSW = float(input_values['BSW'])
    except:
        BSW = -9999
    try: 
        cs = float(input_values['cs'])
    except:
        cs = -9999
    try: 
        vA = float(input_values['vA'])
    except:
        vA = -9999
    
    return Epos, Cd, [nSW, vSW, BSW, cs, vA]    
    
        
    

# -------------- main function ------------------
def getAT(invec, rCME, Epos, silent=False, SSscale =1.):
    
    Elat      = Epos[0]
    Elon0     = Epos[1]
    Er        = Epos[2]
    Erotrate  = Epos[3]
    CMElat    = invec[0]
    CMElon    = invec[1]
    CMEtilt   = invec[2]    
    CMEvel0   = invec[3] * 1e5
    CMEmass   = invec[4] * 1e15
    CMEAW     = invec[5] * dtor
    CMEA      = invec[6]
    CMEB      = invec[7]
    vSW       = invec[8] * 1e5
    SWrho0    = invec[9] * 1.67e-24
    Cd        = invec[10]    

    CdivR = np.tan(CMEAW) / (1. + CMEB + np.tan(CMEAW) * (CMEA + CMEB))

    #Erotrate = 360/365./24/60. # deg per min
    # start w/o Earth rot so not considering changes in Elon due to diff in transit time
    #Erotrate = 0.
    
    # calculate any E rot between the time given and the start of the ANTEATR simulation
    #dElon1 = (t20-tGCS) * Erotrate

    vCME = CMEvel0
    Elon = Elon0
    vExp = SSscale*vCME / (1. + (1+CMEB+CMEA*np.tan(CMEAW))/(np.tan(CMEAW)*CMEB)) # if vCME = vNose and self-sim
    bCME = CdivR * rCME * CMEB
    
    dt = 1
    t  = 0
    # run CME nose to 1 AU
    printR = rCME
    # arrays for saving profiles
    outTs     = []
    outRs     = []
    outvTots  = []
    outvBulks = []
    outvExps  = []
    outAWs    = []
    while rCME <= Er:
        t += dt
        rCME += vCME*dt*60/7e10 
        bCME += vExp*dt*60/7e10
        CtimesR = bCME / CMEB
        #CtimesR = CdivR * rCME
        rcent = rCME - (CMEA+CMEB) * CtimesR
        CMEAW = np.arctan(CtimesR*(1+CMEB)/rcent)
        CMEarea = 4*CMEB*CtimesR**2 * (7e10)**2
        SWrho = SWrho0 * (Er/rcent)**2
        dV = -Cd * CMEarea * SWrho * (vCME-vExp-vSW) * np.abs(vCME-vExp-vSW) * dt * 60 / CMEmass
        vCME += dV
        Elon += Erotrate * dt
        vExp = SSscale*vCME / (1. + (1+CMEB+CMEA*np.tan(CMEAW))/(np.tan(CMEAW)*CMEB)) # if vCME = vNose and self-sim
        if (rCME>printR): 
            printR += 5
            outTs.append(t/60./24.)
            outRs.append(rCME)
            outvTots.append(vCME/1e5)
            outvBulks.append((vCME-vExp)/1e5)
            outvExps.append(vExp/1e5)
            outAWs.append(CMEAW*180./3.14159)
        #print vCME/1e5, vCME / (1. + (1+CMEB+CMEA*np.tan(CMEAW))/(np.tan(CMEAW)))/1e5
    ElonEr = Elon # Earth lon CME reaches the Earth distance, will use to start FIDO
    inCME = False
    prevmin = 9999.
    while not inCME:
        t += dt
        rCME += vCME*dt*60/7e10 
        bCME += vExp*dt*60/7e10
        CtimesR = bCME / CMEB
        #CtimesR = CdivR * rCME
        rcent = rCME - (CMEA+CMEB) * CtimesR
        CMEAW = np.arctan(CtimesR*(1+CMEB)/rcent)
        #print t, rCME, vCME/1e5, vExp/1e5, CMEAW*180./3.14159
        #if rCME > 220: print asf
        CMEarea = 4*CMEB*CtimesR**2 * (7e10)**2
        SWrho = SWrho0 * (Er/rcent)**2
        dV = -Cd * CMEarea * SWrho * (vCME-vExp-vSW) * np.abs(vCME-vExp-vSW) * dt * 60 / CMEmass
        vCME += dV
        Elon += Erotrate * dt
        vExp = SSscale*vCME / (1. + (1+CMEB+CMEA*np.tan(CMEAW))/(np.tan(CMEAW)*CMEB)) # if vCME = vNose and self-sim
        thetas = np.linspace(-math.pi/2, math.pi/2,1001)
        Epos1 = SPH2CART([Er,Elat,Elon])
        temp = rotz(Epos1, -CMElon)
        temp2 = roty(temp, CMElat)
        Epos2 = rotx(temp2, 90.-CMEtilt)
        xFR = rcent + CMEA*CtimesR*np.cos(thetas)
        zFR = CtimesR * np.sin(thetas) 
        dists2 = ((Epos2[0] - xFR)**2 + Epos2[1]**2 + (Epos2[2] - zFR)**2) / (CMEB*CtimesR)**2
        CAang = thetas[np.where(dists2 == np.min(dists2))]
        thismin = np.min(dists2)
        if (rCME>printR): 
            printR += 5
            outTs.append(t/60./24.)
            outRs.append(rCME)
            outvTots.append(vCME/1e5)
            outvBulks.append((vCME-vExp)/1e5)
            outvExps.append(vExp/1e5)
            outAWs.append(CMEAW*180./3.14159)
        #print rCME, thismin, Er,Elat,Elon
        #print thismin, rCME, CMEA*CtimesR, CMEB*CtimesR
        if thismin < 1:
            TT = t/60./24.
            outTs.append(t/60./24.)
            outRs.append(rCME)
            outvTots.append(vCME/1e5)
            outvBulks.append((vCME-vExp)/1e5)
            outvExps.append(vExp/1e5)
            outAWs.append(CMEAW*180./3.14159)
            if not silent:
                print ('Transit Time:     ', TT)
                print ('Final Velocity:   ', vCME/1e5)
                print ('CME nose dist:    ', rCME)
                print ('Earth longitude:  ', Elon)
            #print Elon, rCME, vCME/1e5#,CAang[0]/dtor, np.tan(Epos2[1]/(CMEB*CtimesR))/dtor
            #print CAang[0]/dtor
            inCME = True
            return np.array([outTs, outRs, outvTots, outvBulks, outvExps, outAWs]), Elon, ElonEr       
        elif thismin < prevmin:
            prevmin = thismin
        else:
            return np.array([[9999], [9999], [9999], [9999], [9999], [9999]]), 9999, 9999 

#invec = [CMElat, CMElon, CMEtilt, CMEvel0, CMEmass, CMEAW, CMEA, CMEB, vSW, SWrho0, Cd]
# Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth!
#invec = [-5.,3.,30.,262.,0.56, 17., 0.75, 0.35, 450., 5.0, 1.]               
#print getAT(invec, 21.5,[4.763,0,213.])
