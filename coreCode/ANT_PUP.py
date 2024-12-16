import numpy as np
import math
import sys
import os.path
from scipy.interpolate import CubicSpline
import pickle
import empHSS as emp
from scipy.special import ellipk, ellipe


global rsun, dtor, radeg, kmRs
rsun  =  7e10		 # convert to cm, 0.34 V374Peg
dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees
kmRs  = 1.0e5 / rsun # km (/s) divided by rsun (in cm)
pi = 3.14159

# Empirical function to calculate length of axis
global lenFun
lenCoeffs = [0.61618, 0.47539022, 1.95157615]
lenFun = np.poly1d(lenCoeffs)


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
    
def processANTinputs(input_values, hasPath=False):
    Epos = [0.,0.,213.,1.141e-5] # sat_lat, sat_lon, sat_rad, sat_rot
    # sat rot default is  360/365.256/24./60./60.
    try: 
        Epos[0] = float(input_values['SatLat'])
    except:
        if 'satPath' not in input_values:
            print('Assuming satellite at 0 lat')
    try: 
        Epos[1] = float(input_values['SatLon'])
    except:
        if not hasPath:
            if input_values['satPath'][-4:] != 'sats':
                print('Assuming satellite at 0 lon')
    try: 
        Epos[2] = float(input_values['SatR'])
    except:
        if not hasPath:
            print('Assuming satellite at 213 Rs (L1)')
    try: 
        Epos[3] = float(input_values['SatRot'])
    except:
        if not hasPath:
            print('Assuming satellite orbits at Earth orbital speed')
    try: 
        Cd = float(input_values['SWCd'])
    except:
        print('Assuming radial Cd = 1')
        Cd = 1.
    try: 
        nSW = float(input_values['SWn'])
    except:
        nSW = -9999
    try: 
        vSW = float(input_values['SWv'])
    except:
        vSW = -9999
    try: 
        BSW = float(input_values['SWB'])
    except:
        BSW = -9999
    try: 
        cs = float(input_values['SWcs'])
    except:
        cs = -9999
    try: 
        vA = float(input_values['SWvA'])
    except:
        vA = -9999
    
    return Epos, Cd, [nSW, vSW, BSW, cs, vA]    
    
def getAxisF(deltax, deltap, bp, c, B0, cnm, tau, rho):
    # Nose Direction
    kNose = 1.3726 * deltax / c   
    RcNose = 1./kNose   
    gammaN = bp / RcNose        
    dg = deltap * gammaN
    if dg > 0.9:
        dg = 0.9
        gammaN = dg / deltap
    dg2 = deltap**2 * gammaN**2 
    if dg2 > 1:
        return 9999, 9999
    coeff1N = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6) 
    toAccN = deltap * pi * bp**2 * RcNose * rho
    coeff2N = - deltap**3 *(tau*(tau-1)+0.333)/ 4. * gammaN
    aTotNose = (coeff2N+coeff1N) * B0**2 * RcNose * bp / toAccN 
    # Edge Direction
    kEdge = 40 * deltax / c / np.sqrt(16*deltax+1)**3
    RcEdge = 1./kEdge    
    gammaE = bp / RcEdge
    dg = deltap*gammaE
    if dg > 0.9:
        dg = 0.9
        gammaE = 0.9 / deltap
    dg2 = deltap**2 * gammaE**2 
    if dg2 < 1:
        coeff1E = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6) 
        coeff2E = - deltap**3 *(tau*(tau-1)+0.333) *  gammaE / 4.
        toAccE = deltap * pi * bp**2 * RcEdge * rho
        aTotEdge =  (coeff1E+coeff2E) * B0**2 * RcEdge * bp / toAccE       
        return aTotNose, aTotEdge
    else:
        return 9999, 9999
    
def getCSF(deltax, deltap, bp, c, B0, cnm, tau, rho, Btot2, csTens):
    # Internal cross section forces
    coeff3 = -deltap**3 * 2/ 3/(deltap**2+1)/cnm**2
    # Option to turn of cross-section tension
    if not csTens: coeff3 = 0.
    coeff4 = deltap**3*(tau/3. - 0.2)
    coeff4sw = deltap*(Btot2/B0**2) / 8. #* 2
    kNose = 1.3726 * deltax / c   
    RcNose = 1./kNose   
    toAccN = deltap * pi * bp**2 * RcNose * rho
    aBr = (coeff3 + coeff4 - coeff4sw) * B0**2 * bp * RcNose / toAccN / deltap
    return aBr
    
def getThermF(CMElens, temCME, nCME, SWfront, SWfrontB, SWedge):
    # Get temperatures
    temSWf = SWfront[4]
    temSWb = SWfrontB[4] 
    temSWe = SWedge[4] 
    
    # Scale densities
    nF = SWfront[0] / 1.67e-24 
    nE = SWedge[0] / 1.67e-24 
    nB = SWfrontB[0]  / 1.67e-24 
    
    # average n*T of front and back
    avgNT = 0.5 * (nB * temSWb + nF * temSWf)    
    
    # Calc difference in pressure
    delPr =  1.38e-16 * 2*(nCME * temCME - avgNT)
    delPp =  1.38e-16 * 2*(nCME * temCME - nE * temSWe)

    # Calc gradients
    gradPr = delPr / CMElens[3] 
    gradPp = delPp / CMElens[4]
    return gradPr, gradPp
    
def getDrag(CMElens, vs, Mass, AW, AWp, Cd, SWfront, SWfrontB, SWedge, SWedgeB, ndotz):
    # dragAccels = [dragR, dragEdge, bulk, dragBr, dragBp, dragA, dragC]
    dragAccels = np.zeros(7)
    rhoSWn1 = SWfront[0]
    rhoSWn2 = SWfrontB[0]
    vSWn1 = SWfront[1]
    vSWn2 = SWfrontB[1]
    # Radial Drag
    CMEarea = 4*CMElens[1]*CMElens[4] 
    dragF1 = -Cd*CMEarea*rhoSWn1 * (vs[0]-vSWn1) * np.abs(vs[0]-vSWn1) / Mass
    vFback = vs[0] - 2*vs[3]
    dragF2 = -Cd*CMEarea*rhoSWn2 * (vFback-vSWn2) * np.abs(vFback-vSWn2) / Mass
    dragF = 0.5 * (dragF1 + dragF2)
    CSsquishF = 0.5 * (dragF1 - dragF2) 
    # Edge Drag
    CMEarea2 = 2*CMElens[4]*(CMElens[5] + CMElens[3])
    rhoSWe1 = SWedge[0]
    rhoSWe2 = SWedgeB[0]
    vSWe1 = SWedge[1]
    vSWe2 = SWedgeB[1]
    dragE1 = -Cd*CMEarea2*rhoSWe1 * (vs[1]-np.sin(AW)*vSWe1) * np.abs(vs[1]-np.sin(AW)*vSWe1)  / Mass
    #dragE2 = -Cd*CMEarea2*rhoSWe2 * (vs[1]-np.sin(AW)*vSWe2) * np.abs(vs[1]-np.sin(AW)*vSWe2)  / Mass    
    dragE2 = -Cd*CMEarea2*rhoSWe2 * (vs[1]-2*vs[3]/ndotz-np.sin(AW)*vSWe2) * np.abs(vs[1]-2*vs[3]/ndotz-np.sin(AW)*vSWe2)  / Mass   
    dragE = 0.5 * (dragE1 + dragE2)
    CSsquishE = 0.5 * (dragE1 - dragE2)
    CSsquish = 0.5 * (CSsquishF + CSsquishE)
    # CS Perp Drag
    CMEarea3 = 2*CMElens[3]*lenFun(CMElens[5] / CMElens[6]) * CMElens[6]
    dragAccels[4] = -Cd*CMEarea3*rhoSWn1 * (vs[4]-np.sin(AWp)*vSWn1) * np.abs(vs[4]-np.sin(AWp)*vSWn1) / Mass 
    # Individual components  
    dragAccels[0] = dragF + CSsquish
    dragAccels[1] = dragE 
    dragAccels[2] = dragF* (vs[2]/(vs[2]+vs[5])) 
    dragAccels[3] = CSsquish 
    dragAccels[5] = dragF * (vs[5]/(vs[2]+vs[5]))
    dragAccels[6] = dragAccels[1] - CSsquish / ndotz   
    
    # old version of drag
    #dragAccels[0] = -Cd*CMEarea*rhoSWn1 * (vs[0]-vSWn1) * np.abs(vs[0]-vSWn1) / Mass
    #dragAccels[1] = -Cd*CMEarea2*rhoSWe1 * (vs[1]-np.sin(AW)*vSWe1) * np.abs(vs[1]-np.sin(AW)*vSWe1)  / Mass
    '''dragAccels[2] = dragAccels[0] * (vs[2]/vs[0]) 
    dragAccels[3] = dragAccels[0] * (vs[3]/vs[0])
    dragAccels[5] = dragAccels[0] * (vs[5]/vs[0])
    dragAccels[6] = dragAccels[1] - dragAccels[0] * (vs[3]/vs[0]) / ndotz'''
    
    return dragAccels
    
def calcYawTorque(CMElens, CMEpos, AW, vs, deltax, deltap, yaw, solRotRate, SWfs, Cd, doMH, saveForces=False):        
    # Get radial distance at edges
    # Get xy in CME frame 
    rEdge = CMElens[1]
    rxyL = np.array([CMElens[2], 0, CMElens[1]])
    rxyR = np.array([CMElens[2], 0, -CMElens[1]])
    
    # subtract front distance 
    rxyL[0] -= CMElens[0]
    rxyR[0] -= CMElens[0]

    # rotate by yaw
    rxyL2 = roty(rxyL, -yaw)
    rxyL2[0] += CMElens[0]

    rxyR2 = roty(rxyR, -yaw)
    rxyR2[0] += CMElens[0]
    
    
    rLside = np.sqrt(rxyL2[0]**2 + rxyL2[2]**2)
    rRside = np.sqrt(rxyR2[0]**2 + rxyR2[2]**2)
    
    #Need to project edges onto eq plane to get lon diff for HSS -> rest of stuff ok
    # to assume everything in xz plane
    rxyL3 = rotx(rxyL2, -(90.-CMEpos[2]))
    rxyL4 = roty(rxyL3, -CMEpos[0])
    rxyR3 = rotx(rxyR2, -(90.-CMEpos[2]))
    rxyR4 = roty(rxyR3, -CMEpos[0])
    
    #dlonLSide = np.arctan2(rxyL2[2], rxyL2[0]) / dtor
    #dlonRSide = np.arctan2(rxyR2[2], rxyR2[0]) / dtor
    
    dlonLSide = np.arctan2(rxyL4[1], rxyL4[0]) / dtor
    dlonRSide = np.arctan2(rxyR4[1], rxyR4[0]) / dtor
    
    dtMHL = -dlonLSide * solRotRate 
    dtMHR = -dlonRSide * solRotRate
    
    rLsideProj = np.sqrt(rxyL4[1]**2 + rxyL4[0]**2)
    rRsideProj = np.sqrt(rxyR4[1]**2 + rxyR4[0]**2)
    SWatsatL  = getSWvals(rLsideProj, SWfs, doMH=doMH, returnReg=False, deltatime=dtMHL)
    SWatsatR  = getSWvals(rRsideProj, SWfs, doMH=doMH, returnReg=False, deltatime=dtMHR)

    # Get velocity at front of each side
    vEdgeL, vExpL = getvCMEframe(1, math.pi/2, 0, deltax, deltap, vs) 
    vEdgeR, vExpR = getvCMEframe(1, -math.pi/2, 0, deltax, deltap, vs)
    
    vEdgeL = roty(vEdgeL, -yaw)
    vEdgeR = roty(vEdgeR, -yaw)
    
    rhatCMEframeL = np.array([np.cos(np.abs(dlonLSide*dtor)), 0, np.sin(dlonLSide*dtor)])
    rhatCMEframeR = np.array([np.cos(np.abs(dlonRSide*dtor)), 0, np.sin(dlonRSide*dtor)])

    vrEdgeL = np.dot(rhatCMEframeL, vEdgeL)
    vrEdgeR = np.dot(rhatCMEframeR, vEdgeR)
    
    CMEarea = 2*CMElens[1]*CMElens[4] # Only half the CME length
    dragL = -Cd*CMEarea*SWatsatL[0] * (vrEdgeL-SWatsatL[1]) * np.abs(vrEdgeL-SWatsatL[1]) 
    dragR = -Cd*CMEarea*SWatsatR[0] * (vrEdgeR-SWatsatR[1]) * np.abs(vrEdgeR-SWatsatR[1])
            
    # need to take angle still
    lever = np.sqrt((CMElens[6]+CMElens[3])**2 + (CMElens[4]+CMElens[3])**2)
    lev_angL = np.arctan2((CMElens[0] - rxyL2[0]), rxyL2[2])
    norm_vecL = np.array([np.cos(lev_angL), 0., np.sin(lev_angL)])
    lev_angR = np.arctan2((CMElens[0] - rxyR2[0]), np.abs(rxyR2[2]))
    norm_vecR = np.array([np.cos(lev_angR), 0., -np.sin(lev_angR)])
    torque = -( np.dot(lever*norm_vecL, dragL*rhatCMEframeL) - np.dot(lever*norm_vecR, dragR*rhatCMEframeR))
    if saveForces:
        moarOuts = [-np.dot(lever*norm_vecL, dragL*rhatCMEframeL), np.dot(lever*norm_vecR, dragR*rhatCMEframeR), lever, dragL, dragR, np.dot(norm_vecL, rhatCMEframeL), np.dot(norm_vecR, rhatCMEframeR)]
        return torque, moarOuts
    else:
        return torque

def IVD(vFront, AW, AWp, deltax, deltap, CMElens, alpha, ndotz, fscales):
    vs = np.zeros(7)
    # vs = [vFront 0, vEdge 1, vBulk 2, vexpBr 3, vexpBp 4, vexpA 5, vexpC 6]
    vs[0] = vFront

    f1 = fscales[0]
    f2 = fscales[1]
    # Calculate the nus for conv and self sim
    nu1C = np.cos(AWp)
    nu2C = np.cos(AW)
    nu1S = (CMElens[2] + CMElens[5]) / CMElens[0]
    nu2S =  (CMElens[2]) / CMElens[0]
    # Take the appropriate fraction of each
    nu1 = f1 * nu1C + (1-f1) * nu1S
    nu2 = f2 * nu2C + (1-f2) * nu2S
        
    vs[2] = nu2 * vs[0]
    vs[1] = vs[2] * np.tan(AW)    
    vs[5] = nu1 * vs[0] - vs[2]
    vs[4] = nu1 * vs[0] * np.tan(AWp)
    vs[3] = vs[0] * (1 - nu1)
    vs[6] = vs[0] * (nu2 * np.tan(AW) - alpha * (1 - nu1))    
    
    return vs
    
def initCMEparams(deltaAx, deltaCS, AW, AWp, CMElens, Mass, printNow=False):
    # CME shape [CMEnose, rEdge, d, rr, rp, Lr, Lp]
    # alpha * CMElens[3] is the cross sec width in z dir
    CMElens[4] = np.tan(AWp) / (1 + deltaCS * np.tan(AWp)) * CMElens[0]
    CMElens[3] = deltaCS * CMElens[4]
    CMElens[6] = (np.tan(AW) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + deltaAx * np.tan(AW))  
    #CMElens[6] = CMElens[3] / deltaCSAx 
    CMElens[5] = deltaAx * CMElens[6]
    CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
    CMElens[1] = CMElens[2] * np.tan(AW) 
    # alpha is scaling of rr to rE - Lp
    # (suspect this should have been updated for new parabola shape and never was...
    # not a huge diff tho? same in update CME - CK 23/05/31)
    alpha = (CMElens[1]-CMElens[6])/CMElens[3]
    ndotz = 4*deltaAx / np.sqrt(1 + 16 * deltaAx**2)
    
    # Initiate CME density
    vol = pi*CMElens[3]*CMElens[4] *  lenFun(CMElens[5]/CMElens[6])*CMElens[6]
    rho = Mass / vol    
    return alpha, ndotz, rho
    
def updateSW(Bsw0, BphiBr, vSW, n1AU, rFinal, rFront0, CMElens):
    thisR = CMElens[0] - CMElens[3]
    sideR = np.sqrt(CMElens[2]**2 + (CMElens[1]-CMElens[3])**2)
    avgAxR = 0.5*(CMElens[0] - CMElens[3] + sideR)
    BSWr = Bsw0 * (rFront0 / avgAxR)**2
    BSWphi = BSWr * 2.7e-6 * avgAxR / vSW
    Btot2 = BSWr**2 + BSWphi**2
    rhoSWn = 1.67e-24 * n1AU * (rFinal/CMElens[0])**2
    rhoSWe = 1.67e-24 * n1AU * (rFinal/(CMElens[1]))**2
    return Btot2, rhoSWn, rhoSWe
    
def updateCME(CMElens, Mass):
    # Calc shape params and new AWs
    deltaCS = CMElens[3] / CMElens[4]
    deltaAx = CMElens[5] / CMElens[6]
    deltaCSAx = CMElens[3] / CMElens[6]
    alpha = (CMElens[1]-CMElens[6])/CMElens[3]
    ndotz =  4*deltaAx / np.sqrt(1 + 16 * deltaAx**2)
    AW = np.arctan((CMElens[1])/CMElens[2]) 
    AWp = np.arctan(CMElens[4]/(CMElens[2] + CMElens[5]))
    # Update CME density
    vol = pi*CMElens[3]*CMElens[4] *  lenFun(CMElens[5]/CMElens[6])*CMElens[6]
    rho = Mass / vol        
    return deltaCS, deltaAx, deltaCSAx, alpha, ndotz, AW, AWp, rho
    
def makeSWfuncs(fname, time=None, doAll=False, isArr=False):
    SWfs = []
    # assume n in cm^-3, v in km/s, B in nT, T in K
    # rs in AU whether in text file or passed
    # the output functions are in g, cm/s, G, K and take r in cm

    # check whether fname is a file or the 1 AU sw params
    if isinstance(fname, str):
        ext = fname[-3:]
        if ext != 'npy':
            data = np.genfromtxt(fname, dtype=float)
            rs = data[:,0]  * 1.5e13
            ns = data[:,1]
            vs = data[:,2]
            Brs = data[:,3]
            Blons = data[:,4]
            Ts = data[:,5]
            if doAll:
                vclts = data[:,6]
                vlons = data[:,7]
                Bclts = data[:,8]
                fvlon = CubicSpline(rs, vlons * 1e5, bc_type='natural')
                fvclt = CubicSpline(rs, vclts * 1e5, bc_type='natural')
                fBclt = CubicSpline(rs, Bclts / 1e5, bc_type='natural')
                
        else:
            # dict keys
            #'Blon[nT]', 'vclt[km/s]', 'T[K]', 'n[1/cm^3]', 'vr[km/s]', 'Br[nT]', 'vlon[km/s]', 'Bclt[nT]', 'r[AU]'
            idx = 0
            if time != None: idx=time
            data = np.atleast_1d(np.load(fname, allow_pickle=True, encoding = 'latin1'))[idx] 
            rs = np.array(data['r[AU]']) * 1.5e13
            ns = np.array(data['n[1/cm^3]'])
            vs = np.array(data['vr[km/s]'])
            Brs = np.array(data[ 'Br[nT]']) 
            Blons = np.array(data['Blon[nT]']) 
            Ts = np.array(data['T[K]'])   
            if doAll:
                vclts = np.array(data['vclt[km/s]'])
                vlons  = np.array(data['vlon[km/s]'])
                Bclts  = np.array(data['Bclt[nT]']) 
                fvlon = CubicSpline(rs, vlons * 1e5, bc_type='natural')
                fvclt = CubicSpline(rs, vclts * 1e5, bc_type='natural')
                fBclt = CubicSpline(rs, Bclts / 1e5, bc_type='natural')
                
        # make functions relating r to the 5 parameters
        frho = CubicSpline(rs,  1.67e-24 * ns, bc_type='natural')
        fv = CubicSpline(rs, vs * 1e5, bc_type='natural')
        fBr = CubicSpline(rs, Brs / 1e5, bc_type='natural')
        fBlon = CubicSpline(rs, Blons / 1e5, bc_type='natural')
        fT = CubicSpline(rs, Ts, bc_type='natural')   
        
            
    else:
        nSW = fname[0]
        vSW = fname[1] * 1e5
        BSW = fname[2]
        TSW = fname[3]
        rSW = fname[4] * 1.5e13
        # n profile
        frho = lambda x: 1.67e-24 * nSW * (rSW/x)**2        
        # assume constant v
        if isArr:
            fv = lambda x: vSW*np.ones(len(x))
        else:
            fv = lambda x: vSW
        # B profile
        BphiBr = 2.7e-6 * rSW  / vSW 
        Br_rSW = BSW / np.sqrt(1+BphiBr**2)
        # scale Br based on distance
        fBr = lambda x: Br_rSW * (rSW/x)**2  / 1e5   
        # lon = Br * parker factor
        fBlon = lambda x: Br_rSW * (rSW/x)**2 * 2.7e-6 * x / vSW  / 1e5                   
        # T profile
        #fT = lambda x: TSW * np.power(x/rSW, -0.58)  
        fT = lambda x: TSW * np.power(x/rSW, -1.)  
        # return empty functions for the other parameters we don't have in empirical model
        if isArr:
            fvlon = lambda x: 0*np.ones(len(x))
            fvclt = lambda x: 0*np.ones(len(x))
            fBclt = lambda x: 0*np.ones(len(x))
        else:
            fvlon = lambda x: 0
            fvclt = lambda x: 0
            fBclt = lambda x: 0
        
    if doAll:
        return [frho, fv, fBr, fBlon, fT, fvlon, fvclt, fBclt]  
    else:
        return [frho, fv, fBr, fBlon, fT]  

def getSWvals(r_in, SWfuncs, doMH=False, returnReg=False, deltatime=0):
    # returns the actual values of n, vr, Br, Blon, T
    SWvec = np.zeros(5)
    for i in range(5):
        SWvec[i] = SWfuncs[i](r_in)
    if hasattr(doMH, '__len__'):
        if doMH:
            MHouts, HSSreg = emp.getHSSprops(r_in/1.5e13, doMH[0]+deltatime, doMH[1], doMH[2], doMH[3], doMH[4], doMH[5], returnReg=True)
        else:
            MHouts = emp.getHSSprops(r_in/1.5e13, doMH[0], doMH[1], doMH[2], doMH[3], doMH[4], doMH[5])
            
        SWvec = MHouts[:-1]*SWvec # drop vlon term
    if returnReg:
        return SWvec, HSSreg
    else:
        return SWvec 
    
def getr(rmin, dr, nr, beta, theta, gam, vCME, vSW, vA):
    r = rmin
    diffs = []
    for i in range(nr+1):
        r+= dr
        a = 0.5 * (gam + 1 - r * (gam -1))
        term3 = a
        term2 = -2*a*np.cos(theta)**2 - beta - np.sin(theta)**2 * (a + 0.5*(r-1))
        term1 = np.cos(theta)**2 * (a + 2*beta*np.cos(theta)**2)
        term0 = -beta * np.cos(theta)**4
        coeffs = [term3, term2, term1, term0]
        roots = np.roots(coeffs)
        if True in np.isreal(roots):
            MAs = np.sqrt(r * roots)
            # start with just first term
            u1 = -vA * np.real(MAs[0])
            vS = vSW - u1
            r2 = (vSW - vS) / (vCME - vS)
            if r2 > 1:
                diff = r - r2
                diffs.append(diff)
            else:
                diffs.append(9999)
        else:
            diffs.append(9999)
    idx = np.where(np.abs(diffs) == np.min(np.abs(diffs)))
    if diffs[idx[0][0]] == 9999:
        if ((vCME-vSW)/vA) > 10 and (rmin==1):
           r = getr(3.7, 0.01, 29, beta, theta, gam, vCME, vSW, vA)
        elif ((vCME-vSW)/vA) > 10 and (rmin==1):
            return 9999
        else:
            return 9999
    else:
        r = rmin+dr*(idx[0][0]+1)
    return r
    
def getObSheath(vCME, vSW, theta, cs, vA, gam=5/3):
    beta = cs**2 / vA**2
    r = getr(1.00, 0.1, 29, beta, theta, gam, vCME, vSW, vA)
    if r != 9999:
        r = getr(r-0.1, 0.01, 19, beta, theta, gam, vCME, vSW, vA)
    if r != 9999:
        r = getr(r-0.01, 0.001, 19, beta, theta, gam, vCME, vSW, vA)
    if r == 9999:
        vS, prat = 9999, 9999
    else:        
        vS = (r * vCME - vSW) / (r-1)
        u1 = vSW-vS
        Ma2 = (u1/vA)**2
        prat1 = 1 + gam * (u1/cs)**2 * (1 - 1/r)
        prat2 = 0.5 * np.sin(theta)**2 * gam * (vA/cs)**2 * (1 - (r*(Ma2-np.cos(theta)**2))**2 / (Ma2-r*np.cos(theta)**2)**2)
        prat = prat1 + prat2
        
        prat2B = -gam/2 * (u1**2 *vA**2 *np.sin(theta)**2/cs**2) * (r-1) * (u1**2 * (r+1) - 2*r*vA**2*np.cos(theta)**2) / (u1**2 - r*vA**2 *np.cos(theta)**2)**2
        if prat < 0:
            r, vS, prat = 9999, 9999, 9999
    return r, vS, prat
    
def print2file(vec, fname, fmat):
    outstuff = ''
    for item in vec:
        outstuff = outstuff + fmat.format(item) + ' '
    fname.write(outstuff+'\n')

def print2screen(vec, prefix=''):
    outstuff = prefix
    for item in vec:
        outstuff = outstuff + '{:6.2f}'.format(item) + ' '
    print (outstuff)
    
def makeFailArr(val):
    return np.array([[val]*11]), [[val]*6], [val, val, val, val, val, val], [[val]*2], [val, val, val, val], [[val]*16], [[val]*8]

def add2outs(outsCME, outsSheath, CMEarr, sheatharr):
    # CME: 0 t, 1 r, 2 vFront, 3 AW, 4 AWp, 5 delAx, 6 delCS, 7 delCA, 8 B, 9 Cnm, 10 n, 11 Temp, 12 yaw, 13 yaw v, 14 reg, 15 vEdge 16 vCent 17  vrr 18 vrp 19 vLr 20 vLp 21 vXCent]
    # Sheath: 0 vS, 1 comp, 2 MA, 3 Wid, 4 Dur, 5 Mass, 6 Dens, 7 B, 8 Theta, 9 Temp, 10 Vt, 11 InSheath

    #fullCMEstuff = [0 t/3600./24., 1 CMElens[0]/rsun, 2 AW*180/pi,  3 AWp*180/pi, 4 CMElens[5]/rsun, 5 CMElens[3]/rsun, 6 CMElens[4]/rsun, 7 CMElens[6]/rsun, 8 CMElens[2]/rsun, 9 vs[0]/1e5, 10 vs[1]/1e5, 11 vs[5]/1e5, 12 vs[3]/1e5, 13 vs[4]/1e5, 13 vs[6]/1e5, 14 vs[2]/1e5, 15 rho/1.67e-24, 16 B0*1e5, 17 cnm, 18 np.log10(temCME), 19 yaw, 20 yawv/dtor*3600, 21 (0.5 * yawAcc *3600**2) / dtor, 22  HSSreg]
    outsCME[0].append(CMEarr[0]) 
    outsCME[1].append(CMEarr[1])
    outsCME[2].append(CMEarr[9])
    outsCME[3].append(CMEarr[2])
    outsCME[4].append(CMEarr[3])
    outsCME[5].append(CMEarr[4] / CMEarr[7]) # CMElens[5] / CMElens[6]
    outsCME[6].append(CMEarr[5] / CMEarr[6]) # CMElens[3] / CMElens[4]
    outsCME[7].append(CMEarr[5] / CMEarr[7]) # CMElens[3] / CMElens[6]
    outsCME[8].append(CMEarr[17])
    outsCME[9].append(CMEarr[18])
    outsCME[10].append(CMEarr[16])
    outsCME[11].append(CMEarr[19])
    outsCME[12].append(CMEarr[23])
    outsCME[13].append(CMEarr[20])
    outsCME[14].append(CMEarr[22])
    #CMEarr[10], CMEarr[11], CMEarr[12], CMEarr[13], CMEarr[14], CMEarr[15]]
    outsCME[15].append(CMEarr[10])
    outsCME[16].append(CMEarr[11])
    outsCME[17].append(CMEarr[12])
    outsCME[18].append(CMEarr[13])
    outsCME[19].append(CMEarr[14])
    outsCME[20].append(CMEarr[15])
    
    
    if len(sheatharr) != 0:
        #PUPstuff = [0 r, 1 vShock, 2 Ma, 3 sheath_wid, 4 sheath_dur, 5 sheath_mass/1e15, 6 sheath_dens, 7 np.log10(Tratio*SWfront[4]), 8 Tratio, 9 Bsh, 10 thetaBsh, 11 vtSh, 12 inint]
        
        # 0 vShock, 1 r, 2 Ma, 3 wid, 4 dur, 5 mass, 6 dens 7 temp 8 theta 9 B 10 vt 11 init
        outsSheath[0].append(sheatharr[1])
        outsSheath[1].append(sheatharr[0])
        outsSheath[2].append(sheatharr[2])
        outsSheath[3].append(sheatharr[3])
        outsSheath[4].append(sheatharr[4])
        outsSheath[5].append(sheatharr[5])
        outsSheath[6].append(sheatharr[6])
        outsSheath[7].append(sheatharr[7])
        outsSheath[8].append(sheatharr[10])
        outsSheath[9].append(sheatharr[9])
        outsSheath[10].append(sheatharr[11])
        outsSheath[11].append(sheatharr[12])
        
    return outsCME, outsSheath
    
def add2outsFIDO(outsFIDO, FIDOarr):
    if len(FIDOarr) != 0:
        outsFIDO[0].append(FIDOarr[0])
        outsFIDO[1].append(FIDOarr[1])
        outsFIDO[2].append(FIDOarr[2])
        outsFIDO[3].append(FIDOarr[3])
        outsFIDO[4].append(FIDOarr[4])
        outsFIDO[5].append(FIDOarr[5])
        outsFIDO[6].append(FIDOarr[6])
        outsFIDO[7].append(FIDOarr[7])

    return outsFIDO  

def whereAmI(Epos, CMEpos, CMElens, deltax, deltap, yaw=0):
    # [Er, Elat, Elon], [CMElat, CMElon, CMEtilt], CMElens
    # Get Earth/sat pos in CME coordinate frame
    Epos1 = SPH2CART(Epos)
    temp = rotz(Epos1, -CMEpos[1])
    temp2 = roty(temp, CMEpos[0])
    temp3 = rotx(temp2, 90.-CMEpos[2])
    temp3[0] -= CMElens[0]
    Epos2 = roty(temp3, yaw)
    Epos2[0] += CMElens[0]
    
    # Get the CME axis
    psis = np.linspace(-math.pi/2, math.pi/2, 1001)
    sns = np.sign(psis)
    xFR = CMElens[2] + deltax * CMElens[6] * np.cos(psis)
    zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(psis)) + np.sqrt(1 - np.cos(np.abs(psis))))   
    
    # Determine the closest point on the axis and that distance
    dists2 = ((Epos2[0] - xFR)**2 + Epos2[1]**2 + (Epos2[2] - zFR)**2) / (CMElens[4])**2
    thismin = np.min(dists2)
    thisPsi = psis[np.where(dists2 == np.min(dists2))][0]
    sn = np.sign(thisPsi)
    if sn == 0: sn = 1
    thisPsi = np.abs(thisPsi)
    thisAx = [CMElens[2] + deltax * CMElens[6] * np.cos(thisPsi), 0.,  0.5 * sn * CMElens[6] * (np.sin(thisPsi) + np.sqrt(1 - np.cos(thisPsi)))]
    vp = np.array(Epos2) - np.array(thisAx)
    # Get the normal vector
    if thisPsi == 0: thisPsi = 0.00001
    Nvec = [0.5 * sn*(np.cos(thisPsi) + 0.5*np.sin(thisPsi)/np.sqrt(1-np.cos(thisPsi))), 0., deltax * np.sin(thisPsi)] 
    Nmag = np.sqrt(Nvec[0]**2+Nvec[2]**2)
    norm = np.array(Nvec) / Nmag
    if norm[0] < 0:
        norm = - norm
    thisPsi *= sn

    # Find the sat position within the CS
    vpmag = np.sqrt(np.sum(vp**2))
    # poloidal angle, not the same as parametric t
    arg = np.dot(vp, norm) / vpmag
    if np.abs(arg) > 1:
        arg = 1 * np.sign(arg)
    CSpol = np.arccos(arg) 
    if (vp[1] < 0) and (vp[0] > 0):
        CSpol = -np.abs(CSpol)
    elif (vp[1] < 0) and (vp[0] < 0):
            CSpol = -np.abs(CSpol)
    elif (vp[1] > 0) and (vp[0] > 0):
            CSpol = np.abs(CSpol)
    
    CSxy = np.array([vpmag*np.cos(CSpol), vpmag*np.sin(CSpol)])
    parat = np.arctan2(np.sin(CSpol)*deltap, np.cos(CSpol))
    
    # Get the max R for that parametric t
    maxrFR = np.sqrt(deltap**2 * np.cos(parat)**2 + np.sin(parat)**2) * CMElens[4]
    return vpmag, maxrFR, thisPsi, parat
 
def getBvector(CMElens, minb, mythetaT, thetaP, deltaAx, deltaCS):
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
    pmag = np.sqrt(np.sum(pol**2))
    pol = pol/pmag
    return tan, pol
    
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
    vCS = [xCS * vExps[3]/delCS, yCS*vExps[4], 0] # accounts for dist from cent
    vCSmag = np.sqrt(vCS[0]**2 +vCS[1]**2)
    nCS = np.array([vCS[0]/vCSmag, vCS[1]/vCSmag, 0])
    rotAng = np.arccos(np.dot(nAx, nCS))*np.sign(thetaT)
    if np.abs(rotAng) > np.pi/2:
        rotAng = np.arccos(np.dot(nAx, -nCS))*np.sign(thetaT)
    #nCSatAx = np.array([nCS[0] * np.cos(thetaT), nCS[1], nCS[0] * np.sin(thetaT)])
    #nCSatAx = np.array([nCS[0] * np.cos(np.abs(rotAng)), nCS[1], nCS[0] * np.sin(rotAng)])
    #vCSVec = vCS * nCSatAx
    vCSVec = np.array([vCS[0] * np.cos(np.abs(rotAng)), vCS[1], vCS[0] * np.sin(rotAng)])
    vCMEframe = np.array([vExps[2], 0., 0.]) + vAxVec + vCSVec
    #print ('           ', vCS/1e5, thetaP*180/3.14, nCS, xCS, yCS)
    return vCMEframe, vCSVec

def getFIDO(axDist, maxDistFR, B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, Elat, Elon, CMElat, CMElon, CMEtilt, vs, yaw, comp=1.):
    # Get Btor/Bpol at s/c location
    Btor = B0 * deltap * (tau - (axDist / maxDistFR)**2)
    ecH  = np.sqrt(deltap**2 * np.sin(thisParat)**2 + np.cos(thisParat)**2)
    Bpol = - 2 * deltap * B0 * CMEH * ecH / (deltap**2 + 1) / cnm * (axDist / maxDistFR)
    # Get unit vectors    
    axR = CMElens[0] - CMElens[3]
    tdir, pdir = getBvector(CMElens, axDist, thisPsi, thisParat, deltax, deltap)
    # Convert to s/c field
    Bt_vec = Btor * tdir #* comp 
    Bp_vec = Bpol * pdir
    Btot = Bt_vec + Bp_vec
    # Convert to spacecraft coord system
    temp0 = roty(Btot, -yaw)
    temp = rotx(temp0, -(90.-CMEtilt))
    temp2 = roty(temp, CMElat - Elat) 
    Bvec = rotz(temp2, CMElon - Elon)
    
    # get velocity vector
    vCMEframe, vExpCME = getvCMEframe(axDist / maxDistFR, thisPsi, thisParat, deltax, deltap, vs)
    #print ('          ',thisParat*180/3.14, vCMEframe, vExpCME)
    # rotate to s/c frame
    temp0 = roty(vCMEframe, -yaw)
    temp = rotx(temp0, -(90.-CMEtilt))
    temp2 = roty(temp, CMElat - Elat) 
    vInSitu = rotz(temp2, CMElon - Elon )

    # rotate vexp
    #temp = rotx(vExpCME, -(90.-CMEtilt))
    #temp2 = roty(temp, CMElat - Elat) 
    #vexp = rotz(temp2, CMElon - Elon)
    
    return Bvec, vInSitu, vExpCME

# -------------- main function ------------------
def getAT(invec, Epos, SWparams, SWidx=None, silent=False, fscales=None, pan=False, selfsim=False, csTens=True, thermOff=False, csOff=False, axisOff=False, dragOff=False, name='nosave', satfs=None, flagScales=False, tEmpHSS=False, tDepSW=False, doPUP=True, saveForces=False, MEOWHiSS=False, fullContact=False, aFIDOinside=False, CMEH=1, inorout=1, simYaw=False, SWR=None, sheathParams=[None]):
    # testing things
    #satfsIn = [satfs[0], satfs[0], ['sat1', 'sat2']]
    #satPosIn = [[Epos[0], Epos[1], Epos[2] *7e10, Epos[3]], [Epos[0], Epos[1]-10, (Epos[2]) *7e10, Epos[3]]]
    #satPosIn = [[Epos[0], Epos[1], Epos[2] *7e10, Epos[3]]]
    satfsIn = satfs
    satPosIn = Epos

    CMElat     = invec[0]
    CMElon     = invec[1]
    CMEtilt    = invec[2]    
    vFront     = invec[3] * 1e5
    Mass       = invec[4] * 1e15
    AW         = invec[5] * pi / 180.
    AWp        = invec[6] * pi / 180.
    deltax     = invec[7]
    deltap     = invec[8]
    rFront     = invec[9] * rsun
    Bscale     = np.abs(invec[10])
    B0sign     = np.sign(Bscale)
    Cd         = invec[11]
    tau        = invec[12] # 1 to mimic Lundquist
    cnm        = invec[13] # 1.927 for Lundquist
    temScale   = invec[14] 
    fT         = invec[15] - 1 # 0 for isothermal, 2./3 for adiabatic
    yaw        = invec[16]
    
    # polarity of FR for aFIDOinside set by CMEH, defaults to 1 for positive/right hand (-1 for left)
    # amb SW in or out set by inorout, defaults to 1 for out (-1 for in)
    
    # FIDO returns in satellite coords, not GSE
    
    # if want FIDO profile force it to do full contact
    if aFIDOinside:
        fullContact = True
        
    # Solar rotation rate
    solRotRate = 360. / (24.47 * 24 )
    
    # sheath properties
    if sheathParams[0]:
        sheath_mass = sheathParams[0]*1e15
        sheath_wid  = sheathParams[1]
        sheath_dens = sheathParams[2]
    else:
        sheath_mass = 0.
        sheath_wid  = 0.
        sheath_dens = 0.
        
    hitSheath = False
    inint1 = 0
    global Tratio, rratio
    Tratio = 1
    rratio = 1
           
    
    if not flagScales:
        CMEB = invec[10] / 1e5
        CMET = invec[14]
    
    
    # Set up satellite(s)
    if satfsIn == None:
        doPath = False
        nSats = len(satPosIn)-1
        satNames = satPosIn[-1]
        satPos = {}
        for i in range(nSats):
            satPos[satNames[i]] = satPosIn[i]            
    else:
        nSats = len(satfsIn)
        satPos = {}
        satfs = {}
        if nSats == 1:
            try:
                satNames = satPosIn[1]
                satPos[satPosIn[1][0]] = satPosIn[0]
                satfs[satPosIn[1][0]] = satfsIn[0]
            except:
                satNames = ['sat1']
                satPos['sat1'] = satPosIn[0]
                satfs['sat1'] = satfsIn[0]            
        elif nSats == 2:
            sys.exit('Need to provide list of sat names if using multiple sateliites')
        elif nSats == 0:
            sys.exit('satfs is empty, please fix')
        else:
            satNames = satfsIn[-1]
            nSats -= 1  
            for i in range(nSats):
                satPos[satNames[i]] = satPosIn[i]
                satfs[satNames[i]] = satfsIn[i] 
        doPath = True
    sats2check = list(satNames)

    
    # Get ambient background SW - still need ambient event if doing
    # HSS bc scaled off of it
    # Check if passed 1 AU values -> scaling of empirical model
    # this will used the first sat in satnames so put 1 AU first (or fix this in some clever way later)
    if not isinstance(SWparams, str):    
        if SWR == None:
            SWdist = satPos[satNames[0]][2]/1.5e13
        else:
            SWdist = SWR / 215.
        SWparams = [SWparams[0], SWparams[1], SWparams[2], SWparams[3], SWdist]
    # If given string, func will convert text file to profile
    # Otherwise uses the array
    SWfs = makeSWfuncs(SWparams, time=SWidx)

    # Determine if using MEOW-HiSS or not
    #MEOWHiSS = [800, 1.2]
    doMH = False
    # check if list or array
    if hasattr(MEOWHiSS, '__len__'):
        if len(MEOWHiSS) == 2:
            # If doing MH get the initial values
            MHt0, MHxs0, MHvs, MHa1 = emp.getxmodel(MEOWHiSS[0], MEOWHiSS[1])
            # Get the HSS funcs to scale the ambient background
            vfuncs = emp.getvmodel(MEOWHiSS[0])
            nfuncs = emp.getnmodel(MEOWHiSS[0])
            Brfuncs = emp.getBrmodel(MEOWHiSS[0])
            Blonfuncs = emp.getBlonmodel(MEOWHiSS[0])
            Tfuncs = emp.getTmodel(MEOWHiSS[0])
            vlonfuncs = emp.getvlonmodel(MEOWHiSS[0])
            MHfuncs = [nfuncs, vfuncs, Brfuncs, Blonfuncs, Tfuncs, vlonfuncs]
            doMH = [MHt0+0, MHt0, MHxs0, MHvs, MHa1, MHfuncs]
        else:
            sys.exit('Cannot initiate HSS. Set MEOWHiSS to [CH_area, init_dist]')
    
    writeFile = False
    if name != 'nosave':
        writeFile = True
        fname = 'MH_PARADE_'+name+'.dat'
        f1 = open(fname, 'w')
        if doPUP:
            fname2 = 'MH_PUP_'+name+'.dat'
            f2 = open(fname2, 'w')
        if saveForces:
            fname3 = 'MH_forces_'+name+'.dat'
            f3 = open(fname3, 'w')
            if simYaw:
                fnameYaw = 'MH_torque_'+name+'.dat'
                fYaw = open(fnameYaw, 'w')
            
        if aFIDOinside:
            # Might have multiple satellites so need multiple save files
            fname4s = {}
            if nSats > 1:
                for aSatID in satNames:
                    fname4 = 'MH_FIDO_'+name+'_'+aSatID+'.dat'
                    f4s[aSatID] = open(fname4, 'w')     
            else:
                fname4 = 'MH_FIDO_'+name+'.dat'
                f4s['sat1'] = open(fname4, 'w')                

    t = 0.
    dt = 60. # in seconds
    deltaPrintR = 7e10
    CMElens = np.zeros(8)
    #CMElens = [rFront, rEdge, d, br, bp, a, c, rXCent]
    CMElens[0] = rFront
    alpha, ndotz, rho = initCMEparams(deltax, deltap, AW, AWp, CMElens, Mass)
            
    thisR = CMElens[0] - CMElens[3]
    sideR = np.sqrt(CMElens[2]**2 + (CMElens[1]-CMElens[3])**2)
    avgAxR = 0.5*(CMElens[0] - CMElens[3] + sideR)   
    SWavgAx = getSWvals(avgAxR, SWfs, doMH=doMH)
    Btot2 = SWavgAx[2]**2 + SWavgAx[3]**2
    
    if doMH:
        SWfront, HSSreg  = getSWvals(CMElens[0], SWfs, doMH=doMH, returnReg = doMH)
    else:
        SWfront  = getSWvals(CMElens[0], SWfs, doMH=doMH, returnReg = doMH)
        HSSreg = 0
    # fix for rEdge confusion
    SWedge   = getSWvals(sideR, SWfs, doMH=doMH)
    SWfrontB = getSWvals(CMElens[0]-2*CMElens[3], SWfs, doMH=doMH)
    SWedgeB  = getSWvals(sideR-2*CMElens[3], SWfs, doMH=doMH)
       
    # Set up factors for scaling B through conservation
    # this was scaled of Bsw0 insteand of sqrt(Btot2) before...
    if flagScales:
        B0 = Bscale * np.sqrt(Btot2) / deltap / tau
    else:
        B0 = CMEB/ deltap / tau
    B0scaler = B0 * deltap**2 * CMElens[4]**2 
    initlen = lenFun(CMElens[5]/CMElens[6]) * CMElens[6]
    cnmscaler = cnm / initlen * CMElens[4] * (deltap**2+1)
    initcs = CMElens[3] * CMElens[4]

    # get CME temperature based on expected SW temp at center of nose CS
    temSW = getSWvals(CMElens[0]-CMElens[3], SWfs, doMH=doMH)[4]
    if flagScales:
        temCME = temScale * temSW
    else:
        temCME = CMET
    temscaler = np.power(CMElens[3] * CMElens[4] * initlen , fT) * temCME
    
    # use to pull out inputs for uniform B/T across multiple cases instead
    # of using B/Tscales
    #print (B0*deltap*tau, np.log10(temCME))

    # Set up the initial velocities
    if fscales == None:
        if pan: 
            fscales = [1., 1.]
        elif selfsim:
            fscales = [0., 0.]
        else:
            fscales = [0.5, 0.5]
    vs = IVD(vFront, AW, AWp, deltax, deltap, CMElens, alpha, ndotz, fscales)    
    
    # angular velocity
    yawv = 0. # assume starts stationary, v = angular w (rad/s)
    yawMom = 0. # will 
    yawAcc = 0.
        
    printR = rFront
    runSim = True
    prevmin = {}
    reachedCME = {}
    hitSheath = {}
    inint     = {}
    outSum   = {}
    vsArr    = {}
    angArr   = {}
    for aSatID in satNames:
        prevmin[aSatID] = 9999.
        reachedCME[aSatID] = False
        hitSheath[aSatID] = False
        inint[aSatID] =  0
        outSum[aSatID] = [-9999]
        vsArr[aSatID] = []
        angArr[aSatID] = []

    outsCME = [[] for i in range(21)]
    outsSheath = [[] for i in range(12)]
    outsFIDO = {}
    FIDOstuff = {}
    for aSatID in satNames:
        outsFIDO[aSatID] = [[] for i in range(8)]
        FIDOstuff[aSatID] = []

    while runSim:
    #while CMElens[0] <= 0.9*Er:
        # -----------------------------------------------------------------------------
        # Stuff that's done whether sat in CME or not
        # -----------------------------------------------------------------------------
        
        # Accels order = [Front, Edge, Center, CSr, CSp, Axr, Axp,]
        totAccels = np.zeros(7)
        magAccels = np.zeros(7)
        thermAccels = np.zeros(7) 
         
        #--------------------------- Calculate forces ---------------------------------
        # Drag Force
        if not dragOff:
            dragAccels = getDrag(CMElens, vs, Mass+sheath_mass, AW, AWp, Cd, SWfront, SWfrontB, SWedge, SWedgeB, ndotz)
            # Update accels to include drag forces
            totAccels += dragAccels
        
        torque = 0.
        if simYaw:
            if saveForces:
                torque, torOuts = calcYawTorque(CMElens, [CMElat, CMElon, CMEtilt], AW, vs, deltax, deltap, yaw, solRotRate, SWfs, Cd, doMH, saveForces=True)
            else:
                torque = calcYawTorque(CMElens, [CMElat, CMElon, CMEtilt], AW, vs, deltax, deltap, yaw, solRotRate, SWfs, Cd, doMH)
        
        # Internal flux rope forces
        if not axisOff:
            aTotNose, aTotEdge = getAxisF(deltax, deltap, CMElens[4], CMElens[6], B0, cnm, tau, rho)
            if (aTotNose != 9999) & (aTotEdge != 9999):
                magAccels += [aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), 0, 0, aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz]
            else:
                return makeFailArr(8888)   
        if not csOff:
            aBr = getCSF(deltax, deltap, CMElens[4], CMElens[6], B0, cnm, tau, rho, Btot2, csTens)
            magAccels += [aBr, aBr * ndotz, 0, aBr, aBr/deltap, 0., 0.] 
        totAccels += magAccels        

        # Thermal Expansion
        if not thermOff:
            rside = np.sqrt((CMElens[0] - CMElens[3])**2 + CMElens[4]**2) # side of CS at front
            SWside = getSWvals(rside, SWfs, doMH=doMH)
            aPTr, aPTp = getThermF(CMElens, temCME, rho/1.67e-24, SWfront, SWfrontB, SWside)/rho
            thermAccels += [aPTr, aPTr * ndotz, 0, aPTr, aPTp, 0., 0.] 
        totAccels += thermAccels
                       
        
            
        # ------------------ Run Sheath model ---------------------------------
        if doPUP:
            gam = 5/3.
            BSWr, BSWlon = SWfront[2]*1e5, SWfront[3]*1e5 
            Btot2F = (BSWr**2 + BSWlon**2)/1e10
            cs = np.sqrt(2*(5/3.) * 1.38e-16 * SWfront[4] / 1.67e-24) #SW gamma not CME gamma
            vA = np.sqrt(Btot2F / 4 / 3.14159 / SWfront[0])
            thetaB = np.arctan(np.abs(BSWlon / BSWr))
            if vs[0]/1e5 > SWfront[1]/1e5:
                r, vShock, Pratio = getObSheath(vs[0]/1e5, SWfront[1]/1e5, thetaB, cs=cs/1e5, vA=vA/1e5)    
            else:
                r, vShock, Pratio = 9999, 9999, 9999
                
            if r == 9999:
                r, vShock, Pratio, u1, Btscale, Ma, Tratio = 1, 0, 1, SWfront[1], 1, 0, 1
                Blonsh, Bsh, thetaBsh, vtSh = 0, 0, 0, 0
            else: 
                u1 = vShock*1e5 - SWfront[1]
                Ma = u1 / vA
                Btscale = r * (Ma**2 - np.cos(thetaB)**2) / (Ma**2 - r * np.cos(thetaB)**2)
                sheath_wid += (vShock-vs[0]/1e5)*dt / 7e5                
                Tratio = Pratio /r
                # tangent v in non dHT frame
                vdHT = u1* np.tan(thetaB) / 1e5
                vtSh = u1*(np.tan(thetaB) * (Ma**2 - np.cos(thetaB)**2) / (Ma**2 - r * np.cos(thetaB)**2)) / 1e5 - vdHT
                # Magnetic Field 
            Blonsh = Btscale*BSWlon
            Bsh = np.sqrt(Blonsh**2 + BSWr**2)
            
            thetaBsh = np.arctan(Blonsh/BSWr)*180/3.14159
            
            
                        
        # ---------------- Update everything ---------------------------------------                                                        
        # Update CME shape + position
        CMElens[:7] += vs*dt + 0.5*totAccels*dt**2    
        CMElens[7] = CMElens[2] + CMElens[5]
        theFront = CMElens[0]
        
        # Update sheath mass 
        if doPUP:
            SHarea = 4*CMElens[6]*CMElens[4] 
            SWup = getSWvals(CMElens[0]+sheath_wid*7e10, SWfs, doMH=doMH)
            rhoSWup = SWup[0]
            if r != 1:
                new_mass = rhoSWup * (vShock*1e5 - SWfront[1]) * dt * SHarea
                sheath_mass += new_mass
                rratio = r
            if sheath_wid != 0:
                sheath_dens = sheath_mass/(sheath_wid*7e10*SHarea)/1.67e-24
                    
        # Update velocities
        vs += totAccels * dt
        # Check if moving backwards, can happen with bad inputs (large CME->large Bax force)
        if vs[0] < 0:
            return makeFailArr(8888)
            
        if simYaw:
            bb = CMElens[4]
            RR = CMElens[6]
            ee = np.sqrt(1 - deltax**2)
            Itot = 2 * math.pi * rho * bb**2 * RR * (ellipk(ee**2) * ((deltax*RR+bb)**2 + 0.75*bb**2) + RR**2 *ellipe(ee**2)/(1-ee**2) - 2*(deltax*RR+bb)*RR/(ee*np.sqrt(1-ee**2)) * np.arctan(np.sqrt(ee**2/(1-ee**2))) )
            
            # conservation of ang mom, yawv will slow down as I grows
            yawv = yawMom / Itot 
            yawAcc = torque/Itot
            yaw += (yawv * dt + 0.5 * yawAcc * dt**2) / dtor
            yawv += yawAcc*dt
            yawMom = Itot * yawv
               
        if doPUP:
            sheath_dur = sheath_wid*7e10/vs[0]/3600
        
        # Calc shape params and new AWs
        deltap, deltax, deltaCSAx, alpha, ndotz, AW, AWp, rho = updateCME(CMElens, Mass)
        if alpha < 0: 
            return makeFailArr(8888)   
        
        # Update time        
        t += dt
        
        # Update HSS position
        if hasattr(MEOWHiSS, '__len__'):
            doMH[0] = MHt0 + t/3600

        # Update Earth/satellite position
        for satID in sats2check:
            if not doPath:    
                satPos[satID][1] += satPos[satID][3] * dt
            else:
                satPos[satID][0] = satfs[satID][0](t)
                satPos[satID][1] = satfs[satID][1](t)
                satPos[satID][2]   = satfs[satID][2](t)*7e10

        # Update flux rope field parameters
        lenNow = lenFun(CMElens[5]/CMElens[6])*CMElens[6]
        B0 = B0scaler / deltap**2 / CMElens[4]**2 
        cnm = cnmscaler * lenNow  / CMElens[4] / (deltap**2+1)
        Btor = deltap * tau * B0
        Bpol = 2*deltap*B0/cnm/(deltap**2+1)
        
        # Update temperature
        temCME = temscaler / np.power(CMElens[3] * CMElens[4] * lenNow, fT)
        nowcs = CMElens[3] * CMElens[4]
                
        # Update solar wind properties
        thisR = CMElens[0] - CMElens[3]
        sideR = np.sqrt(CMElens[2]**2 + (CMElens[1]-CMElens[3])**2)
        avgAxR = 0.5*(CMElens[0] - CMElens[3] + sideR)  
        # values at avg axis distance 
        if doMH:
            SWavgAx, reg2 = getSWvals(avgAxR, SWfs, doMH=doMH, returnReg=doMH)
        else:
            SWavgAx = getSWvals(avgAxR, SWfs, doMH=doMH, returnReg=doMH)
            reg2 = 0
        Btot2 = SWavgAx[2]**2 + SWavgAx[3]**2
        # values at front
        if doPUP:
            theFront += sheath_dur*7e10
        if doMH:            
            SWfront, HSSreg  = getSWvals(theFront, SWfs, doMH=doMH, returnReg=doMH)
        else:
            SWfront = getSWvals(theFront, SWfs, doMH=doMH, returnReg = doMH)
            HSSreg = 0
        # other (simpler) locations
        SWedge   = getSWvals(sideR, SWfs, doMH=doMH)
        SWfrontB = getSWvals(CMElens[0]-2*CMElens[3], SWfs, doMH=doMH)
        SWedgeB  = getSWvals(sideR-2*CMElens[3], SWfs, doMH=doMH)
        

        # ------------------------------------------------------------------
        # ------Set up state vectors at the end of a time step--------------
        # ------------------------------------------------------------------
        # CME array
        if HSSreg > 0:
            HSSreg = 6 - HSSreg
        CMEstuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap,  rho/1.67e-24, np.log10(temCME), B0*1e5, yaw, yawv/dtor*3600, HSSreg]
        fullCMEstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME), yaw, yawv/dtor*3600, (0.5 * yawAcc *3600**2) / dtor, HSSreg]
        
        # PUP/sheath array
        if inint1 == 0:
            for satID in sats2check:
                if hitSheath[satID]:
                    inint1 = 1
        PUPstuff = []
        if doPUP:
            PUPstuff = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Tratio, Bsh, thetaBsh, vtSh, inint1]
            PUPscreen = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Bsh, inint1]
        
        # Forces
        if saveForces:
            forcestuff = [t/3600./24., CMElens[0]/rsun, aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz,    aBr, aBr * ndotz, aBr, aBr/deltap,   aPTr, aPTr * ndotz, aPTr, aPTp,     dragAccels[0], dragAccels[1], dragAccels[2], dragAccels[3], dragAccels[4], dragAccels[5], dragAccels[6]]
            
            # Torques
            if simYaw:
                torstuff = [t/3600./24., CMElens[0]/rsun, torOuts[0], torOuts[1], torOuts[2], torOuts[3], torOuts[4], torOuts[5]]
                

 
        # ------------------------------------------------------------------
        # ----------------- Save and print everything ----------------------
        # ------------------------------------------------------------------
        # Printing/saving at lower time resolution so check if it is go time
        printNow = False
        if (CMElens[0]>printR):
            printNow = True
            printR += deltaPrintR
            
            # print to screen    
            if not silent:
                print2screen(CMEstuff)       
                if doPUP:
                    print2screen(PUPscreen, prefix='      ') 
                 
                                        
            # print to file (if doing)
            if writeFile: 
                print2file(fullCMEstuff, f1, '{:8.5f}')
                if doPUP:
                    print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')                
                if saveForces:  
                    print2file(forcestuff, f3, '{:11.3f}')
                    if simYaw:
                        print2file(torstuff, fYaw, '{:12.5f}')
            
            # save SW values to FIDO file (if doing)
            for satID in sats2check:
                if aFIDOinside and (CMElens[0] > satPos[satID][2]*0.5) and not hitSheath[satID] and not reachedCME[satID]:
                    # get SW values
                    if doMH:
                        # need to adjust satR for lon diff from CME nose
                        dlonMH = CMElon - satPos[satID][1]
                        dtMH = solRotRate * dlonMH
                        SWatsat, HSSreg   = getSWvals(satPos[satID][2], SWfs, doMH=doMH, returnReg=doMH, deltatime=dtMH)
                        FIDOstuff[satID] = [t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg]
                        if writeFile: print2file(FIDOstuff[satID], f4s[satID], '{:8.5f}')
                    else:
                        SWatsat   = getSWvals(satPos[satID][2], SWfs)
                        FIDOstuff[satID] = [t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100]
                        if writeFile: print2file(FIDOstuff[satID], f4s[satID], '{:8.5f}')
        # save CME/sheath data
        if printNow:
            outsCME, outsSheath = add2outs(outsCME, outsSheath, fullCMEstuff, PUPstuff)
                    
        # -----------------------------------------------------------------------------
        # Check if in sheath or CME
        # -----------------------------------------------------------------------------
        for satID in sats2check: 
            #Er = 50 * 7e10 # quick switch for debugging and not runnning full thing
            # cant' just swap Er in inputs if scaling SW params off of it...
        
            # Start checking when close to sat distances, don't waste code time before close
            check_dist = 0.5*satPos[satID][2]
            # front for checking purposes -> could be sheath or CME
            theFront = CMElens[0]
            if doPUP: theFront = CMElens[0] + sheath_wid * 7e10
        
            if theFront >= check_dist:
                # Get Earth/sat pos in CME coordinate frame            
                axDist, maxDistFR, thisPsi, thisParat = whereAmI([satPos[satID][2], satPos[satID][0], satPos[satID][1]], [CMElat, CMElon, CMEtilt], CMElens, deltax, deltap, yaw=yaw)

                if doPUP: maxrSH = maxDistFR + sheath_wid * 7e10
                # check if in sheath and save if so
                if doPUP:
                    if (axDist < maxrSH) and (maxDistFR != maxrSH) and (satPos[satID][2] > CMElens[0]):
                        if not hitSheath[satID]:
                            hitSheath[satID] = True
                            inint[satID] = 1
                            outSum[satID] = [t/3600., vs[0]/1e5, vs[3]/1e5, CMElens[0]/7e10, satPos[satID][1], -9999]
                            # Add this specific time step if not already included
                            if not printNow:
                                if not silent:
                                    print2screen(CMEstuff)     
                                    if doPUP:
                                        # need to update init
                                        PUPstuff[12] = inint[satID]
                                        print2screen(PUPscreen, prefix='      ')                                 
                                if writeFile:
                                    print2file(fullCMEstuff, f1, '{:8.5f}')
                                    print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')
                                printNow = True
                        # print sheath info to FIDO output file
                        if (axDist > maxDistFR) and (satPos[satID][2] > CMElens[0]) and printNow and aFIDOinside:
                            tdir, pdir = getBvector(CMElens, axDist, thisPsi, thisParat, deltax, deltap)
                            ndir = np.cross(tdir, pdir)
                            # attempt to get signs correct in sheath based on inorout, flip unit vecs to most aligned choice
                            temp0 = roty(ndir, -yaw)
                            temp = rotx(temp0, -(90.-CMEtilt))
                            temp2 = roty(temp, CMElat - satPos[satID][0]) 
                            ndirSAT = np.array(rotz(temp2, CMElon - satPos[satID][1]))
                        
                            temp0 = roty(pdir, -yaw)
                            temp = rotx(temp0, -(90.-CMEtilt))
                            temp2 = roty(temp, CMElat - satPos[satID][0]) 
                            pdirSAT = np.array(rotz(temp2, CMElon - satPos[satID][1]))
                        
                            # figure out what is most aligned
                            SWatsat   = getSWvals(satPos[satID][2], SWfs)
                            SWBvec = np.array([inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0])
                            # Old version
                            '''if np.dot(ndirSAT, np.array([inorout,0,0])) < 0:
                                ndirSAT = -ndirSAT
                            if np.dot(pdirSAT, np.array([0, -inorout,0])) < 0:
                                pdirSAT = -pdirSAT'''
                            if np.dot(ndirSAT, SWBvec) < 0:
                                ndirSAT = -ndirSAT
                            if np.dot(pdirSAT, SWBvec) < 0:
                                pdirSAT = -pdirSAT
                        
                            BinSitu = Bsh * np.cos(thetaBsh*math.pi/180.) * ndirSAT + Bsh * np.sin(thetaBsh*math.pi/180.) * pdirSAT 
                            # calculate v that is appropriate for that FR location
                            vCMEframe, vExpCME = getvCMEframe(1, thisPsi, thisParat, deltax, deltap, vs)
                            #print ('          ',thisParat*180/3.14, vCMEframe, vExpCME)
                            # rotate to s/c frame
                            temp0 = roty(vCMEframe, -yaw)
                            temp = rotx(temp0, -(90.-CMEtilt))
                            temp2 = roty(temp, CMElat - satPos[satID][0]) 
                            vInSitu = rotz(temp2, CMElon - satPos[satID][1])
                            
                            FIDOstuff[satID] = [t/3600., BinSitu[0], BinSitu[1], BinSitu[2], vInSitu[0]/1e5, sheath_dens, np.log10(Tratio*SWfront[4]),0]
                            if writeFile: print2file(FIDOstuff[satID],  f4s[satID], '{:8.5f}')
                        
                            if not silent:
                                print ('      ', satID, '{:8.5f}'.format(axDist/maxDistFR), '{:8.5f}'.format(thisPsi*180/3.14159), '{:8.5f}'.format(thisParat*180/3.14159))
                
                # check if in FR
                thismin = axDist/maxDistFR
                if axDist < maxDistFR:
                    # start saving at higher res once in CME
                    if not reachedCME[satID]:
                         if len(sats2check)==nSats:
                             deltaPrintR = deltaPrintR/4.
                         # save things for the summary print to screen
                         # want values at first contact even if doing full profile
                         # will replace duration if have more accurate version
                         TT = t/3600.
                         thisDur = 4 * CMElens[3] * (2*(vs[0]-vs[3])+3*vs[3])/(2*(vs[0]-vs[3])+vs[3])**2 / 3600.
                         outSum[satID] = [TT, vs[0]/1e5, vs[3]/1e5, CMElens[0]/7e10, satPos[satID][1], thisDur]
                         vsArr[satID] = np.copy(vs)
                         angArr[satID] = [thisPsi, thisParat]
                         reachedCME[satID] = True
                         
                    # Get the FR properties and print
                    if aFIDOinside and printNow:
                        BSC, vInSitu, printthis = getFIDO(axDist, maxDistFR, B0sign*B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, satPos[satID][0], satPos[satID][1], CMElat, CMElon, CMEtilt, vs, yaw)      
                        #vA = np.sqrt((BSC[0]**2 + BSC[1]**2 + BSC[2]**2) / 4 / 3.14159 / (rho/1.67e-24))*1e5
                        # This adds extra compression if the expansion is compression in the rear
                        '''CMEgam = fT+1
                        cs = np.sqrt(2*(CMEgam) * 1.38e-16 *temCME / 1.67e-24)/1e5
                        mch = -vs[3]*(axDist/maxDistFR)/1e5 / cs
                        comp = 1
                        if mch >1:
                            comp = (CMEgam+1)*mch**2 / (2 + (CMEgam-1)*mch**2)
                            if comp < 1: comp = 1
                            BSC, vInSitu, printthis = getFIDO(axDist, maxDistFR, B0sign*B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, satPos[satID][0], satPos[satID][1], CMElat, CMElon, CMEtilt, vs, comp=comp)''' 
                        # Print to screen so we know where we are in the FR
                        if not silent:
                            print ('      ', satID, '{:8.5f}'.format(axDist/maxDistFR), '{:8.5f}'.format(thisPsi*180/3.14159), '{:8.5f}'.format(thisParat*180/3.14159))
                        # save to file
                        comp = 1 # had used this before but turning off for now but keeping code intact
                        FIDOstuff[satID] = [t/3600., BSC[0]*1e5, BSC[1]*1e5, BSC[2]*1e5, vInSitu[0]/1e5, rho/1.67e-24 * comp, np.log10(temCME), 1]
                        if writeFile: print2file(FIDOstuff[satID], f4s[satID], '{:8.5f}')
                
                    # Add this first timestep if it isn't already included
                    if not printNow and not reachedCME[satID]:
                        outsCME, outsSheath = add2outs(outsCME, outsSheath, fullCMEstuff, PUPstuff)
                        outsFIDO[satID] = add2outsFIDO(outsFIDO[satID], FIDOstuff[satID])                 
                        if writeFile:
                            print2file(fullCMEstuff, f1, '{:8.5f}')
                            if doPUP:
                                print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')   
                            if saveForces:
                                print2file(forcestuff, f3, '{:11.3f}')
                            if aFIDOinside:
                                print2file(FIDOstuff[satID], f4s[satID], '{:8.5f}')
                            
                        if not silent:
                            print2screen(CMEstuff)
                            if doPUP:
                                print2screen(PUPscreen, prefix='      ')                 
                    
                                                        
                    # Exit point for first contact case, otherwise keeps looping
                    # If not doing full contact mode return everything at the time of first FR impact    
                    if not fullContact:
                        # remove sat from list to check
                        sats2check.remove(satID)
                        if len(sats2check)==0:
                            runSim = False
                    
                # Exit for full contact once dist > 1 after first contact    
                elif reachedCME[satID]:
                    sats2check.remove(satID)
                    if len(sats2check)==0:
                        runSim = False
                    thisDur =t/3600 - outSum[satID][0]
                    outSum[satID][5] = thisDur
                    
                    # if doing aFIDOinside add SW padding behind the CME
                    t2 = t
                    for i in range(18+24):
                        t2 += 3600.
                        # Update HSS position
                        if hasattr(MEOWHiSS, '__len__'):
                            doMH[0] = MHt0+t2/3600

                        # Update Earth/satellite position
                        if not doPath:    
                            satPos[satID][1] += satPos[satID][3] * dt
                        else:
                            satPos[satID][0] = satfs[satID][0](t2)
                            satPos[satID][1] = satfs[satID][1](t2)
                            satPos[satID][2] = satfs[satID][2](t2)*7e10
                        # get SW values
                        if doMH:
                            SWatsat, HSSreg   = getSWvals(satPos[satID][2], SWfs, doMH=doMH, returnReg=doMH)
                        else:
                            SWatsat   = getSWvals(satPos[satID][2], SWfs)
                            HSSreg = 0
                    
                        FIDOarr = [t2/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg]
                        outsFIDO[satID] = add2outsFIDO(outsFIDO[satID], FIDOarr) 
                        
                        if writeFile:
                            print2file([t2/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg],  f4s[satID], '{:8.5f}')
                 
                # Continue simulation if distance from axis is decreasing        
                elif (thismin < prevmin[satID]):
                    prevmin[satID] = thismin
                    
                # Exit point if begins moving away from sat
                else: 
                    sats2check.remove(satID)
                    if len(sats2check)==0:
                        runSim = False
                    if not hitSheath[satID]:
                        outsFIDO[satID] = [[8888]*8]
                        if nSats == 1:
                            vsArr[satID] = [8888, 8888, 8888, 8888, 8888, 8888]
                        if len(sats2check)==0:
                            runSim = False
                    elif aFIDOinside:
                        t2 = outsFIDO[satID][0][-1]
                        # if doing aFIDOinside add SW padding behind the CME
                        t2 = t
                        for i in range(18+24):
                            t2 += 3600.
                            # Update HSS position
                            if hasattr(MEOWHiSS, '__len__'):
                                doMH[0] = MHt0+t2/3600

                            # Update Earth/satellite position
                            if not doPath:    
                                satPos[satID][1] += satPos[satID][3] * dt
                            else:
                                satPos[satID][0] = satfs[satID][0](t2)
                                satPos[satID][1] = satfs[satID][1](t2)
                                satPos[satID][2] = satfs[satID][2](t2)*7e10
                            # get SW values
                            if doMH:
                                SWatsat, HSSreg   = getSWvals(satPos[satID][2], SWfs, doMH=doMH, returnReg=doMH)
                            else:
                                SWatsat   = getSWvals(satPos[satID][2], SWfs)
                                HSSreg = 0
                    
                            FIDOarr = [t2/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg]
                            outsFIDO[satID] = add2outsFIDO(outsFIDO[satID], FIDOarr) 
                        
                            if writeFile:
                                print2file([t2/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg],  f4s[satID], '{:8.5f}')
                        
                                 
                # Exit point if center is more than 150 Rs beyond sat pos
                # Don't think this can be hit? hide for now
                '''elif CMElens[0] > satPos[satID][2] + 150 * 7e10:
                    sats2check.remove(satID)
                    if len(sats2check)==0:
                        runSim = False
                    return makeFailArr(9999) '''  
                    

            # Add FIDO stuff to array if still looping. Could be SW, sheath or CME     
            if printNow:
                if satID in sats2check:
                    outsFIDO[satID] = add2outsFIDO(outsFIDO[satID], FIDOstuff[satID])  
    
    # ------------------------------------------------------------------
    # Return things once done with all sats
    # ------------------------------------------------------------------
    if not silent:
        for satID in satNames:
            if outSum[satID] == [-9999]:
                print ('No Contact for '+satID)
            else:  
                print ('Satellite:        ', satID)
                print ('Transit Time:     ', outSum[satID][0])
                print ('Final Velocity:   ', outSum[satID][1], outSum[satID][2])
                print ('CME nose dist:    ', outSum[satID][3])
                print ('Sat. longitude:   ', outSum[satID][4])
                print ('Est. Duration:    ', outSum[satID][5])
            print ('')

    outsCME = np.array(outsCME)
    outsSheath = np.array(outsSheath)
    for i in range(len(outsCME)): outsCME[i] = np.array(outsCME[i])
    for i in range(len(outsSheath)): outsSheath[i] = np.array(outsSheath[i])

    for thissatID in satNames:
        if outSum[thissatID] != [-9999]:
            for i in range(len(outsFIDO[thissatID])): outsFIDO[thissatID][i] = np.array(outsFIDO[thissatID][i])
            outsFIDO[thissatID] = np.array(outsFIDO[thissatID])
    return outsCME, outSum, vsArr, angArr, SWfront, outsSheath, outsFIDO  
        
        

if __name__ == '__main__':
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, Bscale, Cd, tau, cnm, Tscale, gammaT]   
    # Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth    
    # SWparams = [nSW, vSW, BSW, TSW] at Eradius
    
    invecs = [0, 0, 0, 1200.0, 10.0, 46, 18, 0.6, 0.6, 21.5, 3200, 1.0, 1., 1.927, 7.5e5, 1.33, 0] # fast, 
    invecsAvg = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 3.9e5, 1.33, 0] # avg
    invecsAvgLT = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 2e5, 1.33, 0] # avg lower T
    invecsSlow = [0, 0, 0, 350, 2., 25, 7, 0.7, 0.7, 21.5, 500, 1.0, 1., 1.927, 2e5, 1.33, 0] # slow

    satParams = [[0.0, 00.0, 215.0*7e10, 0.0], ['sat1']] #[1.0, 00.0, 215.0, 0.0]
    SWinparams = [5.0, 440, 6.9, 6.2e4]
    MHDparams = [4.5, 386, 3.1, 4.52e4]
    
    fname0 = 'SWnoHSS.dat'
    
    CHtag = ['A', 'B', 'C']
    HSStag = ['1', '2', '3', '4']
    vtag = ['F', 'A', 'L']
    CHareas = [300, 800, 1500]
    HSSdists = [0.2, 0.5, 0.8, 1.1]
    allIns = [invecs, invecsAvg, invecsAvgLT]
    
    '''for k in range(3): # CME speed
        print ('')
        print ('')
        print ('')
        print ('')
        for i in range(3): # CH size
            for j in range(4): # HSS dists
                # T or S for time-dep or static
                print (i+1, j+1, vtag[k]+'T'+CHtag[i]+HSStag[j])
                #CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(allIns[k], satParams, MHDparams, silent=True, flagScales=True, doPUP=True, name=vtag[k]+'T'+CHtag[i]+HSStag[j], MEOWHiSS = [CHareas[i], HSSdists[j]], saveForces=True)
                ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(allIns[k], satParams, SWinparams, silent=True, doPUP=True, name=vtag[k]+'T'+CHtag[i]+HSStag[j], MEOWHiSS = [CHareas[i], HSSdists[j]], simYaw=True, saveForces=True)'''
                
    for k in range(3):
        ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(allIns[k], satParams, SWinparams, silent=True, doPUP=True, name=vtag[k]+'noHSS', simYaw=True, saveForces=True)
    
    # slow push option
    ATresults, outSum, vsArr, angArr, SWparams, PUPresults, FIDOresults = getAT(invecsSlow, satParams, SWinparams, silent=True, doPUP=True, name='slowPushAmb', MEOWHiSS = [800, 0.0], simYaw=True, saveForces=True)
    