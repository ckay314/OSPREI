import numpy as np
import math
import sys
import os.path
from scipy.interpolate import CubicSpline
import pickle
import empHSS as emp



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
        print('Assuming satellite at 0 lat')
    try: 
        Epos[1] = float(input_values['SatLon'])
    except:
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
    dg2 = deltap**2 * gammaN**2 
    coeff1N = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6) 
    toAccN = deltap * pi * bp**2 * RcNose * rho
    coeff2N = - deltap**3 *(tau*(tau-1)+0.333)/ 4. * gammaN
    aTotNose = (coeff2N+coeff1N) * B0**2 * RcNose * bp / toAccN 
    # Edge Direction
    kEdge = 40 * deltax / c / np.sqrt(16*deltax+1)**3
    RcEdge = 1./kEdge    
    gammaE = bp / RcEdge
    dg = deltap*gammaE
    dg2 = deltap**2 * gammaE**2 
    if dg2 < 1:
        coeff1E = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6) 
        coeff2E = - deltap**3 *(tau*(tau-1)+0.333) *  gammaE / 4.
        toAccE = deltap * pi * bp**2 * RcEdge * rho
        aTotEdge =  (coeff1E+coeff2E) * B0**2 * RcEdge * bp / toAccE       
        #print (coeff1N, coeff2N, coeff1E, coeff2E)
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
    #print (coeff3, coeff4, coeff4sw, coeff3 + coeff4 - coeff4sw)
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
    # CME shape
    # alpha * CMElens[3] is the cross sec width in z dir
    CMElens[4] = np.tan(AWp) / (1 + deltaCS * np.tan(AWp)) * CMElens[0]
    CMElens[3] = deltaCS * CMElens[4]
    CMElens[6] = (np.tan(AW) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + deltaAx * np.tan(AW))  
    #CMElens[6] = CMElens[3] / deltaCSAx 
    CMElens[5] = deltaAx * CMElens[6]
    CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
    CMElens[1] = CMElens[2] * np.tan(AW) 
    # alpha is scaling of rr to rE - Lp
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

def getSWvals(r_in, SWfuncs, doMH=False, returnReg=False):
    # returns the actual values of n, vr, Br, Blon, T
    SWvec = np.zeros(5)
    for i in range(5):
        SWvec[i] = SWfuncs[i](r_in)
    if hasattr(doMH, '__len__'):
        if returnReg:
            MHouts, HSSreg = emp.getHSSprops(r_in/1.5e13, doMH[0], doMH[1], doMH[2], doMH[3], doMH[4], doMH[5], returnReg=True)
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
            print ('here')
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

# -------------- main function ------------------
def getAT(invec, Epos, SWparams, SWidx=None, silent=False, fscales=None, pan=False, selfsim=False, csTens=True, thermOff=False, csOff=False, axisOff=False, dragOff=False, name='nosave', satfs=None, flagScales=False, tEmpHSS=False, tDepSW=False, doPUP=True, saveForces=False, MEOWHiSS=False):
    
    Elat      = Epos[0]
    Elon      = Epos[1]
    Er        = Epos[2] *7e10
    Erotrate  = Epos[3]  # should be in deg/sec
    
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
    Bscale     = invec[10]
    Cd         = invec[11]
    tau        = invec[12] # 1 to mimic Lundquist
    cnm        = invec[13] # 1.927 for Lundquist
    temScale   = invec[14] 
    fT         = invec[15] - 1 # 0 for isothermal, 2./3 for adiabatic
    
    # sheath properties
    sheath_mass = 0.
    sheath_wid  = 0.
    sheath_dens = 0.
    hitSheath = False
    inint = 0
    global Tratio, rratio
    Tratio = 1
    rratio = 1
           
    
    if flagScales:
        CMEB = invec[10] / 1e5
        CMET = invec[14]
    
    # Get ambient background SW - still need ambient event if doing
    # HSS bc scaled off of it
    # Check if passed 1 AU values -> scaling of empirical model
    if not isinstance(SWparams, str):    
        SWparams = [SWparams[0], SWparams[1], SWparams[2], SWparams[3], Er/1.5e13]
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

    t = 0.
    dt = 60. # in seconds
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
    SWedge   = getSWvals(CMElens[1], SWfs, doMH=doMH)
    SWfrontB = getSWvals(CMElens[0]-2*CMElens[3], SWfs, doMH=doMH)
    SWedgeB  = getSWvals(CMElens[1]-2*CMElens[3], SWfs, doMH=doMH)
        
    # Set up factors for scaling B through conservation
    # this was scaled of Bsw0 insteand of sqrt(Btot2) before...
    B0 = Bscale * np.sqrt(Btot2) / deltap / tau
    if flagScales: B0 = CMEB/ deltap / tau
    B0scaler = B0 * deltap**2 * CMElens[4]**2 
    initlen = lenFun(CMElens[5]/CMElens[6]) * CMElens[6]
    cnmscaler = cnm / initlen * CMElens[4] * (deltap**2+1)
    initcs = CMElens[3] * CMElens[4]

    # get CME temperature based on expected SW temp at center of nose CS
    temSW = getSWvals(CMElens[0]-CMElens[3], SWfs, doMH=doMH)[4]
    temCME = temScale * temSW
    if flagScales: temCME = CMET
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
            
    # set up whether using path functions or simple orbit
    if satfs == None:
        doPath = False
    else:
        doPath = True
        fLat = satfs[0]
        fLon = satfs[1]
        fR   = satfs[2]
        
    printR = rFront
    inCME = False
    prevmin = 9999.
    
    # arrays for saving profiles
    outTs     = []
    outRs     = []
    outvs     = []
    outAWs    = []
    outAWps   = []
    outdelxs  = []
    outdelps  = []
    outdelCAs = []
    outBs     = []
    outCnms   = []
    outns     = []
    outTems   = []
    outRegs   = []
    
    if doPUP:
        outvShocks  = []
        outcomps    = []
        outMachs    = []
        outWidths   = []
        outDurs     = []
        outMass     = []
        outDens     = []
        outShBs     = []
        outThetaBs  = []
        outShTs     = []
        outShVts    = []
        outInSh     = []
                
    while not inCME:
    #while CMElens[0] <= 0.9*Er:
        # Accels order = [Front, Edge, Center, CSr, CSp, Axr, Axp,]
        totAccels = np.zeros(7)
        magAccels = np.zeros(7)
        thermAccels = np.zeros(7)  
        # Calculate forces
        # Internal flux rope forces
        if not axisOff:
            aTotNose, aTotEdge = getAxisF(deltax, deltap, CMElens[4], CMElens[6], B0, cnm, tau, rho)
            if (aTotNose != 9999) & (aTotEdge != 9999):
                magAccels += [aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), 0, 0, aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz]
            else:
                return np.array([[8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888]]), 8888, 8888, 8888, 8888, 8888, [8888, 8888, 8888], [[8888]*12]     
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
               
        # Drag Force
        if not dragOff:
            dragAccels = getDrag(CMElens, vs, Mass+sheath_mass, AW, AWp, Cd, SWfront, SWfrontB, SWedge, SWedgeB, ndotz)
            # Update accels to include drag forces
            totAccels += dragAccels
            
        # Sheath stuff
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
                        
                                                                
        # Update CME shape
        CMElens[:7] += vs*dt + 0.5*totAccels*dt**2    
        CMElens[7] = CMElens[2] + CMElens[5]
        
        # sheath mass 
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
        
        if doPUP:
            sheath_dur = sheath_wid*7e10/vs[0]/3600
        
        # Calc shape params and new AWs
        deltap, deltax, deltaCSAx, alpha, ndotz, AW, AWp, rho = updateCME(CMElens, Mass)
        if alpha < 0: 
            return np.array([[8888], [8888], [8888],  [8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888]]), 8888, [8888, 8888, 8888, 8888, 8888, 8888], 8888, 8888, 8888, [8888, 8888, 8888], [[8888]*12] , 
                
        t += dt
        if hasattr(MEOWHiSS, '__len__'):
            doMH[0] = MHt0+t/3600

        if not doPath:    
            Elon += Erotrate * dt
        else:
            Elat = fLat(t)
            Elon = fLon(t)
            Er   = fR(t)*7e10
                        
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
        if doMH:
            SWavgAx, reg2 = getSWvals(avgAxR, SWfs, doMH=doMH, returnReg=doMH)
        else:
            SWavgAx = getSWvals(avgAxR, SWfs, doMH=doMH, returnReg=doMH)
            reg2 = 0
        Btot2 = SWavgAx[2]**2 + SWavgAx[3]**2
        
        theFront = CMElens[0]
        if doPUP:
            theFront += sheath_dur*7e10
        if doMH:            
            SWfront, HSSreg  = getSWvals(theFront, SWfs, doMH=doMH, returnReg=doMH)
        else:
            SWfront = getSWvals(theFront, SWfs, doMH=doMH, returnReg = doMH)
            HSSreg = 0
        SWedge   = getSWvals(CMElens[1], SWfs, doMH=doMH)
        SWfrontB = getSWvals(CMElens[0]-2*CMElens[3], SWfs, doMH=doMH)
        SWedgeB  = getSWvals(CMElens[1]-2*CMElens[3], SWfs, doMH=doMH)
        
        if (CMElens[0]>printR):
            if CMElens[0] < 50*7e10: 
                printR += 7e10
            else:
                printR += 7e10
                
            if not silent:
                printStuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap, deltaCSAx, cnm, rho/1.67e-24, np.log10(temCME), B0*1e5, HSSreg]
                printThis = ''
                for item in printStuff:
                    printThis = printThis + '{:6.2f}'.format(item) + ' '
                print (printThis)
                if doPUP:
                    inint = 0
                    if hitSheath: inint = 1
                    printStuff = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Tratio, Bsh, thetaBsh, vtSh, inint]
                    printThis = ''
                    for item in printStuff:
                        printThis = printThis + '{:6.2f}'.format(item) + ' '
                    print ('      ', printThis)            
                #print (CMElens[0]/7e10/215, Ma, Btot2, SWavgAx[2], SWavgAx[3], avgAxR/7e10, vA/1e5, reg2)
                                          
            outTs.append(t/3600./24.)
            outRs.append(CMElens[0]/7e10)
            outvs.append(vs/1e5)
            outAWs.append(AW*180./3.14159)
            outAWps.append(AWp*180/pi)
            outdelxs.append(deltax)
            outdelps.append(deltap)
            outdelCAs.append(deltaCSAx)
            outBs.append(B0*1e5)
            outCnms.append(cnm)
            outns.append(rho/1.67e-24)
            outTems.append(np.log10(temCME))
            outRegs.append(HSSreg)
            if doPUP:
                outvShocks.append(vShock)  
                outcomps.append(r) 
                outMachs.append(Ma) 
                outWidths.append(sheath_wid) 
                outDurs.append(sheath_dur) 
                outMass.append(sheath_mass/1e15) 
                outDens.append(sheath_dens) 
                outShBs.append(Bsh) 
                outThetaBs.append(thetaBsh) 
                outShTs.append(np.log10(Tratio*SWfront[4])) 
                outShVts.append(vtSh) 
                outInSh.append(inint)    
            
            # print to file (if doing)
            if writeFile:
                fullstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME), HSSreg] 
                outstuff2 = ''
                for item in fullstuff:
                    outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' '
                f1.write(outstuff2+'\n')  
                
                # Print PUP stuff to its own file
                if doPUP:
                    PUPstuff = [t/3600./24., CMElens[0]/rsun, r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, Tratio, np.log10(Tratio*SWfront[4]), Bsh, thetaBsh, vtSh, inint]
                    outstuffPUP = ''
                    for item in PUPstuff:
                        outstuffPUP = outstuffPUP + '{:8.5f}'.format(item) + ' '
                    f2.write(outstuffPUP+'\n') 
                                
                # Print forces if doing    
                # [Front, Edge, Center, CSr, CSp, Axr, Axp,]
                if saveForces:
                    forcestuff = [t/3600./24., CMElens[0]/rsun, aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz,    aBr, aBr * ndotz, aBr, aBr/deltap,   aPTr, aPTr * ndotz, aPTr, aPTp,     dragAccels[0], dragAccels[1], dragAccels[2], dragAccels[3], dragAccels[4], dragAccels[5], dragAccels[6]]  
                    outprint = ''
                    for item in forcestuff:
                        outprint = outprint + '{:11.3f}'.format(item)
                    f3.write(outprint+'\n') 
                
         
        # Determine if sat is inside CME once it gets reasonably close
        # Need to check both when first enters sheath (if doing) and CME
        
        #Er = 50 * 7e10 # quick switch for debugging and not runnning full thing
        # cant' just swap Er in inputs if scaling SW params off of it...
        
        check_dist = 0.95*Er
        theFront = CMElens[0]
        if doPUP: theFront = CMElens[0] + sheath_wid * 7e10
        if theFront >= check_dist:
            # Get Earth/sat pos in CME coordinate frame
            Epos1 = SPH2CART([Er,Elat,Elon])
            temp = rotz(Epos1, -CMElon)
            temp2 = roty(temp, CMElat)
            Epos2 = rotx(temp2, 90.-CMEtilt)
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
            thisPsi *= sn

            # Find the sat position within the CS
            vpmag = np.sqrt(np.sum(vp**2))
            CSpol = np.arccos(np.dot(vp, norm) / vpmag) 
            CSxy = np.array([vpmag*np.cos(CSpol), vpmag*np.sin(CSpol)])
            parat = np.arctan(np.tan(CSpol)*deltap)
            # Get the max R for that parametric t
            maxrFR = np.sqrt(deltap**2 * np.cos(parat)**2 + np.sin(parat)**2) * CMElens[4]
            if doPUP: maxrSH = maxrFR + sheath_wid * 7e10

            # check if in sheath
            if doPUP:
                if (vpmag < maxrSH) and (maxrFR != maxrSH):
                    if not hitSheath:
                        hitSheath = True
                        inint = 1
                        if not silent:
                            printStuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap, deltaCSAx, cnm, rho/1.67e-24, np.log10(temCME), B0*1e5, HSSreg]
                            printThis = ''
                            for item in printStuff:
                                printThis = printThis + '{:6.2f}'.format(item) + ' '
                            print (printThis)
                            if doPUP:
                                inint = 0
                                if hitSheath: inint = 1
                                printStuff = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Tratio, Bsh, thetaBsh, vtSh, inint]
                                printThis = ''
                                for item in printStuff:
                                    printThis = printThis + '{:6.2f}'.format(item) + ' '
                                print ('      ', printThis)
                            
                             
                        if writeFile:
                            PUPstuff = [t/3600./24., CMElens[0]/rsun, r, vShock,  Ma, sheath_wid,  sheath_dur, sheath_mass/1e15, sheath_dens, Tratio, np.log10(Tratio*SWfront[4]), Bsh, thetaBsh, vtSh, inint]
                            outstuffPUP = ''
                            for item in PUPstuff:
                                outstuffPUP = outstuffPUP + '{:8.5f}'.format(item) + ' '
                            f2.write(outstuffPUP+'\n')
                    
            
            # check if in FR
            if vpmag < maxrFR:
                deltap, deltax, deltaCSAx, alpha, ndotz, AW, AWp, rho = updateCME(CMElens, Mass)          
                TT = t/3600./24.
                if (t/3600./24.) != outTs[-1]:
                    outTs.append(t/3600./24.)
                    outRs.append(CMElens[0]/7e10)
                    outvs.append(vs/1e5)
                    outAWs.append(AW*180./3.14159)
                    outAWps.append(AWp*180/pi)
                    outdelxs.append(deltax)
                    outdelps.append(deltap)
                    outdelCAs.append(deltaCSAx)
                    outBs.append(B0*1e5)
                    outCnms.append(cnm)
                    outns.append(rho/1.67e-24)
                    outTems.append(np.log10(temCME))
                    outRegs.append(HSSreg)
                    if doPUP:
                        outvShocks.append(vShock)  
                        outcomps.append(r) 
                        outMachs.append(Ma) 
                        outWidths.append(sheath_wid) 
                        outDurs.append(sheath_dur) 
                        outMass.append(sheath_mass/1e15) 
                        outDens.append(sheath_dens) 
                        outShBs.append(Bsh) 
                        outThetaBs.append(thetaBsh) 
                        outShTs.append(np.log10(Tratio*SWfront[4])) 
                        outShVts.append(vtSh) 
                        outInSh.append(inint)
                        
                    if writeFile:
                        fullstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME), HSSreg] 
                        outstuff2 = ''
                        for item in fullstuff:
                            outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' ' 
                        f1.write(outstuff2+'\n')  
                        f1.close()  
                        
                        # Print PUP stuff to its own file
                        if doPUP:
                            inint = 0
                            if hitSheath: inint = 1
                            PUPstuff = [t/3600./24., CMElens[0]/rsun, r, vShock,  Ma, sheath_wid,  sheath_dur, sheath_mass/1e15, sheath_dens, Tratio, np.log10(Tratio*SWfront[4]), Bsh, thetaBsh, vtSh, inint]
                            outstuffPUP = ''
                            for item in PUPstuff:
                                outstuffPUP = outstuffPUP + '{:8.5f}'.format(item) + ' '
                            f2.write(outstuffPUP+'\n') 
                            f2.close()
                            
                            
                        if saveForces:
                            forcestuff = [t/3600./24., CMElens[0]/rsun, aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz,    aBr, aBr * ndotz, aBr, aBr/deltap,   aPTr, aPTr * ndotz, aPTr, aPTp,     dragAccels[0], dragAccels[1], dragAccels[2], dragAccels[3], dragAccels[4], dragAccels[5], dragAccels[6]]  
                            outprint = ''
                            for item in forcestuff:
                                outprint = outprint + '{:11.3f}'.format(item)
                            f3.write(outprint+'\n')
                            f3.close()
                            
                    if not silent:
                        printStuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap, deltaCSAx, cnm, rho/1.67e-24, np.log10(temCME), B0*1e5, HSSreg]
                        printThis = ''
                        for item in printStuff:
                            printThis = printThis + '{:6.2f}'.format(item) + ' '
                        print (printThis)
                        if doPUP:
                            inint = 0
                            if hitSheath: inint = 1
                            printStuff = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Tratio, Bsh, thetaBsh, vtSh, inint]
                            printThis = ''
                            for item in printStuff:
                                printThis = printThis + '{:6.2f}'.format(item) + ' '
                            print ('      ', printThis)            
                    
                
                estDur = 4 * CMElens[3] * (2*(vs[0]-vs[3])+3*vs[3])/(2*(vs[0]-vs[3])+vs[3])**2 / 3600.
                if not silent:
                    print ('Transit Time:     ', TT)
                    print ('Final Velocity:   ', vs[0]/1e5, vs[3]/1e5)
                    print ('CME nose dist:    ', CMElens[0]/7e10)
                    print ('Earth longitude:  ', Elon)
                    print ('Est. Duration:    ', estDur)
                    
                # convenient way to pass SW params if using functions
                inCME = True  
                CMEouts = np.array([outTs, outRs, outvs, outAWs, outAWps, outdelxs, outdelps, outdelCAs, outBs, outCnms, outns, outTems, outRegs])
                if doPUP:
                    sheathOuts = np.array([outvShocks, outcomps, outMachs, outWidths, outDurs, outMass, outDens, outShBs, outThetaBs, outShTs, outShVts, outInSh])
                else:
                    sheathOuts = np.array([[], [], [], [], [], [], [], [], [], [], [], [], []])
                                        
                return CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts 
            elif thismin < prevmin:
                prevmin = thismin
            elif CMElens[0] > Er + 100 * 7e10:
                return np.array([[9999], [9999], [9999],  [9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999]]), 9999, [9999, 9999, 9999, 9999, 9999, 9999], 9999, 9999, 9999, [9999, 9999, 9999], [[9999]*12]  
            else:                
                return np.array([[9999], [9999], [9999],  [9999], [9999], [9999], [9999], [9999], [9999], [9999], [9999]]), 9999, [9999, 9999, 9999, 9999, 9999, 9999], 9999, 9999, 9999, [9999, 9999, 9999], [[9999]*12]
        

if __name__ == '__main__':
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, Bscale, Cd, tau, cnm, Tscale, gammaT]   
    # Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth    
    # SWparams = [nSW, vSW, BSW, TSW] at Eradius
    
    invecs = [0, 0, 0, 1200.0, 10.0, 46, 18, 0.6, 0.6, 21.5, 3200, 1.0, 1., 1.927, 7.5e5, 1.33] # fast, use with flagScales = True 
    invecsAvg = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 3.9e5, 1.33] # avg, use with flagScales = True 
    invecsAvgLT = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 2e5, 1.33] # avg, use with flagScales = True 
    invecsSlow = [0, 0, 0, 350, 2., 25, 7, 0.7, 0.7, 21.5, 500, 1.0, 1., 1.927, 2e5, 1.33] # avg, use with flagScales = True 

    satParams = [0.0, 00.0, 215.0, 0.0]
    SWparams = [5.0, 440, 6.9, 6.2e4]
    MHDparams = [4.5, 386, 3.1, 4.52e4]
    
    fname0 = 'SWnoHSS.dat'
    
    fname = 'StefanDataFull/' + 'HSS_artificial_latfromcenter=00' + 'deg_CHarea=' + '8.0e+10' + 'km2.npy'
    
    CHtag = ['A', 'B', 'C']
    HSStag = ['1', '2', '3', '4']
    vtag = ['F', 'A', 'L']
    CHareas = [300, 800, 1500]
    HSSdists = [0.2, 0.5, 0.8, 1.1]
    allIns = [invecs, invecsAvg, invecsAvgLT]
    
    for k in [2]:#range(2):
        print ('')
        print ('')
        print ('')
        print ('')
        for i in [0]:#range(3):
            for j in [0]:#range(4):
                print (i+1, j+1, vtag[k]+'T'+CHtag[i]+HSStag[j])
                #CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(allIns[k], satParams, MHDparams, silent=True, flagScales=True, doPUP=True, name=vtag[k]+'T'+CHtag[i]+HSStag[j], MEOWHiSS = [CHareas[i], HSSdists[j]], saveForces=True)
                #CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(invecsSlow, satParams, MHDparams, silent=False, flagScales=True, doPUP=True, MEOWHiSS = [CHareas[i], HSSdists[j]], saveForces=True)

                CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(invecsAvgLT, satParams, MHDparams, silent=False, flagScales=True, doPUP=True)
            
                print (vtag[k]+'S'+CHtag[i]+HSStag[j])
                shidx = -1
                if sheathOuts[-1][-1] == 1:
                    shidx = np.min(np.where(sheathOuts[-1] ==1 ))
                print (vs[0]/1e5, vs[3]/1e5, CMEouts[3][-1], CMEouts[4][-1], CMEouts[5][-1], CMEouts[6][-1], CMEouts[8][-1], CMEouts[10][-1], CMEouts[11][-1], CMEouts[0][-1]*24, estDur, sheathOuts[1][shidx], sheathOuts[0][shidx], sheathOuts[2][shidx], sheathOuts[3][shidx], sheathOuts[4][shidx], sheathOuts[5][shidx], sheathOuts[6][shidx], sheathOuts[9][shidx], sheathOuts[7][shidx], sheathOuts[8][shidx], sheathOuts[10][shidx], CMEouts[0][shidx]*24, CMEouts[12][-1])
    
