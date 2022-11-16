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
    return np.array([[val], [val], [val],  [val], [val], [val], [val], [val], [val], [val], [val]]), val, [val, val, val, val, val, val], val, val, val, [val, val, val, val], [[val]*12], [[val]*8]

def add2outs(outsCME, outsSheath, outsFIDO, CMEarr, sheatharr, FIDOarr):
    # CME: 0 t, 1 r, 2 vs, 3 AW, 4 AWp, 5 delAx, 6 delCS, 7 delCA, 8 B, 9 Cnm, 10 n, 11 Temp, 12 reg
    # Sheath: 0 vS, 1 comp, 2 MA, 3 Wid, 4 Dur, 5 Mass, 6 Dens, 7 B, 8 Theta, 9 Temp, 10 Vt, 11 InSheath

    #fullCMEstuff = [0 t/3600./24., 1 CMElens[0]/rsun, 2 AW*180/pi,  3 AWp*180/pi, 4 CMElens[5]/rsun, 5 CMElens[3]/rsun, 6 CMElens[4]/rsun, 7 CMElens[6]/rsun, 8 CMElens[2]/rsun, 9 vs[0]/1e5, 10 vs[1]/1e5, 11 vs[5]/1e5, 12 vs[3]/1e5, 13 vs[4]/1e5, 14 vs[6]/1e5, 15 vs[2]/1e5, 16 rho/1.67e-24, 17 B0*1e5, 18 cnm, 19 np.log10(temCME), 20 HSSreg]
    outsCME[0].append(CMEarr[0]) 
    outsCME[1].append(CMEarr[1])
    outsCME[2].append([CMEarr[9], CMEarr[10], CMEarr[11], CMEarr[12], CMEarr[13], CMEarr[14], CMEarr[15]])
    outsCME[3].append(CMEarr[2])
    outsCME[4].append(CMEarr[3])
    outsCME[5].append(CMEarr[4] / CMEarr[7]) # CMElens[5] / CMElens[6]
    outsCME[6].append(CMEarr[5] / CMEarr[6]) # CMElens[3] / CMElens[4]
    outsCME[7].append(CMEarr[5] / CMEarr[7]) # CMElens[3] / CMElens[6]
    outsCME[8].append(CMEarr[17])
    outsCME[9].append(CMEarr[18])
    outsCME[10].append(CMEarr[16])
    outsCME[11].append(CMEarr[19])
    outsCME[12].append(CMEarr[20])
    
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
    
    if len(FIDOarr) != 0:
        outsFIDO[0].append(FIDOarr[0])
        outsFIDO[1].append(FIDOarr[1])
        outsFIDO[2].append(FIDOarr[2])
        outsFIDO[3].append(FIDOarr[3])
        outsFIDO[4].append(FIDOarr[4])
        outsFIDO[5].append(FIDOarr[5])
        outsFIDO[6].append(FIDOarr[6])
        outsFIDO[7].append(FIDOarr[7])

    return outsCME, outsSheath, outsFIDO  

def whereAmI(Epos, CMEpos, CMElens, deltax, deltap):
    # [Er, Elat, Elon], [CMElat, CMElon, CMEtilt], CMElens
    # Get Earth/sat pos in CME coordinate frame
    Epos1 = SPH2CART(Epos)
    temp = rotz(Epos1, -CMEpos[1])
    temp2 = roty(temp, CMEpos[0])
    Epos2 = rotx(temp2, 90.-CMEpos[2])
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
    CSpol = np.arccos(np.dot(vp, norm) / vpmag) 
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
    vCS = thisr * vExps[3] / delCS
    nCS = np.array([np.cos(thetaP)/delCS, np.sin(thetaP), 0.])
    normN2 = np.sqrt(np.sum(nCS**2))
    nCS = nCS / normN2
    nCSatAx = np.array([nCS[0] * np.cos(thetaT), nCS[1], nCS[0] * np.sin(thetaT)])
    vCSVec = vCS * nCSatAx
    vCMEframe = np.array([vExps[2], 0., 0.]) + vAxVec + vCSVec
   
    return vCMEframe, vCSVec



def getFIDO(axDist, maxDistFR, B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, Elat, Elon, CMElat, CMElon, CMEtilt, vs, comp=1.):
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
    temp = rotx(Btot, -(90.-CMEtilt))
    temp2 = roty(temp, CMElat - Elat) 
    Bvec = rotz(temp2, CMElon - Elon)
    
    # get velocity vector
    vCMEframe, vExpCME = getvCMEframe(axDist / maxDistFR, thisPsi, thisParat, deltax, deltap, vs)
    #print ('          ',thisParat*180/3.14, vCMEframe, vExpCME)
    # rotate to s/c frame
    temp = rotx(vCMEframe, -(90.-CMEtilt))
    temp2 = roty(temp, CMElat - Elat) 
    vInSitu = rotz(temp2, CMElon - Elon)
    
    # rotate vexp
    #temp = rotx(vExpCME, -(90.-CMEtilt))
    #temp2 = roty(temp, CMElat - Elat) 
    #vexp = rotz(temp2, CMElon - Elon)
    
    return Bvec, vInSitu, vExpCME

# -------------- main function ------------------
def getAT(invec, Epos, SWparams, SWidx=None, silent=False, fscales=None, pan=False, selfsim=False, csTens=True, thermOff=False, csOff=False, axisOff=False, dragOff=False, name='nosave', satfs=None, flagScales=False, tEmpHSS=False, tDepSW=False, doPUP=True, saveForces=False, MEOWHiSS=False, fullContact=False, aFIDOinside=False, CMEH=1, inorout=1):
    
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
    
    # polarity of FR for aFIDOinside set by CMEH, defaults to 1 for positive/right hand (-1 for left)
    # amb SW in or out set by inorout, defaults to 1 for out (-1 for in)
    
    # FIDO returns in satellite coords, not GSE
    
    # if want FIDO profile force it to do full contact
    if aFIDOinside:
        fullContact = True
    
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
        if aFIDOinside:
            fname4 = 'MH_FIDO_'+name+'.dat'
            f4 = open(fname4, 'w')

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
    runSim = True
    prevmin = 9999.
    reachedCME = False
    
    outsCME = [[] for i in range(13)]
    outsSheath = [[] for i in range(12)]
    outsFIDO = [[] for i in range(8)]

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
        # Internal flux rope forces
        if not axisOff:
            aTotNose, aTotEdge = getAxisF(deltax, deltap, CMElens[4], CMElens[6], B0, cnm, tau, rho)
            if (aTotNose != 9999) & (aTotEdge != 9999):
                magAccels += [aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), 0, 0, aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz]
            else:
                makeFailArr(8888)   
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
            doMH[0] = MHt0+t/3600

        # Update Earth/satellite position
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
        SWedge   = getSWvals(CMElens[1], SWfs, doMH=doMH)
        SWfrontB = getSWvals(CMElens[0]-2*CMElens[3], SWfs, doMH=doMH)
        SWedgeB  = getSWvals(CMElens[1]-2*CMElens[3], SWfs, doMH=doMH)
        
        
        # ------Set up state vectors at the end of a time step--------------
        # CME array
        CMEstuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap, deltaCSAx, cnm, rho/1.67e-24, np.log10(temCME), B0*1e5, HSSreg]
        fullCMEstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME), HSSreg]
        
        # PUP/sheath array
        inint = 0
        if hitSheath: inint = 1
        PUPstuff = []
        if doPUP:
            PUPstuff = [r, vShock, Ma, sheath_wid, sheath_dur, sheath_mass/1e15, sheath_dens, np.log10(Tratio*SWfront[4]), Tratio, Bsh, thetaBsh, vtSh, inint]
        
        # Forces
        forcestuff = [t/3600./24., CMElens[0]/rsun, aTotNose, aTotEdge * ndotz, aTotEdge * np.sqrt(1-ndotz**2), aTotNose - aTotEdge * np.sqrt(1-ndotz**2), aTotEdge * ndotz,    aBr, aBr * ndotz, aBr, aBr/deltap,   aPTr, aPTr * ndotz, aPTr, aPTp,     dragAccels[0], dragAccels[1], dragAccels[2], dragAccels[3], dragAccels[4], dragAccels[5], dragAccels[6]]
        
        FIDOstuff = [] # will replace if needed
        
        
        
        
        # ----------------- Save and print everything ----------------------
        # Printing/saving at lower time resolution so check if it is go time
        printNow = False
        if (CMElens[0]>printR):
            printNow = True
            printR += deltaPrintR
            
            # print to screen    
            if not silent:
                print2screen(CMEstuff)                
                if doPUP:
                    print2screen(PUPstuff, prefix='      ') 
                 
                                        
            # print to file (if doing)
            if writeFile: 
                print2file(fullCMEstuff, f1, '{:8.5f}')
                if doPUP:
                    print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')                
                if saveForces:  
                    print2file(forcestuff, f3, '{:11.3f}')
            
            # save SW values to FIDO file (if doing)
            if aFIDOinside and (CMElens[0] > Er*0.5) and not hitSheath and not reachedCME:
                # get SW values
                if doMH:
                    SWatsat, HSSreg   = getSWvals(Er, SWfs, doMH=doMH, returnReg=doMH)
                    FIDOstuff = [t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg]
                    if writeFile: print2file(FIDOstuff, f4, '{:8.5f}')
                else:
                    SWatsat   = getSWvals(Er, SWfs)
                    FIDOstuff = [t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100]
                    if writeFile: print2file(FIDOstuff, f4, '{:8.5f}')
                    
                    
                    
                    
        # -----------------------------------------------------------------------------
        # Stuff that's only done once inside the CME/sheath
        # -----------------------------------------------------------------------------
         
        #Er = 50 * 7e10 # quick switch for debugging and not runnning full thing
        # cant' just swap Er in inputs if scaling SW params off of it...
        
        # Start checking when close to sat distances, don't waste code time before close
        check_dist = 0.95*Er
        # front for checking purposes -> could be sheath or CME
        theFront = CMElens[0]
        if doPUP: theFront = CMElens[0] + sheath_wid * 7e10
        
        if theFront >= check_dist:
            # Get Earth/sat pos in CME coordinate frame            
            axDist, maxDistFR, thisPsi, thisParat = whereAmI([Er, Elat, Elon], [CMElat, CMElon, CMEtilt], CMElens, deltax, deltap)

            if doPUP: maxrSH = maxDistFR + sheath_wid * 7e10

            # check if in sheath
            if doPUP:
                if (axDist < maxrSH) and (maxDistFR != maxrSH):
                    if not hitSheath:
                        hitSheath = True
                        inint = 1
                        # Add this specific time step if not already included
                        if not printNow:
                            if not silent:
                                print2screen(CMEstuff)     
                                if doPUP:
                                    # need to update init
                                    PUPstuff[12] = inint
                                    print2screen(PUPstuff, prefix='      ')                                            
                            if writeFile:
                                print2file(fullCMEstuff, f1, '{:8.5f}')
                                print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')
                    # print sheath info to FIDO output file
                    if (axDist > maxDistFR) and (Er > CMElens[0]) and printNow and aFIDOinside:
                        tdir, pdir = getBvector(CMElens, axDist, thisPsi, thisParat, deltax, deltap)
                        ndir = np.cross(tdir, pdir)
                        # attempt to get signs correct in sheath based on inorout, flip unit vecs to most aligned choice
                        temp = rotx(ndir, -(90.-CMEtilt))
                        temp2 = roty(temp, CMElat - Elat) 
                        ndirSAT = np.array(rotz(temp2, CMElon - Elon))
                        
                        temp = rotx(pdir, -(90.-CMEtilt))
                        temp2 = roty(temp, CMElat - Elat) 
                        pdirSAT = np.array(rotz(temp2, CMElon - Elon))
                        
                        # figure out what is most aligned
                        if np.dot(ndirSAT, np.array([inorout,0,0])) < 0:
                            ndirSAT = -ndirSAT
                        if np.dot(pdirSAT, np.array([0, -inorout,0])) < 0:
                            pdirSAT = -pdirSAT
                        
                        BinSitu = Bsh * np.cos(thetaBsh*math.pi/180.) * ndirSAT + Bsh * np.sin(thetaBsh*math.pi/180.) * pdirSAT 
                        
                        FIDOstuff = [t/3600., BinSitu[0], BinSitu[1], BinSitu[2], vs[0]/1e5, sheath_dens, np.log10(Tratio*SWfront[4]),0]
                        if writeFile: print2file(FIDOstuff,  f4, '{:8.5f}')
                        
                        if not silent:
                            print ('      ', '{:8.5f}'.format(axDist/maxDistFR), '{:8.5f}'.format(thisPsi*180/3.14159), '{:8.5f}'.format(thisParat*180/3.14159))
            # check if in FR
            thismin = axDist/maxDistFR
            if axDist < maxDistFR:
                # start saving at higher res once in CME
                if not reachedCME:
                     deltaPrintR = deltaPrintR/2.
                     FRstart = t
                
                if aFIDOinside and printNow:
                    BSC, vInSitu, printthis = getFIDO(axDist, maxDistFR, B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, Elat, Elon, CMElat, CMElon, CMEtilt, vs)         
                    #vA = np.sqrt((BSC[0]**2 + BSC[1]**2 + BSC[2]**2) / 4 / 3.14159 / (rho/1.67e-24))*1e5
                    CMEgam = fT+1
                    cs = np.sqrt(2*(CMEgam) * 1.38e-16 *temCME / 1.67e-24)/1e5
                    mch = -vs[3]*(axDist/maxDistFR)/1e5 / cs
                    comp = 1
                    if mch >1:
                        comp = (CMEgam+1)*mch**2 / (2 + (gam-1)*mch**2)
                        if comp < 1: comp = 1
                        BSC, vInSitu, printthis = getFIDO(axDist, maxDistFR, B0, CMEH, tau, cnm, deltax, deltap, CMElens, thisPsi, thisParat, Elat, Elon, CMElat, CMElon, CMEtilt, vs, comp=comp) 
                        
                    if not silent:
                        print ('      ', '{:8.5f}'.format(axDist/maxDistFR), '{:8.5f}'.format(thisPsi*180/3.14159), '{:8.5f}'.format(thisParat*180/3.14159))
                    
                    
                    # save to file
                    FIDOstuff = [t/3600., BSC[0]*1e5, BSC[1]*1e5, BSC[2]*1e5, vInSitu[0]/1e5, rho/1.67e-24 * comp, np.log10(temCME), 1]
                    if writeFile: print2file(FIDOstuff, f4, '{:8.5f}')
                
                # Add this specific timestep if it isn't already included
                if not printNow and not reachedCME:
                    outsCME, outsSheath, outsFIDO = add2outs(outsCME, outsSheath, outsFIDO, fullCMEstuff, PUPstuff, FIDOstuff)                 
                        
                    if writeFile:
                        print2file(fullCMEstuff, f1, '{:8.5f}')
                        if doPUP:
                            print2file([t/3600./24., CMElens[0]/rsun]+PUPstuff, f2, '{:8.5f}')   
                        if saveForces:
                            print2file(forcestuff, f3, '{:11.3f}')
                        if aFIDOinside:
                            print2file(FIDOstuff,  f4, '{:8.5f}')
                            
                    if not silent:
                        print2screen(CMEstuff)
                        if doPUP:
                            print2screen(PUPstuff, prefix='      ')                 

                reachedCME = True
                TT = t/3600./24.
                
                # If not doing full contact mode return everything at the time of first FR impact    
                if not fullContact:
                    estDur = 4 * CMElens[3] * (2*(vs[0]-vs[3])+3*vs[3])/(2*(vs[0]-vs[3])+vs[3])**2 / 3600.
                    if not silent:
                        print ('Transit Time:     ', TT)
                        print ('Final Velocity:   ', vs[0]/1e5, vs[3]/1e5)
                        print ('CME nose dist:    ', CMElens[0]/7e10)
                        print ('Earth longitude:  ', Elon)
                        print ('Est. Duration:    ', estDur)
                        
                    # Stop simulation and return results 
                    runSim = False  
                    outsCME = np.array(outsCME)
                    outsSheath = np.array(outsSheath)
                    outsFIDO = np.array(outsFIDO)
                    for i in range(len(outsCME)): outsCME[i] = np.array(outsCME[i])
                    for i in range(len(outsSheath)): outsSheath[i] = np.array(outsSheath[i])
                    for i in range(len(outsFIDO)): outsFIDO[i] = np.array(outsFIDO[i])
                    return outsCME, Elon, vs, estDur, thisPsi, thisParat, SWfront, outsSheath, outsFIDO 
            elif reachedCME:
                runSim = False
                actDur = (t-FRstart)/3600
                if not silent:
                    print ('Transit Time:     ', FRstart/3600)
                    print ('Final Velocity:   ', vs[0]/1e5, vs[3]/1e5)
                    print ('CME nose dist:    ', CMElens[0]/7e10)
                    print ('Earth longitude:  ', Elon)
                    print ('Est. Duration:    ', actDur)
                    
                    
                # if aFIDOinside add SW padding behind the CME
                for i in range(18):
                    t += 3600.
                    # Update HSS position
                    if hasattr(MEOWHiSS, '__len__'):
                        doMH[0] = MHt0+t/3600

                    # Update Earth/satellite position
                    if not doPath:    
                        Elon += Erotrate * dt
                    else:
                        Elat = fLat(t)
                        Elon = fLon(t)
                        Er   = fR(t)*7e10
                    # get SW values
                    if doMH:
                        SWatsat, HSSreg   = getSWvals(Er, SWfs, doMH=doMH, returnReg=doMH)
                    else:
                        SWatsat   = getSWvals(Er, SWfs)
                        HSSreg = 0
                    
                    FIDOarr = [t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg]
                    outsFIDO[0].append(FIDOarr[0])
                    outsFIDO[1].append(FIDOarr[1])
                    outsFIDO[2].append(FIDOarr[2])
                    outsFIDO[3].append(FIDOarr[3])
                    outsFIDO[4].append(FIDOarr[4])
                    outsFIDO[5].append(FIDOarr[5])
                    outsFIDO[6].append(FIDOarr[6])
                    outsFIDO[7].append(FIDOarr[7])
                        
                    if writeFile:
                        print2file([t/3600, inorout*SWatsat[2]*1e5, -inorout*SWatsat[3]*1e5, 0., SWatsat[1]/1e5, SWatsat[0]/1.67e-24, np.log10(SWatsat[4]), 100+HSSreg], f4, '{:8.5f}')
                        
                
                    
                outsCME = np.array(outsCME, dtype=object)
                outsSheath = np.array(outsSheath)
                outsFIDO = np.array(outsFIDO)
                for i in range(len(outsCME)): outsCME[i] = np.array(outsCME[i])
                for i in range(len(outsSheath)): outsSheath[i] = np.array(outsSheath[i])
                for i in range(len(outsFIDO)): outsFIDO[i] = np.array(outsFIDO[i])
                return outsCME, Elon, vs, actDur, thisPsi, thisParat, SWfront, outsSheath, outsFIDO
            elif (thismin < prevmin):
                prevmin = thismin
            elif CMElens[0] > Er + 100 * 7e10:
                return makeFailArr(9999)   
            else:                
                return makeFailArr(9999)   

        # save data in arrays with outsFIDO set to whatever is appropriate based on region checking       
        if printNow:
            outsCME, outsSheath, outsFIDO = add2outs(outsCME, outsSheath, outsFIDO, fullCMEstuff, PUPstuff, FIDOstuff)  
        
        
        
        

if __name__ == '__main__':
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, CMEr0, Bscale, Cd, tau, cnm, Tscale, gammaT]   
    # Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth    
    # SWparams = [nSW, vSW, BSW, TSW] at Eradius
    
    invecs = [0, 0, 0, 1200.0, 10.0, 46, 18, 0.6, 0.6, 21.5, 3200, 1.0, 1., 1.927, 7.5e5, 1.33] # fast, use with flagScales = True 
    invecsAvg = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 3.9e5, 1.33] # avg, use with flagScales = True 
    invecsAvgLT = [0, 0, 0, 630, 5., 31, 10, 0.53, 0.7, 21.5, 1350, 1.0, 1., 1.927, 2e5, 1.33] # avg, use with flagScales = True 
    invecsSlow = [0, 0, 0, 350, 2., 25, 7, 0.7, 0.7, 21.5, 500, 1.0, 1., 1.927, 2e5, 1.33] # avg, use with flagScales = True 

    satParams = [1.0, 00.0, 215.0, 0.0]
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
    
    for k in [0]:#range(2): # CME speed
        print ('')
        print ('')
        print ('')
        print ('')
        for i in [0]:#range(3): # CH size
            for j in [0]:#range(4): # HSS dists
                print (i+1, j+1, vtag[k]+'T'+CHtag[i]+HSStag[j])
                #CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(allIns[k], satParams, MHDparams, silent=True, flagScales=True, doPUP=True, name=vtag[k]+'T'+CHtag[i]+HSStag[j], MEOWHiSS = [CHareas[i], HSSdists[j]], saveForces=True)
                CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts, FIDOouts = getAT(invecsAvgLT, satParams, MHDparams, silent=False, flagScales=True, doPUP=True, MEOWHiSS=[400,0.], saveForces=True, name='new')
                
                #CMEouts, Elon, vs, estDur, thisPsi, parat, SWfront, sheathOuts = getAT(invecsAvgLT, satParams, MHDparams, silent=False, flagScales=True, doPUP=True)
            
                '''print (vtag[k]+'S'+CHtag[i]+HSStag[j])
                shidx = -1
                if sheathOuts[-1][-1] == 1:
                    shidx = np.min(np.where(sheathOuts[-1] ==1 ))
                print (vs[0]/1e5, vs[3]/1e5, CMEouts[3][-1], CMEouts[4][-1], CMEouts[5][-1], CMEouts[6][-1], CMEouts[8][-1], CMEouts[10][-1], CMEouts[11][-1], CMEouts[0][-1]*24, estDur, sheathOuts[1][shidx], sheathOuts[0][shidx], sheathOuts[2][shidx], sheathOuts[3][shidx], sheathOuts[4][shidx], sheathOuts[5][shidx], sheathOuts[6][shidx], sheathOuts[9][shidx], sheathOuts[7][shidx], sheathOuts[8][shidx], sheathOuts[10][shidx], CMEouts[0][shidx]*24, CMEouts[12][-1])'''
    
