import numpy as np
import math
import sys
import os.path

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
    
def processANTinputs(input_values):
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
        print('Assuming satellite at 213 Rs (L1)')
    try: 
        Epos[3] = float(input_values['SatRot'])
    except:
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
        return aTotNose, aTotEdge
    else:
        return 9999, 9999
    
def getCSF(deltax, deltap, bp, c, B0, cnm, tau, rho, Btot2, csTens):
    # Internal cross section forces
    coeff3 = -deltap**3 * 2/ 3/(deltap**2+1)/cnm**2
    # Option to turn of cross-section tension
    if not csTens: coeff3 = 0.
    coeff4 = deltap**3*(tau/3. - 0.2)
    coeff4sw = deltap*(Btot2/B0**2) / 8.
    kNose = 1.3726 * deltax / c   
    RcNose = 1./kNose   
    toAccN = deltap * pi * bp**2 * RcNose * rho
    aBr = (coeff3 + coeff4 - coeff4sw) * B0**2 * bp * RcNose / toAccN / deltap
    return aBr
    
def getThermF(CMElens, temCME, nCME, nSW):
    # Get temperatures
    temSW = 6.2e4 * np.power((CMElens[0] - CMElens[3]) / 213. / rsun, -0.58)
    temSWf = 6.2e4 * np.power(CMElens[0] / 213. / rsun, -0.58)
    temSWb = 6.2e4 * np.power((CMElens[0] - 2*CMElens[3]) / 213. / rsun, -0.58)
    #temSWr = 0.5 * (temSWf + temSWb)
    rE = np.sqrt((CMElens[0] - CMElens[3])**2 + CMElens[4]**2)
    temSWe = 6.2e4 * np.power(rE / 212. / rsun, -0.58)    
    # Scale densities
    nE = nSW * (CMElens[0] / rE)**2
    #nC = nSW * ((CMElens[0] / (CMElens[0] - CMElens[3])))**2
    nB = nSW * ((CMElens[0] / (CMElens[0] - 2*CMElens[3])))**2
    # average n*T of front and back
    avgNT = 0.5 * (nB * temSWb + nSW * temSWf)    
    # Calc difference in pressure
    #delP = 1.38e-16 * (nCME * temCME - 2 * nSW * temSW)
    #delPr = 2 * 1.38e-16 * (nCME * temCME - 2 * nC* temSWr)
    delPr =  1.38e-16 * 2*(nCME * temCME - avgNT)
    delPp =  1.38e-16 * 2*(nCME * temCME - nE * temSWe)
    # Calc gradients
    gradPr = delPr / CMElens[3]
    gradPp = delPp / CMElens[4]
    return gradPr, gradPp
    
def getDrag(CMElens, vs, Mass, AW, AWp, Cd, rhoSWn, rhoSWe, vSW, ndotz):
    # dragAccels = [dragR, dragEdge, bulk, dragBr, dragBp, dragA, dragC]
    dragAccels = np.zeros(7)
    # Radial Drag
    CMEarea = 4*CMElens[1]*CMElens[4] 
    dragAccels[0] = -Cd*CMEarea*rhoSWn * (vs[0]-vSW) * np.abs(vs[0]-vSW) / Mass
    # Edge Drag
    CMEarea2 = 2*CMElens[4]*(CMElens[5] + CMElens[3])
    dragAccels[1] = -Cd*CMEarea2*rhoSWe * (vs[1]-np.sin(AW)*vSW) * np.abs(vs[1]-np.sin(AW)*vSW)  / Mass
    # CS Perp Drag
    CMEarea3 = 2*CMElens[3]*lenFun(CMElens[5] / CMElens[6]) * CMElens[6]
    dragAccels[4] = -Cd*CMEarea3*rhoSWn * (vs[4]-np.sin(AWp)*vSW) * np.abs(vs[4]-np.sin(AWp)*vSW) / Mass 
    # Individual components  
    dragAccels[2] = dragAccels[0] * (vs[2]/vs[0]) 
    dragAccels[3] = dragAccels[0] * (vs[3]/vs[0])
    dragAccels[5] = dragAccels[0] * (vs[5]/vs[0])
    dragAccels[6] = dragAccels[1] - dragAccels[0] * (vs[3]/vs[0]) / ndotz    
    return dragAccels
    
def IVD(vFront, AW, AWp, deltax, deltap, deltaCA, alpha, ndotz, fscales):
    vs = np.zeros(7)
    # vs = [vFront 0, vEdge 1, vBulk 2, vexpBr 3, vexpBp 4, vexpA 5, vexpC 6]
    vs[0] = vFront

    f1 = fscales[0]
    f2 = fscales[1]
    # Calculate the nus for conv and self sim
    nu1C = np.cos(AWp)
    nu2C = np.cos(AW)
    nu1S = 1 / (1 + deltap * np.tan(AWp))
    nu2S = 1 - (1 - nu1S) * (1 + deltax / deltaCA)
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
    
def initCMEparams(deltaAx, deltaCS, deltaCSAx, AW, AWp, CMElens, Mass, printNow=False):
    # CME shape
    # alpha * CMElens[3] is the cross sec width in z dir
    CMElens[4] = np.tan(AWp) / (1 + deltaCS * np.tan(AWp)) * CMElens[0]
    CMElens[3] = deltaCS * CMElens[4]
    CMElens[6] = CMElens[3] / deltaCSAx 
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
    BSWr = Bsw0 * (rFront0 / CMElens[0])**2
    BSWphi = BSWr * 2.7e-6 * CMElens[0]  / vSW
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
                
    

# -------------- main function ------------------
def getAT(invec, Epos, silent=False, fscales=None, pan=False, csTens=True, thermOff=False, csOff=False, axisOff=False, dragOff=False, name='nosave'):
    
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
    deltaCSAx  = invec[9]
    rFront     = invec[10] * rsun
    Bscale     = invec[11]
    n1AU       = invec[12]
    vSW        = invec[13] * 1e5
    B1AU       = invec[14] # may not actually be 1AU if rFinal isn't
    Cd         = invec[15]
    tau        = invec[16] # 1 to mimic Lundquist
    cnm        = invec[17] # 1.927 for Lundquist
    temScale   = invec[18] 
    
    writeFile = False
    if name != 'nosave':
        writeFile = True
        fname = 'PARADE_'+name+'.dat'
        f1 = open(fname, 'w')
    
    t = 0.
    dt = 60. # in seconds
    CMElens = np.zeros(8)
    #CMElens = [rFront, rEdge, d, br, bp, a, c, rXCent]
    CMElens[0] = rFront
    alpha, ndotz, rho = initCMEparams(deltax, deltap, deltaCSAx, AW, AWp, CMElens, Mass)
        
    # B1AU is the total magnetic field strength at rFinal
    # need to split into Br and Bphi
    BphiBr = 2.7e-6 * Er  / vSW 
    Br1AU = B1AU / np.sqrt(1+BphiBr**2)
    # Initial Solar Wind Properties
    Bsw0 = Br1AU * (Er/CMElens[0])**2 / 1e5    
    Btot2, rhoSWn, rhoSWe = updateSW(Bsw0, BphiBr, vSW, n1AU, Er, rFront, CMElens)
    
    # Set up factors for scaling B through conservation
    B0 = Bscale * Bsw0 / deltap / tau
    B0scaler = B0 * deltap**2 * CMElens[4]**2 
    initlen = lenFun(CMElens[5]/CMElens[6]) * CMElens[6]
    cnmscaler = cnm / initlen * CMElens[4] * (deltap**2+1)

    # get CME temperature based on expected SW temp at center of nose CS
    temSW = 6.2e4 * np.power((CMElens[0] - CMElens[3]) / 212. / rsun, -0.58)
    temCME = temScale * temSW
    temscaler = np.power(CMElens[3] * CMElens[4] * initlen, 2./3) * temCME

    # Set up the initial velocities
    if fscales == None:
        if pan: 
            fscales = [1., 1.]
        else:
            fscales = [0., 0.]
    vs = IVD(vFront, AW, AWp, deltax, deltap, deltaCSAx, alpha, ndotz, fscales)    
            
    # calculate any E rot between the time given and the start of the ANTEATR simulation
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
                return np.array([[8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888], [8888]]), 8888, 8888, 8888, 8888, 8888 
                
        if not csOff:
            aBr = getCSF(deltax, deltap, CMElens[4], CMElens[6], B0, cnm, tau, rho, Btot2, csTens)
            magAccels += [aBr, aBr * ndotz, 0, aBr, aBr/deltap, 0., 0.] 
        totAccels += magAccels 
        
        # Thermal Expansion
        if not thermOff:
            aPTr, aPTp = getThermF(CMElens, temCME, rho/1.67e-24, rhoSWe/1.627e-24)/rho
            thermAccels += [aPTr, aPTr * ndotz, 0, aPTr, aPTp, 0., 0.] 
        totAccels += thermAccels
               
        # Drag Force
        if not dragOff:
            dragAccels = getDrag(CMElens, vs, Mass, AW, AWp, Cd, rhoSWn, rhoSWe, vSW, ndotz)
            # Update accels to include drag forces
            totAccels += dragAccels
                    
        # Update CME shape
        CMElens[:7] += vs*dt + 0.5*totAccels*dt**2    
        CMElens[7] = CMElens[2] + CMElens[5]
        
        #print (CMElens[0]/rsun, CMElens[3]/rsun, aTotNose, aTotEdge, aBr)
        # Update velocities
        vs += totAccels * dt
        
        # Calc shape params and new AWs
        deltap, deltax, deltaCSAx, alpha, ndotz, AW, AWp, rho = updateCME(CMElens, Mass)
        if alpha < 0: 
            #print (CMElens/rsun)
            return np.array([[8888], [8888], [8888],  [8888], [8888], [8888], [8888], [8888], [8888], [8888]]), 8888, [8888, 8888, 8888, 8888, 8888, 8888], 8888, 8888, 8888 
                
        # Update flux rope field parameters
        lenNow = lenFun(CMElens[5]/CMElens[6])*CMElens[6]
        B0 = B0scaler / deltap**2 / CMElens[4]**2 
        cnm = cnmscaler * lenNow  / CMElens[4] / (deltap**2+1)
        Btor = deltap * tau * B0
        Bpol = 2*deltap*B0/cnm/(deltap**2+1)
        # Update temperature
        temCME = temscaler / np.power(CMElens[3] * CMElens[4] * lenNow, 2./3)
                
        # Update solar wind properties
        Btot2, rhoSWn, rhoSWe = updateSW(Bsw0, BphiBr, vSW, n1AU, Er, rFront, CMElens)
            
        Elon += Erotrate * dt
        t += dt
        if (CMElens[0]>printR):
            if CMElens[0] < 50*7e10: 
                printR += 7e10
            else:
                printR += 5*7e10
            outTs.append(t/3600./24.)
            outRs.append(CMElens[0]/7e10)
            outvs.append(vs/1e5)
            outAWs.append(AW*180./3.14159)
            outAWps.append(AWp*180/pi)
            outdelxs.append(deltax)
            outdelps.append(deltap)
            outdelCAs.append(deltaCSAx)
            outBs.append(B0)
            outCnms.append(cnm)
            outns.append(rho/1.67e-24)
            outTems.append(np.log10(temCME))
            if True:
                printStuff = [t/3600., CMElens[0]/7e10, vs[0]/1e5, AW*radeg, AWp*radeg, deltax, deltap, deltaCSAx, cnm, rho/1.67e-24, temCME, rho/1.67e-24, B0*1e5]
                if not silent:
                    printThis = ''
                    for item in printStuff:
                        printThis = printThis + '{:6.2f}'.format(item) + ' '
                    print (printThis)
            # print to file (if doing)
            if writeFile:
                fullstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME)] 
                outstuff2 = ''
                for item in fullstuff:
                    outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' '
                f1.write(outstuff2+'\n')  
            
            
        # Determine if sat is inside CME once it gets reasonably close
        if CMElens[0] >= 0.95*Er:
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
            maxr = np.sqrt(deltap**2 * np.cos(parat)**2 + np.sin(parat)**2) * CMElens[4]
            
            if vpmag < maxr:
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
                    outBs.append(B0)
                    outCnms.append(cnm)
                    outns.append(rho/1.67e-24)
                    outTems.append(np.log10(temCME))
                    if writeFile:
                        fullstuff = [t/3600./24., CMElens[0]/rsun, AW*180/pi,  AWp*180/pi, CMElens[5]/rsun, CMElens[3]/rsun, CMElens[4]/rsun, CMElens[6]/rsun, CMElens[2]/rsun, vs[0]/1e5, vs[1]/1e5, vs[5]/1e5, vs[3]/1e5, vs[4]/1e5, vs[6]/1e5, vs[2]/1e5, rho/1.67e-24, B0*1e5, cnm, np.log10(temCME)] 
                        outstuff2 = ''
                        for item in fullstuff:
                            outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' ' 
                    
                if writeFile: f1.write(outstuff2+'\n')
                estDur = 4 * CMElens[3] * (2*(vs[0]-vs[3])+3*vs[3])/(2*(vs[0]-vs[3])+vs[3])**2 / 3600.
                if not silent:
                    print ('Transit Time:     ', TT)
                    print ('Final Velocity:   ', vs[0]/1e5)
                    print ('CME nose dist:    ', CMElens[0]/7e10)
                    print ('Earth longitude:  ', Elon)
                    print ('Est. Duration:    ', estDur)
                    
                inCME = True   
                return np.array([outTs, outRs, outvs, outAWs, outAWps, outdelxs, outdelps, outdelCAs, outBs, outCnms, outns, outTems]), Elon, vs, estDur, thisPsi, parat  
            elif thismin < prevmin:
                prevmin = thismin
            elif CMElens[0] > Er + 100 * 7e10:
                return np.array([[9999], [9999], [9999],  [9999], [9999], [9999], [9999], [9999], [9999], [9999]]), 9999, [9999, 9999, 9999, 9999, 9999, 9999], 9999, 9999, 9999 
            else:                
                return np.array([[9999], [9999], [9999],  [9999], [9999], [9999], [9999], [9999], [9999], [9999]]), 9999, [9999, 9999, 9999, 9999, 9999, 9999], 9999, 9999, 9999  
        

if __name__ == '__main__':
    # invec = [CMElat, CMElon, tilt, vr, mass, cmeAW, cmeAWp, deltax, deltap, deltaCA, CMEr0, Bscale, nSW, vSW, BSW, Cd, tau, cnm]        
    # Epos = [Elat, Elon, Eradius] -> technically doesn't have to be Earth!
    invecF = [0, 0, 0, 1250, 10, 45, 10, 0.7, 1, 0.3, 10, 3, 6.9, 440, 5.7, 1., 1, 1.927]
    satParams = np.array([0, 0, 213.0, 0*1.141e-05])
        
    #outs, Elon, vs, estDur, thetaT, thetaP = getAT(invecF, satParams, fscales=[0.5,0.5], name='temp')
    #outs, Elon, vs, estDur, thetaT, thetaP = getAT(invecF, satParams, pan=True, name='temp')
    #outs, Elon, vs, estDur, thetaT, thetaP = getAT(invecF, satParams, pan=True, name='temp')

    outs, Elon, vs, estDur, thetaT, thetaP = getAT(invecF, satParams, fscales=[0.5,0.5], name='temp')
