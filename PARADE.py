import numpy as np

rsun = 7e10
pi = 3.14159

# Empirical function to calculate length of axis
global lenFun
lenCoeffs = [0.61618, 0.47539022, 1.95157615]
lenFun = np.poly1d(lenCoeffs)
# calc as lenFun(a/c)

# Empirical functions to calculate hoop force coefficient
'''g1Coeffs = [ 2.32974897, 13.57489783, 28.39890803, 26.50096997, 15.37622926, -3.01860085]
g2Coeffs = [ 2.94681341, 17.03267855, 35.68191889, 33.73752841, 16.91187526,  0.29796235]
global g1, g2
g1 = np.poly1d(g1Coeffs)
g2 = np.poly1d(g2Coeffs)'''
#Calc hoop coeffs as -2.405*np.exp(g1(np.log(b/Rc))) + 2*np.exp(g2(np.log(b/Rc)))



def runParade(invec, pan=False, silent=False, save=True, csTens=True, csOff=False, axisOff=False, dragOff=False, name='temp'):
    # invec = [0: vCME (km/s), 1: mass (1e15 g), 2: AW (deg), 3: shape A, 4: shape B, 5: rFront (rsun), 6: Btor/Bswr  7: nSW (cm^-3), 8: vSW (km/s), 9: BSW (nT), 10: Cd, 11: rFinal (rsun)], 12: tau, 13: cnm
    # Solar wind values all at rFinal
    vFront = invec[0] * 1e5
    Mass   = invec[1] * 1e15
    AW     = invec[2] * pi / 180.
    AWp    = invec[3] * pi / 180.
    deltax = invec[4]
    deltap = invec[5]
    rFront = invec[6] * rsun
    Bscale = invec[7]
    n1AU   = invec[8]
    vSW    = invec[9] * 1e5
    B1AU   = invec[10] # may not actually be 1AU if rFinal isn't
    Cd     = invec[11]
    rFinal = invec[12] *rsun
    tau    = invec[13] # 1 to mimic Lundquist
    cnm    = invec[14] # 1.927 for Lundquist
    
    # Open a file to save output
    # will default to PARADE_temp.dat
    fname = 'PARADE_'+name+'.dat'
    f1 = open(fname, 'w')
    
    # B1AU is the total magnetic field strength at rFinal
    # need to split into Br and Bphi
    BphiBr = 2.7e-6 * rFinal  / vSW 
    Br1AU = B1AU / np.sqrt(1+BphiBr**2)
    # Br at the CME front at the beginning of run
    Bsw0 = Br1AU * (rFinal/rFront)**2 / 1e5
    
    t = 0.
    dt = 60. # in seconds

    # CME shape
    # alpha * br is the cross sec width in z dir
    alpha = np.sqrt(1+16*deltax**2)/4/deltax
    # normal vector dot z
    ndotz = 1./alpha
    # Convert AW and shape params to actual lengths
    Deltabr = deltap * np.tan(AWp) / (1 + deltap * np.tan(AWp))
    br = Deltabr * rFront
    c  = (np.tan(AW)*(1-Deltabr) - alpha*Deltabr)/(1+deltax*np.tan(AW)) * rFront
    a  = deltax * c
    bp = br / deltap
    d  = rFront - a - br
    rCent = d+a
    rEdge = c + br*alpha
    CMEBr = br / c
    CMEBp = bp / c
       
    # Save the initial R and CME B values
    rFront0 = rFront
    initB0 = Bscale * Bsw0 / deltap / tau
            
    # Calc AW again as will later, mostly saving as new var name
    AW = np.arctan((br*alpha+c)/d)
    # AW in perp (y) direction
    AWp = np.arctan(bp/(d+a))
    
    # Initial values of things we use to scale B parameters
    initlen = lenFun(a/c)*c
    initbp = bp
    initdeltap = deltap
    initcnm = cnm
    
    # Two options for initial velocities:
    # 1. Convective velocities - following traditional
    # pancaking method with few diff assumptions
    if pan:
        vexpA  = vFront * (np.cos(AWp) - np.cos(AW))
        vexpBr = 2*vFront * (1 - np.cos(AWp))
        vexpBp = vFront * np.sin(AWp)    
        vBulk  = vFront * np.cos(AW)
        vEdge  = vFront * np.sin(AW)
        vexpC = vEdge - alpha * vexpBr
    else:    
        # 2. self-similar velocities
        vexpC = vFront * c/rFront
        vexpA = vexpC * deltax
        vexpBr = vexpC * CMEBr
        vexpBp = vexpC * CMEBp
        vBulk = vFront - vexpA - vexpBr
        vEdge = vexpC + vexpBr

    # Simulation loop
    i = 0    
    printR = int(rFront)
    while rFront <= rFinal:
        # Set accels to 0
        totAccelBr = 0.
        totAccelBp = 0.
        totAccelD = 0.
        totAccelE = 0.
        totAccelF = 0.
        totAccelA = 0.
        totAccelC = 0.
        
        # Calc shape params
        deltap = br / bp

        # Update solar wind properties
        rhoSWn = 1.67e-24 * n1AU * (rFinal/rFront)**2
        rhoSWe = 1.67e-24 * n1AU * (rFinal/(rEdge))**2
        # Bsw values at nose
        BSWr = Bsw0 * (rFront0 / rFront)**2
        BSWphi = BSWr * 2.7e-6 * rFront  / vSW 
        
        # Update flux rope field parameters
        lenNow = lenFun(a/c)*c
        B0 = initB0 * initdeltap**2/deltap**2 * initbp**2 / bp**2 #(CAtor / (br*bp))
        cnm = initcnm * lenNow / initlen * initbp / bp * (initdeltap**2+1) / (deltap**2+1)
            
        # Update CME density
        vol = pi*br*bp *  lenFun(a/c)*c
        rho = Mass / vol
                
        # Force at nose
        # Internal flux rope forces
        kNose = 1.3726 * deltax / c   
        #kNose = a / c**2 # torus version
        RcNose = 1./kNose   
        gammaN = br / RcNose        
        dg = deltap * gammaN
        dg2 = deltap**2 * gammaN**2 
        coeff1N = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6)
        toAccN = deltap * pi * bp**2 * RcNose * rho
        coeff2N = - deltap**3 *(tau*(tau-1)+0.333)/ 4. * gammaN
        if not axisOff:
            aTotNose = (coeff2N+coeff1N) * B0**2 * RcNose * bp / toAccN        
            totAccelF += aTotNose
            totAccelA += aTotNose
            
        # Force at flank
        # Internal flux rope forces
        kEdge = 40 * deltax / c / np.sqrt(16*deltax+1)**3
        #kEdge = c / a**2 # torus version
        RcEdge = 1./kEdge    
        gammaE = br / RcEdge
        dg = deltap*gammaE
        dg2 = deltap**2 * gammaE**2 
        coeff1E = deltap**2 / dg**3 / np.sqrt(1-dg2)/(1+deltap**2)**2 / cnm**2 * (np.sqrt(1-dg2)*(dg2-6)-4*dg2+6)
        coeff2E = - deltap**3 *(tau*(tau-1)+0.333) *  gammaE / 4.
        toAccE = deltap * pi * bp**2 * RcEdge * rho
        if not axisOff:
            aTotEdge =  (coeff1E+coeff2E) * B0**2 * RcEdge * bp / toAccE
            totAccelE += aTotEdge
            totAccelC += aTotEdge
        
        # Internal cross section forces
        Btot2 = BSWr**2 + BSWphi**2
        coeff3 = -deltap**3 * 2/ 3/(deltap**2+1)/cnm**2
        # Option to turn of cross-section tension
        if not csTens: coeff3 = 0.
        coeff4 = deltap**3*(tau/3. - 0.2)
        coeff4sw = 0*deltap*(Btot2/B0**2) / 8.
        if not csOff:
            aBr = (coeff3 + coeff4 - coeff4sw) * B0**2 * bp * RcNose / toAccN / deltap
            totAccelBr += aBr
            totAccelBp += aBr * deltap
            totAccelF += aBr
            totAccelE += aBr
                        
        # Radial Drag
        dragR, dragC, dragBp = 0., 0., 0.
        if not dragOff:
            CMEarea = 4*c*bp 
            dragR  =  -Cd * CMEarea * rhoSWn * (vFront-vSW) * np.abs(vFront-vSW)  / Mass
            # C dir Drag
            CMEarea2 = 2*bp*(a+br)
            dragC  = -Cd * CMEarea2 * rhoSWe * (vEdge - np.sin(AW)*vSW) * np.abs(vEdge  - np.sin(AW)*vSW)  / Mass
            # Bp dir Drag
            CMEarea3 = 2*br*lenFun(a/c)*c
            dragBp = -Cd * CMEarea3 * rhoSWn * (vexpBp - np.sin(AWp)*vSW) * np.abs(vexpBp - np.sin(AWp)*vSW) / Mass

        # Update accels to include drag forces
        totAccelF += dragR  
        totAccelBr += dragR * (vexpBr/vFront)
        totAccelBp += dragBp 
        totAccelD += dragR * (vBulk/vFront)      
        totAccelE += dragC         
        totAccelA += dragR * (vexpA/vFront)
        totAccelC += dragC * (vexpC/vEdge)
        
        # Update CME shape
        rEdge += vEdge*dt + 0.5*totAccelE*dt**2
        rFront += vFront*dt + 0.5*totAccelF*dt**2
        br += vexpBr*dt + 0.5*totAccelBr*dt**2
        bp += vexpBp*dt + 0.5*totAccelBp*dt**2
        d += vBulk*dt + 0.5*totAccelD*dt**2
        # This gives the same as calc from accels
        #c = rEdge - alpha*br
        #a = rFront - d - br    
        a += vexpA * dt + 0.5*totAccelA*dt**2
        c += vexpC * dt + 0.5*totAccelC*dt**2
    
        rCent = d + a
        
        # Update velocities
        vFront += totAccelF * dt
        vexpBr += totAccelBr * dt
        vexpBp += totAccelBp * dt
        vBulk += totAccelD * dt
        vEdge += totAccelE * dt
        #vexpC = vEdge - alpha*vexpBr
        #vexpA = vFront - vexpBr - vBulk
        vexpC += totAccelC * dt
        vexpA += totAccelA * dt
        
        # Calc shape params and new AWs
        deltax = a/c
        CMEBr = br/c
        CMEBp = bp/c
        AW = np.arctan((c+br*alpha)/d) # semi approx
        AWp = np.arctan(bp/(d+a))
        alpha = np.sqrt(1+16*deltax**2)/4/deltax
        ndotz = 1./alpha
        b = np.sqrt(br * bp)
        
        # Calc new max mag field values
        Btor = deltap * tau * B0
        Bpol = 2*deltap*B0/cnm/(deltap**2+1)
                    
        # Print things
        if rFront > printR:
            if not silent:
                lessstuff = [i/60, rFront/rsun, AW*180/pi, AWp*180/pi, deltax, CMEBr, CMEBp, vFront/1e5, vexpBr/1e5, vexpBp/1e5, Btor*1e5, Bpol*1e5, np.sqrt(Btot2)*1e5, cnm, rhoSWn/1.67e-24, br/rsun]
                outstuff = ''
                for item in lessstuff:
                    outstuff = outstuff + '{:5.3f}'.format(item) + ' '
                print (outstuff)  

            if save:
                fullstuff = [i/60./24., rFront/rsun, AW*180/pi,  AWp*180/pi, a/rsun, br/rsun, bp/rsun, c/rsun, d/rsun, vFront/1e5, vEdge/1e5, vexpA/1e5, vexpBr/1e5, vexpBp/1e5, vexpC/1e5, vBulk/1e5, rho/1.67e-24, B0*1e5, cnm, rhoSWn/1.67e-24, BSWr*1e5, BSWphi*1e5, coeff1N, coeff2N, coeff1E, coeff2E, coeff3, coeff4, coeff4sw, dragR, dragC, dragBp] 
                outstuff2 = ''
                for item in fullstuff:
                    outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' '
                f1.write(outstuff2+'\n')  
            printR += 7e10          
        i +=1
    f1.write(outstuff2+'\n')   
    f1.close()   
    t = 2*br/(vFront-vexpBr)
    v0 = vFront - vexpBr
    if not silent:
        print('Transit Time (hr):  ', i/60.)
        print('In Situ Duration (hr):  ', t/3600 + vexpBr*t/(vFront-vexpBr)/3600)  
    return fullstuff
        
if __name__ == "__main__":
        
    # Inputs ---------------------------------
    # A [425, 1, 27, 0.7, 0.5, 1, 10, 1.25, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    # B [889, 5, 40.9, 0.7, 0.5, 1, 10, 3, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    # C [1083, 9.9, 46.7, 0.7, 0.5, 1, 10, 5, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    # D [1547, 50, 60.7, 0.7, 0.5, 1, 10, 7, 6.9, 440, 5.7, 1, 213., 1, 1.927']
    #[1300, 6.9, 60, 0.8, 0.5, 1, 10, 3, 6.9, 440, 8, 1, 213., 1, 1.927]
    #[1300, 7, 60, 0.7, 0.5, 1, 10, 7, 6.9, 400, 4, 1, 213., 1, 1.555] new version with C 1.555

    #invec = [1300, 7, 60, 0.7, 0.5, 10, 7, 6.9, 400, 4, 1, 213., 1, 1.555]
    invecF = [1250, 10, 45, 12.5, 0.7, 1., 10, 3, 6.9, 440, 5.7, 1, 213., 1.1, 1.927]
    invecS = [600, 2, 30, 5, 0.7,  1, 10, 1.33, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    invecF = [1250, 10, 45, 10, 0.7, 1, 10, 3, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    invecE = [2000, 50, 60, 15, 0.7, 1, 10, 8, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    runParade(invecE)

    '''invecA = [1083, 9.9, 46.7, 0.7, 0.5, 1, 10, 3, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    invecB = [1547, 50, 60.7, 0.7, 0.5, 1, 10, 7, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    invecC = [1547, 50, 60.7, 0.7, 0.5, 1, 10, 7, 6.9, 440, 5.7, 1, 213., 1, 1.927]
    invecD = [1547, 50, 60.7, 0.7, 0.5, 1, 10, 7, 6.9, 440, 5.7, 1, 213., 1, 1.927]

    invecSold = [600, 5, 30, 0.7, 0.3, 1, 10, 1.33, 6.9, 440, 5.7, 1, 213., 1, 1.927]

    runParade(invecS, pan=True, csOff=True, axisOff=True, dragOff=True, name='SP10')
    runParade(invecS, pan=True, csOff=True, axisOff=True, dragOff=False, name='SP1D')
    runParade(invecS, pan=True, csTens=False, axisOff=True, dragOff=True, name='SP20')
    runParade(invecS, pan=True, csTens=False, axisOff=True, dragOff=False, name='SP2D')
    runParade(invecS, pan=True, axisOff=True, dragOff=True, name='SP30')
    runParade(invecS, pan=True, axisOff=True, dragOff=False, name='SP3D')
    runParade(invecS, pan=True, dragOff=True, name='SP40')
    runParade(invecS, pan=True, dragOff=False, name='SP4D')'''

    #runParade(invecB)
    #runParade(invecC)
    #runParade(invecD)