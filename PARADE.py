import numpy as np

rsun = 7e10
pi = 3.14159

# Empirical function to calculate length of axis
global lenFun
lenCoeffs = [0.61618, 0.47539022, 1.95157615]
lenFun = np.poly1d(lenCoeffs)
# calc as lenFun(a/c)

# Empirical functions to calculate hoop force coefficient
g1Coeffs = [ 2.32974897, 13.57489783, 28.39890803, 26.50096997, 15.37622926, -3.01860085]
g2Coeffs = [ 2.94681341, 17.03267855, 35.68191889, 33.73752841, 16.91187526,  0.29796235]
global g1, g2
g1 = np.poly1d(g1Coeffs)
g2 = np.poly1d(g2Coeffs)
#Calc hoop coeffs as -2.405*np.exp(g1(np.log(b/Rc))) + 2*np.exp(g2(np.log(b/Rc)))



def runParade(invec):
    # invec = [0: vCME (km/s), 1: mass (1e15 g), 2: AW (deg), 3: shape A, 4: shape B, 5: rFront (rsun), 6: Btor/Bswr  7: nSW (cm^-3), 8: vSW (km/s), 9: BSW (nT), 10: Cd, 11: rFinal (rsun)]
    # Solar wind values all at rFinal
    vFront = invec[0] * 1e5
    Mass   = invec[1] * 1e15
    CMEAW  = invec[2] * pi / 180.
    CMEA   = invec[3]
    CMEBr  = invec[4]
    CMEBp  = CMEBr
    rFront = invec[5] * rsun
    Bscale = invec[6]
    n1AU   = invec[7]
    vSW    = invec[8] * 1e5
    B1AU   = invec[9] # may not actually be 1AU if rFinal isn't
    Cd     = invec[10]
    rFinal = invec[11] *rsun
    
    # Open a file to save output
    fname = 'PARADE_'+invec[12]+'.dat'
    f1 = open(fname, 'w')
    
    # B1AU is the total magnetic field strength at rFinal
    # need to split into Br and Bphi
    BphiBr = 2.7e-6 * rFinal  / vSW 
    Br1AU = B1AU / np.sqrt(1+BphiBr**2)
    # Br at the CME front at the beginning of run
    Bsw0 = Br1AU * (rFinal/rFront)**2 / 1e5
    
    t = 0.
    dt = 60. # in seconds

    # Save the initial R and CME B values
    rFront0 = rFront
    Btor0 = Bscale * Bsw0
    # Assume a Lundquist force free flux to start
    Bpol0 = 0.519 * Btor0

    # CME shape
    # alpha * br is the cross sec width in z dir
    alpha = np.sqrt(1+16*CMEA**2)/4/CMEA
    # normal vector dot z
    ndotz = 1./alpha
    # Convert AW and shape params to actual lengths
    CdivR = np.tan(CMEAW) / (1. + CMEBr*alpha + np.tan(CMEAW) * (CMEA + CMEBr))
    c = CdivR * rFront
    a = CMEA * c
    br = CMEBr * c
    bp = CMEBp * c
    # average b is geometric mean of br bp
    b = np.sqrt(br*bp)
    d = rFront - a - br
    rCent = d+a
    rEdge = c + br*alpha
    
    # Calc AW again as will later, mostly saving as new var name
    AW = np.arctan((br*alpha+c)/d)
    # AW in perp (y) direction
    AWp = np.arctan(bp/(d+a))
    
    # Initial length and area
    initlen = lenFun(a/c)*c
    CApol = initlen*br # init cross section area TO CALC pol
    CAtor = br * bp      # TO CALC tor
    
    # Calculate convective velocities - following traditional
    # pancaking method with few diff assumptions
    vexpA  = vFront * (np.cos(AWp) - np.cos(AW))
    vexpBr = vFront * (1 - np.cos(AWp))
    vexpBp = vFront * np.sin(AWp)
    
    vBulk  = vFront * np.cos(AW)
    vEdge  = vFront * np.sin(AW)
    vexpC = vEdge - alpha * vexpBr
        
    # Initial velocities = self-similar values
    #vexpC = vFront * c/rFront
    #vexpA = vexpC * CMEA
    #vexpBr = vexpC * CMEBr
    #vexpBp = vexpC * CMEBp
    #vBulk = vFront - vexpA - vexpBr

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

        # Update solar wind properties
        # Currently using values at front edge
        # might want to switch to values at avg R
        # (if so make the pre loop calcs match!)
        rhoSWn = 1.67e-24 * n1AU * (rFinal/rFront)**2
        #rEdge = np.sqrt(c**2 + d**2)
        rhoSWe = 1.67e-24 * n1AU * (rFinal/(rEdge))**2
        #SWrho = 1.67e-24 * nSW
        BSWr = Bsw0 * (rFront0 / rFront)**2
        BSWphi = BSWr * 2.7e-6 * rFront  / vSW 
    
        # Update flux rope field
        lenNow = lenFun(a/c)*c
        Btor = Btor0 * (CAtor / (br*bp))
        Bpol = Bpol0 * (CApol / (lenNow*np.sqrt(br*bp)))
    
        # Update CME density
        vol = pi*br*bp *  lenFun(a/c)*c
        rho = Mass / vol
        
        # Force at nose
        # Internal flux rope forces
        # Using avg b for these calcs - doesn't terribly matter
        # since JXB axis not really important
        kNose = 1.3726 * CMEA / c   
        #kNose = a / c**2
        RcNose = 1./kNose   
        gammaN = b / RcNose
        coeff1N = (-2.405*np.exp(g1(np.log(gammaN))) + 2*np.exp(g2(np.log(gammaN)))) / gammaN
        coeff2N = -0.1347 * gammaN
        aTotNose = (coeff1N+coeff2N) * Btor**2 / 4 / pi / rho / b
        totAccelF += aTotNose
                   
        # Force at flank
        # Internal flux rope forces
        kEdge = 40 * CMEA / c / np.sqrt(16*CMEA+1)**3
        #kEdge = c / a**2
        RcEdge = 1./kEdge    
        gammaE = b / RcEdge
        coeff1E = (-2.405*np.exp(g1(np.log(gammaE))) + 2*np.exp(g2(np.log(gammaE)))) / gammaE
        coeff2E = -0.1347 * gammaE
        aTotEdge = (coeff1E+coeff2E) * Btor**2 / 4 / pi / rho / b 
        totAccelE += aTotEdge
        
        # Internal cross section forces
        Btot2 = BSWr**2 + BSWphi**2
        coeff3 = - 0.4080
        coeff4 = 0.4733 - 0.5 * (Btot2/Btor**2)
        aTotCS = (coeff3+coeff4) * Btor**2 / 4 / pi / rho  / b
        epscs = (br / bp)
        aTotCSr = (coeff3*epscs+coeff4/np.sqrt(epscs))* Btor**2 / 4 / pi / rho /b
        aTotCSp = (coeff3/epscs+coeff4*np.sqrt(epscs)) * Btor**2 / 4 / pi / rho  /b
        aTotCS = (coeff3+coeff4) * Btor**2 / 4 / pi / rho / b 
        totAccelBr += aTotCSr
        totAccelBp += aTotCSp 

        # Radial Drag
        CMEarea = 4*c*br 
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

        # Update CME shape
        rEdge += vEdge*dt + 0.5*totAccelE*dt**2
        rFront += vFront*dt + 0.5*totAccelF*dt**2
        br += vexpBr*dt + 0.5*totAccelBr*dt**2
        bp += vexpBp*dt + 0.5*totAccelBp*dt**2
        d += vBulk*dt + 0.5*totAccelD*dt**2
        c = rEdge - alpha*br
        a = rFront - d - br        
        rCent = d + a
        # Calc shape params and new AWs
        CMEA = a/c
        CMEBr = br/c
        CMEBp = bp/c
        AW = np.arctan((c+br*alpha)/d) # semi approx
        AWp = np.arctan(bp/(d+a))
        alpha = np.sqrt(1+16*CMEA**2)/4/CMEA
        ndotz = 1./alpha
        b = np.sqrt(br * bp)
        
        # Update velocities
        vFront += totAccelF * dt
        vexpBr += totAccelBr * dt
        vexpBp += totAccelBp * dt
        vBulk += totAccelD * dt
        #vFront = vBulk + vexpA + vexpBr
        vEdge += totAccelE * dt
        #vexpC += totAccelC  * dt
        vexpC = vEdge - alpha*vexpBr
        vexpA = vFront - vexpBr - vBulk
    
        # Print things
        if rFront > printR:
            lessstuff = [i/60, rFront/rsun, AW*180/pi, AWp*180/pi, CMEA, CMEBr, CMEBp, vFront/1e5, vexpA/1e5, vexpBr/1e5, vBulk/1e5, vEdge/1e5, br/rsun, np.sqrt(Btor**2+Bpol**2)*1e5, CMEBr/CMEBp]  
            fullstuff = [i/60./24., rFront/rsun, AW*180/pi,  AWp*180/pi, a/rsun, br/rsun, bp/rsun, c/rsun, d/rsun, vFront/1e5, vEdge/1e5, vexpA/1e5, vexpBr/1e5, vexpBp/1e5, vexpC/1e5, vBulk/1e5, rho/1.67e-24, Btor*1e5, Bpol*1e5, rhoSWn/1.67e-24, BSWr*1e5, BSWphi*1e5, coeff1N, coeff2N, coeff1E, coeff2E, coeff3, coeff4, dragR, dragC, dragBp] 
            outstuff = ''
            for item in lessstuff:
                outstuff = outstuff + '{:8.5f}'.format(item) + ' '
            print (outstuff)
            outstuff2 = ''
            for item in fullstuff:
                outstuff2 = outstuff2 + '{:8.5f}'.format(item) + ' '
            f1.write(outstuff2+'\n')  
            printR += 7e10          
        i +=1
    f1.write(outstuff2+'\n')  
    f1.close()      
        
    
# Inputs ---------------------------------
# A [425, 1, 27, 0.7, 0.5, 10, 1.5, 6.9, 440, 5.7, 1, 213., 'A']
# B [889, 5, 40.9, 0.7, 0.5, 10, 3, 6.9, 440, 5.7, 1, 213., 'B']
# C [1083, 9.9, 46.7, 0.7, 0.5, 10, 5, 6.9, 440, 5.7, 1, 213., 'C']
# D [1547, 50, 60.7, 0.7, 0.5, 10, 7, 6.9, 440, 5.7, 1, 213., 'D']
#[1300, 6.9, 60, 0.8, 0.5, 10, 3, 6.9, 440, 8, 1, 213.,  'temp']

invecA =[1300, 6.9, 60, 0.8, 0.5, 10, 3, 6.9, 440, 8, 1, 213,  'temp']
invecB = [1547, 50, 60.7, 0.7, 0.5, 10, 7, 6.9, 440, 5.7, 1, 213., 'D10b']
invecC = [1547, 50, 60.7, 0.7, 0.5, 10, 7, 6.9, 440, 5.7, 1, 213., 'D10c']
invecD = [1547, 50, 60.7, 0.7, 0.5, 10, 7, 6.9, 440, 5.7, 1, 213., 'D10d']

runParade(invecA)
#runParade(invecB)
#runParade(invecC)
#runParade(invecD)