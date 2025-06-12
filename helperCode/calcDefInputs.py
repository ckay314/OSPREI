import numpy as np

def getProps(rfront, v, AW, AWp, delAx, delCS, Cnm=1.927, tau=1, kappa=0):
    dtor = 3.14159 / 180
    rFront = rfront * 7e10
    
    # check if given GCS kappa value, if so recalc AW to get
    # the AW at angle beta=0 from Thernisien paper
    if kappa != 0:
        sinAlpha = np.sin(AW*dtor)
        # Calculate face on (half) width at beta = 0 for GCS shape
        AW  = np.arctan( (sinAlpha + kappa * np.sqrt(1 - kappa**2 + sinAlpha**2)) / (1-kappa**2)) / dtor
        AWp = AW / 3.
    
    AW = AW * dtor
    AWp = AWp *dtor
    
    
    lenCoeffs = [0.61618, 0.47539022, 1.95157615]
    lenFun = np.poly1d(lenCoeffs)
    
    # need CME R and len to convert phiflux to B0
    rCSp = np.tan(AWp) / (1 + delCS * np.tan(AWp)) * rFront
    rCSr = delCS * rCSp
    Lp = (np.tan(AW) * (rFront - rCSr) - rCSr) / (1 + delAx * np.tan(AW))  
    Ltorus = lenFun(delAx) * Lp
    # Ltorus too short, need to include legs?
    rCent = rFront - delAx*Lp - rCSp
    Lleg = np.sqrt(Lp**2 + rCent**2) - 7e10 # dist from surface
    Ltot = Ltorus + 2 * Lleg 
    
    avgR = (0.5*rCSp * Lleg * 2 + rCSp * Ltorus) / (Lleg*2 + Ltorus)
    
    # Old version    
    #mass = 0.005756*v -0.84
    # LLAMAICE 1.0 relation
    mass = 0.010 * v + 0.16
    #mass = mass * 2
    
    
    # Karin
    #phiflux = (0.2082 * mass +2.244) * 1e21
    # Gopal
    KE = 0.5 * mass*1e15 * (v*1e5)**2 /1e31
    phiflux = np.power(10, np.log10(KE / 0.19) / 1.87)*1e21
    B0 = phiflux * Cnm * (delCS**2 + 1) / avgR / Ltot / delCS**2 *1e5
    Bcent = delCS * tau * B0
    
    vSheath = 0.129 * v + 376
    vIS = (vSheath + 51.73) / 1.175
    vExp = 0.175 * vIS -51.73
    logTIS = 3.07e-3 * vIS +3.65
    TIP = np.power(10, logTIS) * np.power(215*7e10/rFront, 0.7)
    
    return AW/dtor, AWp/dtor, mass, TIP, Bcent
    
#outs = getProps(21.5, 677, 60, 20, 0.75, 1)
outs = getProps(21.5, 1613, 41, 14, 0.75, 1)
print (outs)
