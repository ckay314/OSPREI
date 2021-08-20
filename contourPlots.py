import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
from scipy.interpolate import CubicSpline, interp2d
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable



global dtor
dtor = math.pi / 180.


# make label text size bigger
plt.rcParams.update({'font.size':16})
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Set up the path variable
mainpath = '/Users/ckay/OSPREI/' #MTMYS

# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
sys.path.append(os.path.abspath(mainpath+'codes/')) #MTMYS
import OSPREI  as OSP
from ForeCAT_functions import rotx, roty, rotz, SPH2CART, CART2SPH
from CME_class import cart2cart
from PARADE import lenFun
from processOSPREI import txt2obj, readInData
import FIDO

# things in the mega-array
# 0, 1       - in/out, padded in/out (better for rotations)
# 2, 3       - thetaT, thetaP
# 4, 5, 6    - nAx x, y, z
# 7, 8, 9    - nCS x, y, z
# 10, 11, 12 - tAx x, y, z
# 13, 14, 15 - tCS x, y, z
# 16, 17     - vAx, vCS
# 18, 19, 20 - vF x, y, z
# 21, 22, 23 - vAx + vBulk x, y, z
# 24, 25     - vF dot rhat, vAB dot rhat
# 26         - duration
# 27, 28, 29 - Btor x, y, z
# 30, 31, 32 - Bpol x, y, z
# 33, 34     - Btor dot zhat, Bpol dot zhat
# 35, 36     - Kp front, center
# 37, 38     - T, n


def makePostFCdata(ResArr, calcwid=90, plotwid=40):
    print ('not done yet')
    
def makePostANTdata(ResArr, calcwid=90, plotwid=40):
    # Start by filling in the area that corresponds to the CME in a convenient frame
    # then rotate it the frame where Earth is at [0,0] at the time of impact for all CMEs
    # using its exact position in "real" space (which will vary slightly between CMEs).
    # Also derive parameters based on Earth's coord system if there were changes in it's
    # position (i.e. vr for rhat corresponding to each potential shifted lat/lon)
            
    # simulation parameters
    ngrid = 2*plotwid+1
    ncalc = 2*calcwid+1
    nThings = 39
    shiftx, shifty = calcwid, calcwid
    counter = 0
    
    # get impacts, may be less than nEns
    hits = []
    for i in range(nEns):
        if (not ResArr[i].miss) and (not ResArr[i].fail):
            hits.append(i)
    allGrid = np.zeros([len(hits), ngrid, ngrid, nThings])
    
    for key in hits:
        newGrid = np.zeros([ncalc,ncalc,nThings])
        
        # pull in things from ResArr
        thisLat = ResArr[key].FClats[-1]
        thisLon = ResArr[key].FClons[-1]
        thisTilt  = ResArr[key].FCtilts[-1]
        thisAW    = ResArr[key].ANTAWs[-1]
        thisAWp   = ResArr[key].ANTAWps[-1]
        thisR     = ResArr[key].ANTrs[-1]
        thisDelAx = ResArr[key].ANTdelAxs[-1]
        thisDelCS = ResArr[key].ANTdelCSs[-1]
        thisDelCA = ResArr[key].ANTdelCSAxs[-1]
        thesevs = np.array([ResArr[key].ANTvFs[-1], ResArr[key].ANTvEs[-1], ResArr[key].ANTvBs[-1], ResArr[key].ANTvCSrs[-1], ResArr[key].ANTvCSps[-1], ResArr[key].ANTvAxrs[-1], ResArr[key].ANTvAxps[-1]])             
        thisB0    = ResArr[key].ANTB0s[-1] * np.sign(float(OSP.input_values['FRBscale']))
        thisTau   = ResArr[key].ANTtaus[-1]
        thisCnm   = ResArr[key].ANTCnms[-1]
        thisPol   = int(float(OSP.input_values['FRpol']))
        thislogT  = ResArr[key].ANTlogTs[-1]
        thisn     = ResArr[key].ANTns[-1]
        #thisTilt = 90
        # Calculate the CME lengths 
        #CMElens = [CMEnose, rEdge, d, br, bp, a, c]
        CMElens = np.zeros(7)
        CMElens[0] = thisR
        CMElens[4] = np.tan(thisAWp*dtor) / (1 + thisDelCS * np.tan(thisAWp*dtor)) * CMElens[0]
        CMElens[3] = thisDelCS * CMElens[4]
        CMElens[6] = (np.tan(thisAW*dtor) * (CMElens[0] - CMElens[3]) - CMElens[3]) / (1 + thisDelAx * np.tan(thisAW*dtor))  
        CMElens[5] = thisDelAx * CMElens[6]
        CMElens[2] = CMElens[0] - CMElens[3] - CMElens[5]
        CMElens[1] = CMElens[2] * np.tan(thisAW*dtor)
       
       
        # Find the location of the axis
        nFR = 31 # axis resolution
        thetas = np.linspace(-math.pi/2, math.pi/2, nFR)    
        sns = np.sign(thetas)
        xFR = CMElens[2] + thisDelAx * CMElens[6] * np.cos(thetas)
        zFR = 0.5 * sns * CMElens[6] * (np.sin(np.abs(thetas)) + np.sqrt(1 - np.cos(np.abs(thetas))))   
        top = [xFR[-1], 0.,  zFR[-1]+CMElens[3]]
        topSPH = CART2SPH(top)
        # br is constant but axis x changes -> AWp varies
        varyAWp = np.arctan(CMElens[4] / xFR)/dtor 
        # get the axis coords in spherical
        axSPH = CART2SPH([xFR,0.,zFR])
        # get the round outer edge from the CS at last thetaT
        thetaPs = np.linspace(0, math.pi/2,21)
        xEdge = xFR[-1] 
        yEdge = CMElens[4] * np.sin(thetaPs)
        zEdge = zFR[-1] + (thisDelCS * CMElens[4] * np.cos(thetaPs))
        edgeSPH = CART2SPH([xEdge, yEdge, zEdge])
        #for i in range(21):
        #    print (thetaPs[i]/dtor, edgeSPH[1][i], edgeSPH[2][i])
                    
        temp = CMElens[6] / thisDelCS / CMElens[4]
        tts = np.linspace(91,150,60)*dtor
        toZero = temp * np.sqrt((2*np.cos(tts)+np.sqrt(1+np.cos(tts)))**2 + 16*thisDelAx**2*np.sin(tts)**2) - 4 * np.tan(tts)
        
        # figure out mappings between the latitude and other variables
        lat2theT =  CubicSpline(axSPH[1],thetas/dtor,bc_type='natural')
        lat2AWp = CubicSpline(axSPH[1],varyAWp,bc_type='natural')
        lat2xFR = CubicSpline(axSPH[1],xFR,bc_type='natural')
        minlat, maxlat = int(round(np.min(axSPH[1]))), int(round(np.max(axSPH[1])))
        minlon, maxlon = shiftx-int(round(thisAWp)),shiftx+int(round(thisAWp))
        lat2AWpEd = CubicSpline(edgeSPH[1][::-1],edgeSPH[2][::-1],bc_type='natural')    
        lon2TP = CubicSpline(edgeSPH[2],thetaPs/dtor,bc_type='natural') 
        minlat2, maxlat2 = int(round(maxlat)), int(round(topSPH[1]))        
        
        # Loop through in latitude and fill in points that are in the CME
        # SHOULD ADD A DOUBLE CHECK ON LIMITS OF RANGE OF CME VS PLOT
        for i in range(minlat, maxlat+1):
            # Find the range to fill in lon
            idy =  i+shifty
            nowAWp = np.round(lat2AWp(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            # Fill from minlon to maxlon
            newGrid[idy,  minlon:maxlon+1,0] = 1
           
            # Pad things 2 deg outside "correct" lon range
            newGrid[idy,  minlon-2:maxlon+3,1] = 1
            # Pad around the top and bottom of the CME
            '''if i == minlat: 
               newGrid[idy-1,  minlon-2:maxlon+3,1] = 1
               newGrid[idy-2,  minlon-2:maxlon+3,1] = 1
            if i == maxlat: 
               newGrid[idy+1,  minlon-2:maxlon+3,1] = 1
               newGrid[idy+2,  minlon-2:maxlon+3,1] = 1'''
               
            # Calculate the parameteric thetas for each point in the CME
            # ThetaT - start with relation to lat
            thetaT = lat2theT(i)
            newGrid[idy,  minlon:maxlon+1,2] = thetaT 
            
            # ThetaP - making geometric approx to calc sinThetaP = axis_X tanLon / rperp
            # which ignores that this perp width is not at axis but axis + xCS (which is f(thetaP))
            # (CME is oriented so AWp is in lon direction before we rotate it)
            theselons = np.arange(-nowAWp, nowAWp+1)
            sinTP = lat2xFR(i) * np.tan(theselons*dtor) / CMElens[4]
            # Clean up any places with sinTP > 1 from our geo approx so it just maxes out not blows up
            sinTP[np.where(np.abs(sinTP) > 1)] = np.sign(sinTP[np.where(np.abs(sinTP) > 1)]) * 1
            thetaPs = np.arcsin(sinTP)/dtor
            newGrid[idy,  minlon:maxlon+1,3] = thetaPs
        
        for i in range(minlat2, maxlat2):
            # Find the range to fill in lon
            idy  =  -i+shifty
            idy2 =  i+shifty
            nowAWp = np.round(lat2AWpEd(i))
            minlon, maxlon = shiftx-int(nowAWp),shiftx+int(nowAWp)
            # Fill from minlon to maxlon
            newGrid[idy,  minlon:maxlon+1,0] = 1
            newGrid[idy2,  minlon:maxlon+1,0] = 1
            
            # Pad things 2 deg outside "correct" lon range
            newGrid[idy,  minlon-4:maxlon+5,1] = 1
            newGrid[idy2,  minlon-4:maxlon+5,1] = 1
           
            # Pad around the top and bottom of the CME
            if i == maxlat2-1: 
               newGrid[idy-1,  minlon-2:maxlon+3,1] = 1
               newGrid[idy-2,  minlon-2:maxlon+3,1] = 1
               newGrid[idy2+1,  minlon-2:maxlon+3,1] = 1
               newGrid[idy2+2,  minlon:maxlon+1,1] = 1
            
            # add angle things
            # ThetaT
            newGrid[idy,  minlon:maxlon+1,2] = -90
            newGrid[idy2,  minlon:maxlon+1,2] = 90
            # ThetaP
            theseLons = np.array(range(minlon,maxlon+1))
            newGrid[idy,  minlon:maxlon+1,3] = lon2TP(np.abs(theseLons-90))*np.sign(theseLons-90)
            newGrid[idy2,  minlon:maxlon+1,3] = lon2TP(np.abs(theseLons-90))*np.sign(theseLons-90)
            
        
             
        # Clean up  angles so sin or cos don't blow up
        fullTT = newGrid[:,:,2]
        fullTT[np.where(np.abs(fullTT)<1e-5)] = 1e-5
        fullTT[np.where(np.abs(fullTT)>0.9*90)] = np.sign(fullTT[np.where(np.abs(fullTT)>.9*90)]) * 0.9 * 90
        newGrid[:,:,2] = fullTT * dtor
        fullTP = newGrid[:,:,3]
        fullTP[np.where(np.abs(fullTP)<1e-5)] = np.sign(fullTP[np.where(np.abs(fullTP)<1e-5)]) * 1e-5
        fullTP[np.where(np.abs(fullTP)>0.9*90)] = np.sign(fullTP[np.where(np.abs(fullTP)>0.9*90)]) * 0.9 * 90
        newGrid[:,:,3] = fullTP * dtor
        
        '''fig = plt.figure()
        plt.contourf(newGrid[:,:,3])
        plt.show()
        print (sd)'''
            
        # get rid of other theta cleaning?
        
        # Vectors ---------------------------------------------------------------------------------
        # normal to the axis
        # turn off error from div by zero outside CME range
        np.seterr(divide='ignore', invalid='ignore')
        nAxX = 0.5 * (np.cos(np.abs(newGrid[:,:,2])) + np.sin(np.abs(newGrid[:,:,2])) / np.sqrt(1 - np.cos(np.abs(newGrid[:,:,2]))))
        nAxZ = thisDelAx * np.sin(newGrid[:,:,2])
        nAxMag = np.sqrt(nAxX**2 + nAxZ**2) # nAxY is zero in this frame
        newGrid[:,:,4] = nAxX / nAxMag *newGrid[:,:,0]
        newGrid[:,:,6] = nAxZ / nAxMag *newGrid[:,:,0]
        mag = np.sqrt(newGrid[:,:,4]**2 + newGrid[:,:,5]**2 + newGrid[:,:,6]**2)    
                
        # normal to the CS
        nCSx0 = np.cos(newGrid[:,:,3]) / thisDelCS
        nCSy = np.sin(newGrid[:,:,3])
        nCSx = nCSx0 * np.cos(newGrid[:,:,2])
        nCSz = nCSx0 * np.sin(newGrid[:,:,2])
        nCSmag = np.sqrt(nCSx**2 + nCSy**2 + nCSz**2)
        newGrid[:,:,7] = nCSx / nCSmag
        newGrid[:,:,8] = nCSy / nCSmag
        newGrid[:,:,9] = nCSz / nCSmag
        mag = np.sqrt(newGrid[:,:,7]**2 + newGrid[:,:,8]**2 + newGrid[:,:,9]**2)    
        
        # tangent to the axis
        # can just swap normal components since axis is in xz plan
        newGrid[:,:,10], newGrid[:,:,12] = -newGrid[:,:,6], newGrid[:,:,4]
        
        # tangent to the CS
        tCSx = nCSy * np.cos(newGrid[:,:,2])
        tCSz = -nCSy * np.sin(newGrid[:,:,2])
        tCSy = np.cos(newGrid[:,:,3]) / thisDelCS
        tCSmag = np.sqrt(tCSx**2 + tCSy**2 + tCSz**2)
        newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15] = -tCSx / tCSmag,  tCSy / tCSmag,  tCSz / tCSmag
        mag = np.sqrt(newGrid[:,:,14]**2 + newGrid[:,:,15]**2 + newGrid[:,:,13]**2)    
        
        
        # Coordinate transformations -------------------------------------------------------------
        # Calculate how far to shift the CME in lat/lon to put Earth at 0,0
        dLon = OSP.satPos[1] + ResArr[key].ANTtimes[-1] * 24 * 3600 * OSP.Sat_rot - thisLon
        dLat = OSP.satPos[0] - thisLat
                
        # Create the background meshgrid and shift it
        XX, YY = np.meshgrid(range(-shiftx,shiftx+1),range(-shifty,shifty+1))   
        XX = XX.astype(float) - dLon
        YY = YY.astype(float) - dLat

        # Rotate newGrid array based on CME tilt
        newGrid = ndimage.rotate(newGrid, 90-thisTilt, reshape=False)
        # Force in/out to be 0/1 again
        newGrid[:,:,0] = np.rint(newGrid[:,:,0])
        newGrid[:,:,1] = np.rint(newGrid[:,:,1])
        
        # Rotate the actual components of the normal and tangent vectors to correct lat/lon/tilt 
        # before using them to define velocity or magnetic vectors
        # normal to axis
        rotNAx =  cart2cart([newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6]], dLat, dLon, thisTilt)
        newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6] = rotNAx[0], rotNAx[1], rotNAx[2]
        # normal to CS
        rotNCS =  cart2cart([newGrid[:,:,7], newGrid[:,:,8], newGrid[:,:,9]], dLat, dLon, thisTilt)
        newGrid[:,:,7], newGrid[:,:,8], newGrid[:,:,9] = rotNCS[0], rotNCS[1], rotNCS[2]
        # tangent to axis
        rotTAx =  cart2cart([newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12]], dLat, dLon, thisTilt)
        newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12] = rotTAx[0], rotTAx[1], rotTAx[2]
        # tangent to CS
        rotTCS =  cart2cart([newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15]], dLat, dLon, thisTilt)
        newGrid[:,:,13], newGrid[:,:,14], newGrid[:,:,15] = rotTCS[0], rotTCS[1], rotTCS[2]
        
                
        # Velocity ----------------------------------------------------------------------------------
        # local vAx
        newGrid[:,:,16] = np.sqrt(thisDelAx**2 * np.cos(newGrid[:,:,2])**2 + np.sin(newGrid[:,:,2])**2) * thesevs[6]
        # local vCS
        newGrid[:,:,17] = np.sqrt(thisDelCS**2 * np.cos(newGrid[:,:,3])**2 + np.sin(newGrid[:,:,3])**2) * thesevs[4]
                
        # vBulk in direction of CME nose (which is at -dLat, -dLon)
        rhatCME = np.array([np.sin((90+dLat)*dtor) * np.cos(-dLon*dtor), np.sin((90+dLat)*dtor) * np.sin(-dLon*dtor), np.cos((90+dLat)*dtor)])
        vBulk = thesevs[2] * rhatCME
        
        # vFront = vBulk * rhatCME + vAx * nAx + vExp * nCS
        newGrid[:,:,18] = vBulk[0] + newGrid[:,:,4] * newGrid[:,:,16] + newGrid[:,:,7] * newGrid[:,:,17]
        newGrid[:,:,19] = vBulk[1] + newGrid[:,:,5] * newGrid[:,:,16] + newGrid[:,:,8] * newGrid[:,:,17]
        newGrid[:,:,20] = vBulk[2] + newGrid[:,:,6] * newGrid[:,:,16] + newGrid[:,:,9] * newGrid[:,:,17]
        
        # vAx = vBulk * rhatCME + vAx * nAx
        newGrid[:,:,21] = vBulk[0] + newGrid[:,:,4] * newGrid[:,:,16] 
        newGrid[:,:,22] = vBulk[1] + newGrid[:,:,5] * newGrid[:,:,16] 
        newGrid[:,:,23] = vBulk[2] + newGrid[:,:,6] * newGrid[:,:,16]
                
        # local radial unit vector if earth was at a given lat/lon instead of 0,0
        rhatE = np.zeros([ncalc,ncalc,3])
        colat = (90 - YY) * dtor
        rhatE[:,:,0] = np.sin(colat) * np.cos(XX*dtor) 
        rhatE[:,:,1] = np.sin(colat) * np.sin(XX*dtor)
        rhatE[:,:,2] = np.cos(colat)
        
        # take dot product of velocities with local rhat to get magnitude
        newGrid[:,:,24] = rhatE[:,:,0] * newGrid[:,:,18] + rhatE[:,:,1] * newGrid[:,:,19] + rhatE[:,:,2] * newGrid[:,:,20]
        newGrid[:,:,25] = rhatE[:,:,0] * newGrid[:,:,21] + rhatE[:,:,1] * newGrid[:,:,22] + rhatE[:,:,2] * newGrid[:,:,23]
        
        # local z unit vector if earth was at a given lat/lon instead of 0,0 (same as colat vec)
        zhatE = np.zeros([ncalc,ncalc,3])
        zhatE[:,:,0] = -np.abs(np.cos(colat)) * np.cos(XX*dtor)
        zhatE[:,:,1] = -np.cos(colat) * np.sin(XX*dtor)
        zhatE[:,:,2] = np.sin(colat)
        
        # local y unit vector, will need for Kp clock angle
        yhatE = np.zeros([ncalc,ncalc,3])
        yhatE[:,:,0] = -np.sin(XX*dtor)
        yhatE[:,:,1] = np.cos(XX*dtor)
        
        
        # Duration --------------------------------------------------------------
        br = CMElens[3]
        bCS = 2 * np.cos(newGrid[:,:,3]) * br
        # need to rotate rhatE about axis tangent by actual CS pol ang (not parametric ang)
        realTP = np.arctan(np.tan(newGrid[:,:,3])/thisDelCS)
        # Pull axis vectors into convenient names
        tax, tay, taz    = newGrid[:,:,10], newGrid[:,:,11], newGrid[:,:,12] 
        n1ax, n1ay, n1az = newGrid[:,:,4], newGrid[:,:,5], newGrid[:,:,6]
        # Calculate other normal from cross product
        nax, nay, naz = n1ay * taz - n1az * tay, n1az * tax - n1ax * taz, n1ax * tay - n1ay * taz
        rdotn = rhatE[:,:,0] * nax + rhatE[:,:,1] * nay + rhatE[:,:,2] * naz 
        # Get the rhat with the part in the second normal direction removed so we can calc rotation angle
        rhatE2 =  np.zeros([ncalc,ncalc,3])
        rhatE2[:,:,0], rhatE2[:,:,1], rhatE2[:,:,2] = rhatE[:,:,0] - rdotn * nax, rhatE[:,:,1] - rdotn * nay, rhatE[:,:,2] - rdotn * naz
        rhatE2mag = np.sqrt(rhatE2[:,:,0]**2 + rhatE2[:,:,1]**2 + rhatE2[:,:,2]**2)
        rhatE2[:,:,0], rhatE2[:,:,1], rhatE2[:,:,2] = rhatE2[:,:,0] / rhatE2mag, rhatE2[:,:,1] / rhatE2mag, rhatE2[:,:,2] / rhatE2mag
        rdotnAx = rhatE2[:,:,0] * n1ax + rhatE2[:,:,1] * n1ay + rhatE2[:,:,2] * n1az
        # Stop from blowing up too much for oblique angles
        rdotnAx[np.where(rdotnAx < 0.01)] = 0.01
        wid = bCS / rdotnAx * newGrid[:,:,0]
        # Find the time it takes to cross b (half duration with no expansion)
        halfDurEst = 0.5 * wid * 7e5 / newGrid[:,:,24] / 3600.
        # Estimate amount CME grows in that time
        newbr = bCS + newGrid[:,:,17] * halfDurEst *3600/ 7e5 
        newbr = newbr * newGrid[:,:,0]
        growth = newbr / bCS
        growth[np.where(growth<1)] = 1
        # Take average b as original plus growth from half first transit time
        newGrid[:,:,26] = 2 * growth * halfDurEst 
        
        # Magnetic field ----------------------------------------------------------------
        # Toroidal field
        BtCoeff = thisDelCS * thisB0 * thisTau
        newGrid[:,:,27], newGrid[:,:,28], newGrid[:,:,29] = BtCoeff * newGrid[:,:,10], BtCoeff * newGrid[:,:,11], BtCoeff * newGrid[:,:,12]
        
        # Poloidal field
        BpCoeff = thisPol * 2 * thisDelCS / (1 + thisDelCS**2) * np.sqrt(thisDelCS**2 * np.sin(newGrid[:,:,3]*dtor)**2 + np.cos(newGrid[:,:,3]*dtor)**2) * np.abs(thisB0) / thisCnm
        newGrid[:,:,30], newGrid[:,:,31], newGrid[:,:,32] = BpCoeff * newGrid[:,:,13], BpCoeff * newGrid[:,:,14], BpCoeff * newGrid[:,:,15]
        
        # Local z hat components
        # Toroidal
        tempZ = zhatE[:,:,0] * newGrid[:,:,27] + zhatE[:,:,1] * newGrid[:,:,28] + zhatE[:,:,2] * newGrid[:,:,29]
        tempZ[np.isnan(tempZ)] = 0
        newGrid[:,:,33] = tempZ
        # Poloidal
        newGrid[:,:,34] = zhatE[:,:,0] * newGrid[:,:,30] + zhatE[:,:,1] * newGrid[:,:,31] + zhatE[:,:,2] * newGrid[:,:,32]
        
        
        # Kp ----------------------------------------------------------------------------
        # Start at the front (from Bpol)
        # Need clock ang in local frame -> need By 
        By = yhatE[:,:,0] * newGrid[:,:,30] + yhatE[:,:,1] * newGrid[:,:,31] # yhat has no z comp
        clockAng = np.arctan2(By, newGrid[:,:,34])
        Bperp = np.sqrt(By**2 + newGrid[:,:,34]**2)
        # Calculate Kp with the usual algorithm
        dphidt = np.power(newGrid[:,:,24] , 4/3.) * np.power(Bperp, 2./3.) *np.power(np.abs(np.sin(clockAng/2)), 8/3.)
        newGrid[:,:,35] = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
        # Repeat process for the center
        # Need to scale Bz, Btor decreases proportional to CS area, simplify to growth^2
        By = yhatE[:,:,0] * newGrid[:,:,27] + yhatE[:,:,1] * newGrid[:,:,28] # yhat has no z comp
        # clockAng is measured clockwise from 12 o'clock/North
        clockAng = np.arctan2(By, newGrid[:,:,33])*newGrid[:,:,0]        
        Bperp = np.sqrt(By**2 + newGrid[:,:,33]**2) / growth**2
        dphidt = np.power(newGrid[:,:,25] , 4/3.) * np.power(Bperp, 2./3.) *np.power( np.abs(np.sin(clockAng/2)), 8/3.)
        newGrid[:,:,36] = 9.5 - np.exp(2.17676 - 5.2001e-5*dphidt)
        # Adjust Btor dot z for expansion
        newGrid[:,:,33] = newGrid[:,:,33] / growth**2 * newGrid[:,:,0] 
        
        # Temperature and density -------------------------------------------------------
        newGrid[:,:,37] = thislogT * newGrid[:,:,1] # uniform temp within CME 
        newGrid[:,:,38] =    thisn * newGrid[:,:,1] # uniform dens within CME 
        
              
        # Interpolate and cut out window around Earth/sat to use for summing and plotting
        # Integer shift
        delxi, delyi = -shiftx-int(XX[0,0]), -shifty-int(YY[0,0])
        # Remainder from integer
        delx, dely = int(XX[0,0]) - XX[0,0], int(YY[0,0]) - YY[0,0]
        # Perform shift in x
        startidx = shiftx - plotwid + delxi
        leftX = newGrid[:,startidx:startidx+2*plotwid+1,:]
        rightX = newGrid[:,startidx+1:startidx+2*plotwid+2,:]
        subGrid = (1-delx) * leftX + delx * rightX
        # Perform shift in y
        startidy = shifty - plotwid + delyi
        botY = subGrid[startidy:startidy+2*plotwid+1, :]
        topY = subGrid[startidy+1:startidy+2*plotwid+2, :]
        subGrid = (1-dely) * botY + dely * topY
        # Clean up in/out smearing
        subGrid[:,:,0] = np.rint(subGrid[:,:,0])
        subGrid[:,:,1] = np.rint(subGrid[:,:,1])
        for i in range(nThings):
            subGrid[:,:,i] = subGrid[:,:,i] * subGrid[:,:,1]
        allGrid[counter,:,:,:] = subGrid
        counter += 1
    
        # center at 90+delyi,90+delxi for newGrid               
        
    return allGrid


def plotAll(allGrid, plotwid=40, pname='temp.png'):
    fig, axes = plt.subplots(2, 5, figsize=(10,5))
    cmap1 = cm.get_cmap("plasma",lut=10)
    cmap1.set_bad("k")
    # Reorder Axes
    axes = [axes[0,0], axes[0,1], axes[0,2], axes[0,3], axes[0,4], axes[1,0], axes[1,1], axes[1,2], axes[1,3], axes[1,4]]
    labels = ['Chance of Impact (%)', 'B$_z$ Front (nT)', 'v$_r$ Front (km/s)',  'Kp Front', 'n (cm$^{-1}$)', 'Duration (hr)', 'B$_z$ Center (nT)', 'v$_r$ Center (km/s)',  'Kp Center', 'log(T) (K)']
    
    
    # Get the number of CMEs in each grid cell
    nCMEs = np.sum(allGrid[:,:,:,1]*allGrid[:,:,:,0], axis=0)
    ngrid = 2 * plotwid+1
    toPlot = np.zeros([ngrid,ngrid,10])
    toPlot[:,:,0] = nCMEs / (nEns-nFails) * 100
    toPlot[:,:,1] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,34], axis=0) / nCMEs
    toPlot[:,:,2] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,24], axis=0) / nCMEs
    toPlot[:,:,3] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,35], axis=0) / nCMEs
    toPlot[:,:,4] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,38], axis=0) / nCMEs
    toPlot[:,:,5] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,26], axis=0) / nCMEs
    toPlot[:,:,6] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,33], axis=0) / nCMEs
    toPlot[:,:,7] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,25], axis=0) / nCMEs
    toPlot[:,:,8] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,36], axis=0) / nCMEs
    toPlot[:,:,9] = np.sum(allGrid[:,:,:,0]* allGrid[:,:,:,37], axis=0) / nCMEs

    # adjust limits? especially for Bz panels
        
    
    subXX, subYY = np.meshgrid(range(-plotwid,plotwid+1),range(-plotwid,plotwid+1)) 
    caxes = np.empty(10)
    divs = np.empty(10)
        
    for i in range(len(axes)):
        axes[i].set_facecolor('k')
        axes[i].set_aspect('equal', 'box')
        
        toPlotNow = toPlot[:,:,i]
        cent, rng = np.mean(toPlotNow[nCMEs>0]), 1.5*np.std(toPlotNow[nCMEs>0])
        if i == 0: cent, rng = 50, 50
        toPlotNow[nCMEs==0] = np.inf
        c = axes[i].pcolor(subXX,subYY,toPlotNow,cmap=cmap1,  vmin=cent-rng, vmax=cent+rng, shading='auto')
        div = make_axes_locatable(axes[i])
        if i < 5:
            cax = div.append_axes("top", size="5%", pad=0.05)
        else:
            cax = div.append_axes("bottom", size="5%", pad=0.05)
        cbar = fig.colorbar(c, cax=cax, orientation='horizontal')
        if i < 5:
            cax.xaxis.set_ticks_position('top')
            cax.set_title(labels[i], fontsize=12)    
        else:    
            cbar.set_label(labels[i], fontsize=12)    
            
            
        axes[i].plot(0, 0, 'o', ms=15, mfc='#98F5FF')
        if i > 4:
            axes[i].set_xticklabels([])
        else:
            axes[i].xaxis.set_ticks([-plotwid, -plotwid/2, 0, plotwid/2, plotwid])
            axes[i].tick_params(axis='x', which='major', pad=5)
        if i not in [0,5]: axes[i].set_yticklabels([])
            
            
    axes[2].plot(-21.9,-11.3,'o')    
    plt.xticks(fontsize=10)    
    plt.subplots_adjust(wspace=0.1, hspace=0.18,left=0.05,right=0.95,top=0.85,bottom=0.12)    
    plt.savefig(pname)
    plt.savefig('Contours.pdf')
        

       
    
if __name__ == '__main__':

    # Get all the parameters from text files and sort out 
    # what we actually ran
    OSP.setupOSPREI()
    ResArr = txt2obj()

    
    global nEns, nFails, nHits
    nFails, nHits = 0, 0
    nEns = len(ResArr.keys())
    for key in ResArr.keys():
        if ResArr[key].fail:
            nFails +=1
        if not ResArr[key].miss:
            nHits +=1
    
    allGrid = makePostANTdata(ResArr)
    
    plotAll(allGrid)

