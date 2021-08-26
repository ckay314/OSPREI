import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import math
import matplotlib.cm as cm
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import CubicSpline

# This has the xyz coordinate at 1.05 Rs saved for each
# latitude and longitude position -> easy way to convert 
# from Bxyz to Br
global xyz
f3 = open('/Users/ckay/PickleJar/xyz.pkl', 'rb')
xyz = pickle.load(f3)
f3.close()

global dtor
dtor = math.pi/180.


def getsubB(date, x1,x2,y1,y2, height):
    # Input is idxs, should correspond to 0.5 deg resolution.
    # Origin for y is lower -> 0 is colat 180, 181 is colat 0 
    fname = '/Users/ckay/PickleJar/PFSS'+ date + 'a3.pkl'
    
    # Load slice from pickle pickle
    f1 = open(fname, 'rb')
    B = pickle.load(f1)
    f1.close()
    
    # Assuming 0.01 Rs resolution in height
    idx = int((height-1.)*100)
    Bslice = B[idx,:,:,:]

    # calc Br from Bxyz
    global Br
    Br = np.zeros([361, 720])
    for i in range(361):
    	for j in range(720):
		    Br[i,j] = (Bslice[i,j,0] * xyz[i,j,0] +  Bslice[i,j,1] * xyz[i,j,1] 
				+  Bslice[i,j,2] * xyz[i,j,2])# / 1.05
    
    # Take a sub portion of Br                
    subB = Br[y1:y2+1,x1:x2+1]
    
    # Calculate lats and lons from this range
    # assuming 361x720 resolution
    nTheta, nPhi = 361, 720
    Thetas = np.linspace(math.pi,0,nTheta)
    # shift end points slightly to avoid div by sintheta=0
    Thetas[0] -= 0.0001
    Thetas[-1] += 0.0001
    Phis = np.linspace(0,2*math.pi,nPhi, endpoint=False)
    subT = Thetas[y1:y2+1]
    subP = Phis[x1:x2+1]
    lons, lats = np.meshgrid(subP,subT)
    
    return subB, [lons*180./math.pi,90.-lats*180./math.pi]
    

def extractPIL(date, x1,x2,y1,y2, height):
    # Grab a subsection of the magnetic field around the AR
    global subB, pos
    subB, pos = getsubB(date, x1,x2,y1,y2, height)
       
    # Calculate the average properties of the region    
    stdB = np.std(np.abs(subB))
    meanB = np.mean(np.abs(subB))

    # Mask the magnetic field array to grab only higher strength portion
    global maskit, d
    maskit = np.abs(subB) < meanB
    d = np.ma.array(subB, mask=maskit)
    #print 'Average B: ', np.mean(np.abs(d))
    #print 'Average B scaled (nT): ', np.mean(np.abs(d)) * (height/213.)**2 * 1e5

    # Calculate 2 PIL positions by balancing the sum of B^2/dist on each side
    # and by finding where |Br| is a minumum
    ys = pos[1][:,0]
    xs = pos[0][0,:]
    PILxs = []
    PILxs2 = []
    
    for i in range(len(ys)):
        # Take a slice of B with constant latitude
        theseB = subB[i,:]
        
        # Method 1
        tracker = 99999999999.
        thisx = xs[0]
        # Find where the extreme values of Br exist
        # and only look between them
        minBidx = np.where(subB[i,:]==np.min(subB[i,:]))[0]
        maxBidx = np.where(subB[i,:]==np.max(subB[i,:]))[0]
        if minBidx > maxBidx:
            startIdx = maxBidx[0]
            stopIdx = minBidx[0]
        else:
            startIdx = minBidx[0]
            stopIdx = maxBidx[0]        
        # Make sure we have an even range outside the max points or can accidentally 
        # weight to one side or the other, should probably do for actual PIL point too?
        dist = (stopIdx - startIdx)/2
        # id2 defined so can do :id2
        if (startIdx - dist) < 0:
            if stopIdx+dist > len(xs)-1:
                id2 = len(xs)
                id1 = startIdx - (len(xs)-stopIdx-1)
            else:
                id1 = 0
                id2 = stopIdx + startIdx + 1
        elif (stopIdx + dist) > len(xs)-1:
            id2 = len(xs)
            id1 = startIdx - (len(xs)-stopIdx-1)
        else:
            id1 = startIdx - dist
            id2 = stopIdx + dist + 1
        id1, id2 = int(id1), int(id2)
        # Force it to be a distance "pad" from the edges
        pad = int(0.25 * dist)
        for d0 in range(startIdx+pad,stopIdx-pad):
            thisBal = np.abs(np.sum(theseB[id1:d0]**2/np.abs(xs[id1:d0]-xs[d0])) - np.sum(theseB[d0+1:id2]**2/np.abs(xs[d0+1:id2]-xs[d0])))
            if thisBal < tracker:
                tracker = thisBal
                thisx = xs[d0]
        PILxs.append(thisx)
    
        # second set of PIL points based on min in Br
        minBidx2 = np.where(np.abs(subB[i,:])==np.min(np.abs(subB[i,:])))[0]
        PILxs2.append(xs[minBidx2][0])

    PILxs = np.array(PILxs)
    PILxs2 = np.array(PILxs2)

    return PILxs, PILxs2, ys

def fitPIL(PILxs,PILxs2,ys, plotit=False):
    # Determine when the two PIL fits are "close enough"
    # to a match and only use those points
    PILdiff = np.abs(PILxs-PILxs2)
    closish = np.where(PILdiff <=10.)[0]
       
    # Only take PIL in unmasked range 
    masky= np.ma.array(pos[1], mask=maskit)
    minyID, maxyID = int(np.where(ys == np.min(masky))[0]), int(np.where(ys == np.max(masky))[0])
    closish = np.array(closish)
    closish = closish[np.where((closish>=minyID) & (closish <=maxyID))]
    
    # Fit a spline to the PIL
    thefit = CubicSpline(ys[closish],PILxs2[closish],bc_type='natural')
    fitxs = thefit(ys[closish])
    fitys = ys[closish]
       
    # Set up the plot if we are plotting
    if plotit:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        meanB = np.mean(np.abs(subB))
        maxB = 0.5*np.max(np.abs(Br))
        ax.contourf(pos[0],pos[1],d, 20, vmin=-maxB, vmax=maxB)
        ax.plot(PILxs[minyID:maxyID+1],ys[minyID:maxyID+1],'o', color='aqua')
        ax.plot(PILxs2,ys,'o', color='gray')
        ax.plot(PILxs2[closish],ys[closish],'ko')
        ax.plot(fitxs,fitys,'k--', linewidth=3)

    # Split the PIL into segments based on it changing direction
    # Calculate change in lon between points and sign of change
    xdiffs = fitxs[1:] - fitxs[:-1]
    xdsign = np.sign(xdiffs)
    # Find point with no change, may be part of PIL if relatively vertical
    the0s = np.where(xdsign==0)[0]
    # Get all nonzeros, use first and last to establish bounds
    nonzeros = np.where(xdsign != 0)[0]
    for idx in range(nonzeros[0],nonzeros[-1]+1):
        # If an id not in nonzeros see if the nonzeros surrounding have same sign
        # If so switch its sign to match
        if idx not in nonzeros:
            id1 = np.max(nonzeros[np.where(nonzeros < idx)])
            id2 = np.min(nonzeros[np.where(nonzeros > idx)])
            if xdsign[id1]*xdsign[id2] == 1: xdsign[idx] = xdsign[id1]
    # Recalculate nonzeros with filled in values
    nonzeros = np.where(xdsign != 0)[0]
    # Find the sign of the first point (i.e. moving left or right)
    current_sign = xdsign[nonzeros[0]]
    all_segs = []
    this_seg = []
    # Loop through points, if sign stays same add to segment
    for idx in range(nonzeros[0],nonzeros[-1]+1):
        if xdsign[idx] == current_sign: 
            this_seg.append(idx)
        # If sign changes then start new segment and add previous
        # to the list if length > 1
        elif xdsign[idx] == -current_sign:
            if len(this_seg) > 1:
                all_segs.append(this_seg)
            this_seg = [idx]
            current_sign = xdsign[idx]
    # Add in the last segment from the loop
    if len(this_seg) > 1:
        all_segs.append(this_seg)

    # If we have one or two segments return the average lat and lon
    # and their tilt.  Two is a somewhat arbitrary cutoff but there is 
    # certainly a limit to how complex of a PIL system we can accurately
    # reproduce.  Could alternatively use a length cutoff but should 
    # change with AR size?
    outlats =[]    
    outlons =[]    
    outtilts =[]    
    outAWs = []
    outBs = []
    colors = ['darkviolet', 'green']
    if len(all_segs) <=2:    
        for i in range(len(all_segs)):
            ids = np.array(all_segs[i])+1
            xs = fitxs[ids] 
            ys = fitys[ids]
            m = np.polyfit(xs, ys, 1)
            fpoly = np.poly1d(m)
            linPIL = fpoly(xs)
            # Add PILs to the plot
            if plotit: 
                ax.plot(xs,ys, '--',linewidth=5, color=colors[i])
                ax.plot(xs, linPIL, linewidth=5, color=colors[i])
                ax.scatter([0.5*(xs[0]+xs[-1])], [0.5*(linPIL[0]+linPIL[-1])], s=150, zorder=10, color=colors[i], edgecolor='w')
            outlats.append(0.5*(linPIL[0]+linPIL[-1]))
            outlons.append(0.5*(xs[0]+xs[-1]))
            outtilts.append(np.arctan2(m[0],1.) * 180/math.pi)
            # Calculate length in degrees (half-width)
            dlon = (xs[-1]-xs[0]) * dtor
            lat1, lat2 = linPIL[0]*dtor, linPIL[-1]*dtor
            PILlen =  np.abs(np.arccos(np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(dlon)))/dtor
            outAWs.append(PILlen/2.)
            
            # Calculate magnetic field strength value
            allys = pos[1][:,0]
            yidx = []
            #allxs = pos[0][0,:]
            for yval in ys: yidx.append(np.where(allys == yval)[0])
            outBs.append(np.mean(np.abs(d[np.array(yidx),:])))
            
            
    if plotit:
        ax.set_aspect('equal')
        plt.show()
    print ('  ', len(all_segs), 'with lengths ', [len(i) for i in all_segs])
    return outlats, outlons, outtilts, outAWs, outBs


def ForeCATARPILER(date,x1,x2,y1,y2):
    heights = []
    lats = []
    lons = []
    tilts = []
    AWs = []
    Bs = []
    # CR, x,x,y,y, height
    for i in range(16):
        height = 1.1 + i*0.01 
        heights.append(height)
        PILxs, PILxs1, ys = extractPIL(date,x1,x2,y1,y2,height)
        outlats, outlons, outtilts, outAWs, outBs = fitPIL(PILxs, PILxs1, ys, plotit=True)
        lats.append(outlats)
        lons.append(outlons)
        tilts.append(outtilts)
        AWs.append(outAWs)
        Bs.append(outBs)

    finalnumber = len(outlats)   
    ids = [] 
    i = len(lats)-1
    while i > 0:
        if len(lats[i]) == finalnumber:
            ids.append(i)
            i -= 1
        else:
            i = -9999
    print( '')
    
    for i in range(finalnumber):
        print ('PIL number ', i)
        print ('Height ', 'Lat ', 'Lon ', 'Tilt ')#, 'AW ', 'B  ')
        temp = np.zeros([len(ids),3])
        counter = 0
        for j in ids:
            print (heights[j], lats[j][i], lons[j][i], tilts[j][i])#, AWs[j][i], Bs[j][i]*(heights[j]/213.)**2*1e5)
            temp[counter,:] = [lats[j][i], lons[j][i], tilts[j][i]]#, AWs[j][i], Bs[j][i]*(heights[j]/213.)**2*1e5]
            counter +=1
        print ('Average PIL lat/lon/tilt')#'/AW/B')
        print (np.mean(temp[:,0]), np.mean(temp[:,1]), np.mean(temp[:,2]))#, np.mean(temp[:,3])), np.mean(temp[:,4]))
        print (' ' )
    #PILxs, PILxs1, ys = extractPIL(date,x1,x2,y1,y2,1.15)
    #outlats, outlons, outtilts, outAWs, outBs = fitPIL(PILxs, PILxs1, ys, plotit=True)
    #for i in range(len(outlats)):
    #    print outlats[i], outlons[i], outtilts[i], outAWs[i], outBs[i]*(1.15/213.)**2*1e5
        
#ForeCATARPILER('20120712',125,250,110,170)    
#ForeCATARPILER('20100801', 120,185,185,240)    
#ForeCATARPILER('20130411', 135,190,175,235)  

#ForeCATARPILER('20200621', 600,700,220,270) 
#ForeCATARPILER('20200621', 630,720,105,185)    
   
#ForeCATARPILER('20201026', 105,185,215,270)    

# new recent test cases
#ForeCATARPILER('20210422', 510,540,118,143)    
#ForeCATARPILER('20210422', 510,540,118,143)    

  
#fitPIL('20120712',125,250,110,170,1.15, plotit=True)        

#fitPIL('20120712',370,430,130,165,1.13)    
# fitPIL('20120712',125,250,110,170,height)
# fitPIL('20120712',370,430,130,170,height)

ForeCATARPILER('20210222', 60,220,85,160)    

