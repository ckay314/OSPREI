import numpy as np
import math
import pickle
import CME_class as CC
import ForeCAT_functions as FC

global dtor, radeg
dtor  = 0.0174532925   # degrees to radians
radeg = 57.29577951    # radians to degrees

def init_CPU(CR, Ntor, Npol, picklejar, picklename):
    global rsun, kmRs, RSS, dR, RsR
    rsun  = FC.rsun
    RSS   = FC.Rss
    RsR = rsun/7e10 # star radius in solar radii
    kmRs  = 1.0e5 / rsun   # km (/s) divided by rsun (in cm)
    #dR = (RSS - 1.0) / 150.

    # Determine the poloidal and toroidal spacing (already in CME class???)
    #if Ntor !=1: delta_maj = 120. / (Ntor - 1.) * dtor  # point spacing along toroidal
    #if Ntor ==1: delta_maj = 0.
    #delta_min = 120. / (Npol - 1.) * dtor  # point spacing along poloidal
    #t_angs = np.array([delta_maj * (int(Ntor / 2) - int(i/Npol)) for i in range(Npoints)], dtype=np.float32)
    #p_angs = np.array([ delta_min * (i - int(Npol / 2)) for i in range(Npol)] * Ntor, dtype=np.float32)

    global B_low, B_high, B_mid
    # load the pickles which hold the mag field data
    f1 = open(picklejar+'PFSS_'+picklename+'a3.pkl', 'rb')
    #print "loading low pickle ..."
    B_low = pickle.load(f1)
    f1.close()
    f1 = open(picklejar+'PFSS_'+picklename+'b3.pkl', 'rb')
    #print "loading high pickle ..."
    B_high = pickle.load(f1)
    f1.close() 

    # assume that low and high have the same nR
    global nRlow, nR, dR, midID
    nRlow = B_low.shape[0]
    nR = 2*nRlow - 1
    dR = (RSS - 1.0) / (nR-1)
    midID = (int(nRlow/10)+1)*5
    
    # make B mid here
    B_mid = np.zeros([2*midID,361,720,4])
    n1 = nRlow - midID
    B_mid[:n1,:,:,:] = B_low[midID:, :, :,:]
    n2 = 2*midID - n1 + 1
    B_mid[n1-1:,:,:,:] = B_high[:n2,:,:,:]
    # flag for in low/high pickle
    Bzone = 0
    #print (sd)


def calc_posCPU(CME):
    # new hybrid shape version
    xAx = CME.deltaAx * CME.Lp * np.cos(CME.t_angs)
    zAx = 0.5 * CME.Lp * np.sign(CME.t_angs)*(np.sin(np.abs(CME.t_angs)) + np.sqrt(1 - np.cos(np.abs(CME.t_angs))))
    xCS = CME.deltaCS * CME.rp * np.cos(CME.p_angs)
    yCS = CME.rp * np.sin(CME.p_angs)
    sinTheta =  4 * CME.deltaAx * np.sin(CME.t_angs) / np.sqrt((2*np.cos(CME.t_angs) + np.sqrt(1 + np.cos(CME.t_angs)))**2 + 16 * CME.deltaAx**2 * np.sin(CME.t_angs)**2)
    cosTheta = np.sqrt(1-sinTheta**2)
    xs = CME.cone[1,0] + xAx + xCS * cosTheta
    ys = yCS
    zs = zAx + xCS * sinTheta
   
    RLLT = [CME.cone[1,0], CME.cone[1,1] * dtor, CME.cone[1,2] * dtor, CME.tilt * dtor]
    # make a vector for position in CME coordinates
    CCvec = [xs,ys,zs]
    CCvec1 = FC.rotx(CCvec,-(90-CME.tilt))
    CCvec2 = FC.roty(CCvec1,-CME.cone[1,1])
    xyzpos = FC.rotz(CCvec2, CME.cone[1,2])
    CMErs  = np.sqrt(xyzpos[0]**2+xyzpos[1]**2+xyzpos[2]**2)
    CMElats = 90. - np.arccos(xyzpos[2]/CMErs) / dtor
    CMElons = np.arctan(xyzpos[1]/xyzpos[0]) / dtor
    negxs = np.where(xyzpos[0]<0)
    CMElons[negxs] +=180.
    CMElons[np.where(CMElons<0)] += 360.
    SPHpos = [CMErs, CMElats, CMElons]
    for i in range(CC.Npoints):
        CME.points[i][0,:] = [xyzpos[0][i], xyzpos[1][i], xyzpos[2][i]]
        CME.points[i][1,:] = [SPHpos[0][i], SPHpos[1][i], SPHpos[2][i]]
    # convenient to store these in class as arrays for serial version
    CME.rs = CMErs
    CME.lats = CMElats
    CME.lons = CMElons
    CME.xs = xyzpos[0][:]
    CME.ys = xyzpos[1][:]
    CME.zs = xyzpos[2][:]


def calc_forcesCPU(CME):
	# will repeatedly need sin/cos of lon/colat -> calc once here
	sc = np.sin((90.-CME.lats)*dtor)
	cc = np.cos((90.-CME.lats)*dtor)
	sl = np.sin(CME.lons*dtor)
	cl = np.cos(CME.lons*dtor)
	scangs = [sc,cc,sl,cl]
	# unit vec in radial direction
	rhat = np.transpose(np.array([sc*cl, sc*sl, cc]))
	
	# calculate mag field at CME points 
	BCME = getBCPU(CME.rs, CME.lats, CME.lons,scangs)
	Bhat = BCME[:,:3]/(BCME[:,3].reshape([-1,1]))
	
	# calculate mag field at adjacent points for gradients
	# really only need magnitude...
	Blat1 = getBCPU(CME.rs, CME.lats+0.5, CME.lons,scangs)
	Blat2 = getBCPU(CME.rs, CME.lats-0.5, CME.lons,scangs)
	dB2dlat = (Blat1[:,3]**2 - Blat2[:,3]**2)/2./(0.5*dtor*RsR*7e10*CME.rs)
	Blon1 = getBCPU(CME.rs, CME.lats, CME.lons+0.5,scangs)
	Blon2 = getBCPU(CME.rs, CME.lats, CME.lons-0.5,scangs)
	dB2dlon = (Blon1[:,3]**2 - Blon2[:,3]**2)/2./(0.5*dtor*RsR*7e10*CME.rs*sc)
	dB2 = np.transpose(np.array([cc*cl*dB2dlat + sl*dB2dlon, cc*sl*dB2dlat - cl*dB2dlon, -sc*dB2dlat]))
		
	# calculate tension force----------------------------------------
	sp = np.sin(CME.p_angs)
	cp = np.cos(CME.p_angs)
	st = np.sin(CME.t_angs)
	ct = np.cos(CME.t_angs)

    # good to here w/o shape
    
	# for convenience make short shape names
	a, b, c = CME.Lr, CME.rp, CME.Lp

	# toroidal tangent vector
	xt = np.transpose(np.array([-(a + b*cp)*st, 0*sp, (c+b*cp)*ct]))
	xtmag = np.sqrt(xt[:,0]**2 + xt[:,1]**2 + xt[:,2]**2)
	tgt_t = xt / xtmag.reshape([-1,1])

	# poloidal tangent vector
	xp = np.transpose(np.array([-b*sp*ct, b*cp, -b*sp*st]))
	xpmag = np.sqrt(xp[:,0]**2 + xp[:,1]**2 + xp[:,2]**2)
	tgt_p = xp / xpmag.reshape([-1,1])

	# normal vector = cross product
	norm = np.cross(tgt_p,tgt_t)

	# calculate second derivatives
	xpp = np.transpose(np.array([-b*cp*ct, -b*sp, -b*cp*st]))
	xtt = np.transpose(np.array([-(a + b*cp)*ct, 0*sp, -(c+b*cp)*st]))
	xpt = np.transpose(np.array([b*sp*st, 0*sp,-b*sp*ct]))
	
	# coefficients of the first fundamental form
	E1FF = xp[:,0]**2 + xp[:,1]**2 + xp[:,2]**2
	F1FF = xp[:,0]*xt[:,0] + xp[:,1]*xt[:,1] + xp[:,2]*xt[:,2]
	G1FF = xt[:,0]**2 + xt[:,1]**2 + xt[:,2]**2

	# coefficients of second fundamental form
	e2FF = norm[:,0] * xpp[:,0] + norm[:,1] * xpp[:,1] + norm[:,2] * xpp[:,2]	
	f2FF = norm[:,0] * xpt[:,0] + norm[:,1] * xpt[:,1] + norm[:,2] * xpt[:,2]	
	g2FF = norm[:,0] * xtt[:,0] + norm[:,1] * xtt[:,1] + norm[:,2] * xtt[:,2]	

	# rotate vectors to sun frame
	tempvec = [tgt_t[:,0], tgt_t[:,1], tgt_t[:,2]]
	tempvec = FC.rotx(tempvec,-(90.-CME.tilt))
	tempvec = FC.roty(tempvec,-CME.cone[1,1])
	tempvec = FC.rotz(tempvec, CME.cone[1,2])
	tgt_t = np.transpose(np.array(tempvec))
	tempvec = [tgt_p[:,0], tgt_p[:,1], tgt_p[:,2]]
	tempvec = FC.rotx(tempvec,-(90.-CME.tilt))
	tempvec = FC.roty(tempvec,-CME.cone[1,1])
	tempvec = FC.rotz(tempvec, CME.cone[1,2])
	tgt_p = np.transpose(np.array(tempvec))
	tempvec = [norm[:,0], norm[:,1], norm[:,2]]
	tempvec = FC.rotx(tempvec,-(90.-CME.tilt))
	tempvec = FC.roty(tempvec,-CME.cone[1,1])
	tempvec = FC.rotz(tempvec, CME.cone[1,2])
	norm = np.transpose(np.array(tempvec))

	# solar B in pol/tor components
	Bt_u = Bhat[:,0] * tgt_t[:,0] + Bhat[:,1] * tgt_t[:,1] + Bhat[:,2] * tgt_t[:,2]	
	Bp_u = Bhat[:,0] * tgt_p[:,0] + Bhat[:,1] * tgt_p[:,1] + Bhat[:,2] * tgt_p[:,2]	
    

	# unit tangent vec for projected direction of draping on surface
	DD = np.transpose(np.array([np.sign(Bp_u) * np.sqrt(1-Bt_u**2), Bt_u]))

	# determine geodesic curvature and tension force
	k = (e2FF*DD[:,0]**2 + f2FF*DD[:,0]*DD[:,1] + g2FF*DD[:,1]**2) / (E1FF*DD[:,0]**2 + F1FF*DD[:,0]*DD[:,1] + G1FF*DD[:,1]**2)
	Ft_mag = np.abs(k) * BCME[:,3]**2 / 4. / 3.14159 / (RsR*7e10)

	# remove radial component
	ndr = norm[:,0]*rhat[:,0] + norm[:,1]*rhat[:,1] + norm[:,2]*rhat[:,2]
	n_nr = np.transpose(np.array([norm[:,0] -ndr*rhat[:,0], norm[:,1] -ndr*rhat[:,1], norm[:,2] -ndr*rhat[:,2]])) 
	Ftens = np.transpose(np.array([-Ft_mag * n_nr[:,0],-Ft_mag * n_nr[:,1],-Ft_mag * n_nr[:,2]])) 

	# determine mag pressure force
	# first remove component of pressure gradients parallel to draped B
	drBhat = np.transpose(np.array([DD[:,0]*tgt_p[:,0] + DD[:,1]*tgt_t[:,0], DD[:,0]*tgt_p[:,1] + DD[:,1]*tgt_t[:,1], DD[:,0]*tgt_p[:,2] + DD[:,1]*tgt_t[:,2]]))
	dB2parmag = dB2[:,0]*drBhat[:,0] + dB2[:,1]*drBhat[:,1] + dB2[:,2]*drBhat[:,2]
	dB2perp = np.transpose(np.array([dB2[:,0] - dB2parmag*drBhat[:,0], dB2[:,1] - dB2parmag*drBhat[:,1], dB2[:,2] - dB2parmag*drBhat[:,2]])) 
	Fpgrad = dB2perp/8/3.14159
	
	# clean out any NaNs
	Ftens[np.where(np.isnan(Ftens) == True)] = 0
	Fpgrad[np.where(np.isnan(Fpgrad) == True)] = 0
	
	# put in CME object
	for i in range(CC.Npoints):
		CME.defforces[i][0,:] = Ftens[i,:]
		CME.defforces[i][1,:] = Fpgrad[i,:]
	CME.Ftens = Ftens	
	CME.Fpgrad = Fpgrad
        


def getBCPU(Rin, lat, lon, scangs, printit=False):
	# relvant globals FC.SW_v, FC.rotrate, RsR, RSS
	# doing all vec at once? is wise?
	R = np.copy(Rin)
	#R +=215.# for testing!!!!

	# check if over source surface
	aboveSS = np.where(R >= RSS)
	scale = R/RSS
	scale[np.where(scale < 1)] = 1.
	R[aboveSS] = R[aboveSS]*0 + RSS
		
	Rids = (R-1.)/dR
	Rids = Rids.astype(int)

	bottomR = 1.0
	# determine whether in low/mid/high pickle
	#minRids = np.min(Rids)
	maxRids = np.max(Rids)
	#ssIDs   = np.where(Rids >=150)
	if maxRids < nRlow-1:
		theChosenPickle = B_low
	elif maxRids < 3*midID - 10:
		theChosenPickle = B_mid
		Rids -= midID # reindex ids
		bottomR = 1. + midID * dR
	else:
		theChosenPickle = B_high
		Rids -= (nRlow - 1)
		bottomR = 1. + dR * (nRlow -1)
	latIDs = lat*2 + 179
	latIDs = latIDs.astype(int)
	lonIDs = lon*2
	lonIDs = lonIDs.astype(int)
	
	# determine the B at the 8 adjacent points
	# assuming B low for now!!!!!
	B1 = theChosenPickle[Rids, latIDs+1, lonIDs%719,:]
	B2 = theChosenPickle[Rids, latIDs+1, (lonIDs+1)%719,:]
	B3 = theChosenPickle[Rids, latIDs, lonIDs%719,:]
	B4 = theChosenPickle[Rids, latIDs, (lonIDs+1)%719,:]
	upRids = Rids+1
	upRids[aboveSS] = -1
	B5 = theChosenPickle[upRids, latIDs+1, lonIDs%719,:]
	B6 = theChosenPickle[upRids, latIDs+1, (lonIDs+1)%719,:]
	B7 = theChosenPickle[upRids, latIDs, lonIDs%719,:]
	B8 = theChosenPickle[upRids, latIDs, (lonIDs+1)%719,:]

	# determine weighting of interpolation points
	flat = lat*2 - latIDs + 179
	flon = lon*2 - lonIDs
	om = 0.5 * dtor # assuming half deg spacing
	som = np.sin(om)	

	# lon slerps
	sflon1 = np.sin((1.-flon)*om).reshape([-1,1]) 
	sflon2 = np.sin(flon*om).reshape([-1,1]) 
	Ba = (B1 * sflon1 + B2 * sflon2) / som
	Bb = (B3 * sflon1 + B4 * sflon2) / som
	Bc = (B5 * sflon1 + B6 * sflon2) / som
	Bd = (B7 * sflon1 + B8 * sflon2) / som

	# lat slerps
	sflat1 = np.sin((1.-flat)*om).reshape([-1,1])  
	sflat2 = np.sin(flat*om).reshape([-1,1]) 
	Baa = (Bb * sflat1 + Ba * sflat2) / som
	Bbb = (Bd * sflat1 + Bc * sflat2) / som
	
	# radial linterp
	fR = ((R-bottomR)/dR - Rids).reshape([-1,1]) 
	Bmag = (1-fR) * Baa + fR*Bbb
	
	# unpack the array to [#,#] instead of [#][#]
	Bmag = Bmag.reshape([len(Bmag),4])
	
	# scale if over SS height
	Bmag *= (1./(scale.reshape([-1,1])**2))

	# adjust to Parker spiral angle above SS
	# passing scangs = [sc,cc,sl,cl]
	if maxRids == nR-1:
		#print np.min(Rids)
		Br = Bmag[:,3]
		#FC.SW_v = 400.e5 # for testing!!!!
		Bphi = -Br * (R*scale-RSS) * RsR * 7e10 * FC.rotrate * scangs[0] / FC.SW_v
		BmagPS = np.sqrt(Br**2 + Bphi**2)
		Bx = scangs[0] * scangs[3] * Br - scangs[2] * Bphi
		By = scangs[0] * scangs[2] * Br + scangs[3] * Bphi
		Bz = scangs[1] * Br
		Bmag[aboveSS,0] = Bx[aboveSS]
		Bmag[aboveSS,1] = By[aboveSS]
		Bmag[aboveSS,2] = Bz[aboveSS]
		Bmag[aboveSS,3] = BmagPS[aboveSS]
        
    # return field scaled by PFSSscale    
	return Bmag*FC.PFSSscale   


def calc_torqueCPU(CME):
	sp = np.sin(CME.p_angs)
	cp = np.cos(CME.p_angs)
	st = np.sin(CME.t_angs)
	ct = np.cos(CME.t_angs)
	
	# determine distance along x-axis, will subtract from xyz to get lever arm
	toraxx = [CME.cone[1,0] + (CME.Lr+CME.rr*cp)*ct, 0.*cp, 0.*cp]
    
	toraxx = np.transpose(np.array(toraxx))
	tempvec = [toraxx[:,0], toraxx[:,1], toraxx[:,2]]
	tempvec = FC.rotx(tempvec,-(90.-CME.tilt))
	tempvec = FC.roty(tempvec,-CME.cone[1,1])
	tempvec = FC.rotz(tempvec, CME.cone[1,2])
	toraxx = np.transpose(np.array(tempvec))
	# lever arm
	LA = [CME.xs-toraxx[:,0], CME.ys-toraxx[:,1], CME.zs-toraxx[:,2]]
	LA = np.transpose(np.array(LA))

	# sum forces
	totF = CME.Ftens + CME.Fpgrad

	# torque is cross product of lever arm and def force
	FcrossLA = [LA[:,1]*totF[:,2] - LA[:,2]*totF[:,1], LA[:,2]*totF[:,0] - LA[:,0]*totF[:,2], LA[:,0]*totF[:,1] - LA[:,1]*totF[:,0]]
	FcrossLA = np.transpose(np.array(FcrossLA))

	# dot with rhat to take radial component
	tottor = FcrossLA[:,0]*CME.rhat[0]+FcrossLA[:,1]*CME.rhat[1]+FcrossLA[:,2]*CME.rhat[2]

	return np.sum(tottor)*rsun
