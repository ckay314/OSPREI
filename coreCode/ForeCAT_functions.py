from __future__ import division
from matplotlib.patches import Patch
from pylab import *
import numpy as np
import math
import sys
import os
import pickle as pickle
import CME_class as CC

global dtor, radeg

dtor  = 0.0174532925  # degrees to radians
radeg = 57.29577951    # radians to degrees

def readinputfile():
    # Get the CME number
    global fprefix
    if len(sys.argv) < 2: 
        #sys.exit("Need an input file")
        print('No input file given!')
        sys.exit()
    else:
        input_file = sys.argv[1]
        inputs = np.genfromtxt(input_file, dtype=str, encoding='utf8')
        fprefix = input_file[:-4]
        input_values = get_inputs(inputs)
    return input_values, inputs

def get_inputs(inputs):
    # this contains all the ForeCAT things but everything else used by OSPREI
    possible_vars = ['CMElat', 'CMElon', 'CMEtilt', 'CMEyaw', 'CMEvr', 'CMEAW', 'CMEAWp', 'CMEdelAx', 'CMEdelCS', 'CMEr', 'FCtprint', 'date', 'FCmagname', 'FCrmax', 'FCRotCME', 'FCNtor', 'FCNpol', 'L0', 'FCraccel1', 'FCraccel2', 'FCvrmin', 'FCAWmin', 'FCAWr', 'CMEM', 'FCrmaxM', 'SunR', 'SunRotRate', 'SunRss', 'PFSSscale', 'saveData', 'printData', 'useFCSW', 'IVDf1', 'IVDf2', 'IVDf','time', 'SWCd', 'SWCdp', 'SWn', 'SWv', 'SWB', 'SWT', 'FRB', 'FRtau', 'FRCnm', 'FRpol', 'FRT', 'Gamma','suffix', 'nRuns', 'SatLat', 'SatLon', 'SatR', 'SatRot', 'models', 'ObsDataFile', 'SWfile', 'flagScales', 'doPUP', 'satPath', 'MHarea', 'MHdist', 'doMH', 'isSat', 'obsFRstart', 'obsFRend', 'obsShstart', 'simYaw', 'SWR']
    # if matches add to dictionary
    input_values = {}
    # Set up defaults that we have to have to run and might be wanted for ensembles
    # Will overwrite values if given, this just lets us ensemble about defaults for subtle
    # things that might be wanted.  Anything big like mass or position we want to force there
    # to be an input value if we want to ensemble it
    
    # This is all replaced by new checker in OSPREI?
    '''input_values['SWCdp'] = 1.   
    input_values['CMEr'] = 1.1
    input_values['FCraccel1'] = 1.3
    input_values['FCraccel2'] = 10.0
    input_values['FCvrmin'] = 50#*1.e5
    input_values['FCAWmin'] = 5.
    input_values['FCAWr'] = 1.
    input_values['FCrmaxM'] = 21.5
    input_values['CMEdelAx'] = 0.75
    input_values['CMEdelCS'] = 1.
    input_values['FRtau'] = 1.
    input_values['FRCnm'] = 1.927
    input_values['FRpol'] = 1
    input_values['IVDf1'] = 0.5
    input_values['IVDf2'] = 0.5
    input_values['Gamma'] = 1.33'''
    
             
    for i in range(len(inputs)):
        temp = inputs[i]
        if temp[0][:-1] in possible_vars:
            input_values[temp[0][:-1]] = temp[1]
        else:
            print (temp[0][:-1], ' not a valid input ')
            
    return input_values

def getInps(input_values, flagDate=False):
    global rsun, rotrate, kmRs, Rss
    # assume solar defaults
    rsun = 7e10
    rotrate = 2.8e-6
    Rss = 2.5
    if 'SunR' in input_values:  rsun = float(input_values['SunR'])
    if 'SunRotRate' in input_values:  rotrate = float(input_values['SunRotRate'])
    if 'SunRss' in input_values:  Rss = float(input_values['SunRss'])
    kmRs  = 1.0e5 / rsun 
    
    
    # pull parameters for initial position
    try:
        ilat = float(input_values['CMElat'])
        ilon = float(input_values['CMElon'])
        tilt = float(input_values['CMEtilt'])
    except:
        print('Missing at least one of ilat, ilon, tilt.  Cannot run without :(')
        sys.exit()
    
    init_pos = [ilat, ilon, tilt]
    
    # pull Carrington Rotation (or other ID for magnetogram)
    # code runs wrt CR instead of date b/c originally used CR
    global CR
    try: 
        CR = int(input_values['date'])
    except:
        if not flagDate:
            print('Missing magnetogram date or Carrington Rotation ID.  Cannot run without :(')
            sys.exit()
        
    # check for drag coefficient
    global Cd
    Cd = float(input_values['SWCdp'])
    
    # check for CME shape and initial nose distance
    global rstart, deltaAx, deltaCS
    rstart = float(input_values['CMEr'])
    deltaAx = float(input_values['CMEdelAx'])
    deltaCS = float(input_values['CMEdelCS'])
    
    # get distance where we stop the simulation
    try: 
        rmax = float(input_values['FCrmax'])
    except:
        #print('Assuming ForeCAT simulation stops at 10 Rs')
        rmax = 10.
    
    # determine frequency to print to screen
    tprint = 10. # default value
    if 'FCtprint' in input_values: tprint = float(input_values['FCtprint'])
        
    # determine if including rotation
    global rotCME
    rotCME = True
    if 'FCRotCME' in input_values: 
        if input_values['FCRotCME'] == 'False': 
            rotCME = False
        
    # determine torus grid resolution
    Ntor = 15
    Npol = 13
    if 'FCNtor' in input_values:  Ntor = int(input_values['FCNtor'])
    if 'FCNpol' in input_values:  Npol = int(input_values['FCNpol'])
    
    # determine L0 parameter
    global lon0
    lon0 = 0.
    if 'L0' in input_values:  lon0 = float(input_values['L0'])
                
    # get radial propagation model params
    global rga, rap, vmin, vmax, a_prop
    rga = float(input_values['FCraccel1'])
    rap = float(input_values['FCraccel2'])
    vmin = float(input_values['FCvrmin']) * 1e5
    try:
        vmax = float(input_values['CMEvr']) *1e5
    except:
        print('Need final CME speed CMEvr')
        sys.exit()
    a_prop = (vmax**2 - vmin **2) / 2. / (rap - rga) # [ cm/s ^2 / rsun]
    
    # get expansion model params
    global aw0, awR, awM
    aw0 = float(input_values['FCAWmin'])
    awR = float(input_values['FCAWr'])
    try:
        awM = float(input_values['CMEAW'])
    except:
        print('Need final CME angular width CMEAW')  
        sys.exit()          
    global user_exp 
    user_exp = lambda R_nose: aw0 + (awM-aw0)*(1. - np.exp(-(R_nose-1.)/awR))
    # check if we have AWp given, for now use ratio of AWp/AWmax to scale
    # expansion model until adding expansion forces
    global AWratio
    try:
        AWp = float(input_values['CMEAWp'])
        AWratio = lambda R_nose: (AWp/awM)*(1. - np.exp(-(R_nose-1.)/awR))
    except:
        # arbitrary default of 3
        AWratio = 1/3.
            
    # mass
    global rmaxM, max_M
    rmaxM = float(input_values['FCrmaxM']) 
    try:
        max_M = float(input_values['CMEM']) * 1e15
    except:
        print('Assuming 1e15 g CME')            
        max_M = 1e15
    global user_mass
    user_mass = lambda R_nose: np.min([max_M / 2. * (1 + (R_nose-rstart)/(rmaxM - rstart)), max_M])
    
    global saveData
    saveData = False
    if 'saveData' in input_values: 
        if input_values['saveData'] == 'True': 
            saveData = True
    
    global printData
    printData = True
    if 'printData' in input_values: 
        if input_values['printData'] == 'False': 
            saveData = False
            
    global PFSSscale
    if 'PFSSscale' in input_values:
        PFSSscale = float(input_values['PFSSscale'])
    else:
        PFSSscale = 1.
            
    return init_pos, rmax, tprint, Ntor, Npol
    

def initdefpickle(CR, picklejar, picklename):
	global dists
	# distance pickle [inCH, fromHCS, fromCH, fromPS(calc later maybe)]
	f1 = open(picklejar+'PFSS_'+picklename+'dists3.pkl', 'rb')

	#print "loading distance pickle ..."
	dists = pickle.load(f1)
	f1.close()
	# make arrays not lists
	dists = np.array(dists)


def user_vr(R_nose, rhat):
    if R_nose  <= rga: vtemp = vmin
    elif R_nose > rap: vtemp = vmax
    else: vtemp = np.sqrt(vmin**2 + 2. * a_prop * (R_nose - rga))
    return vtemp, vtemp*rhat

def openfile(CME):
    global outfile
    outfile = open(fprefix + ".dat", "w")
    printstep(CME)
    
def printstep(CME):
    thislon = CME.points[CC.idcent][1,2]
    if lon0 > -998:
        thislon -= lon0
        thislon += rotrate * 60. * radeg * CME.t
        thislon = thislon % 360.
    tilt = CME.tilt
    if tilt > 180: tilt -=360.
    vCME = np.sqrt(np.sum(CME.vels[0,:]**2))/1e5
    vdef = np.sqrt(np.sum((CME.vdefLL+CME.vdragLL)**2))/1e5
    # outdata is [t, lat, lon, tilt, vCME, vDef, AW, A, B]
    outdata = [CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], thislon, tilt, vCME, vdef, CME.AW*radeg, CME.AWp*radeg, CME.deltaAx, CME.deltaCS, CME.deltaCSAx, CME.FRBtor, CME.vs[4]/1e5]
    outless = [CME.t, CME.points[CC.idcent][1,0], CME.points[CC.idcent][1,1], thislon, tilt, vCME, vdef, CME.AW*radeg, CME.AWp*radeg]
    outprint = ''
    if printData: 
        outprint = ''
        for i in outless:
            outprint = outprint +'{:7.3f}'.format(i) + ' '
        print (outprint)  
    
    if saveData: 
        outprint = ''
        for i in outdata:
            outprint = outprint +'{:7.3f}'.format(i) + ' '
        outfile.write(outprint+'\n')
    

def calc_drag(CME):
#only calculate nonradial drag (ignore CME propagation)

	# need to calculate SW density use Guhathakurta 2006 model 
	# which depends on angular distance from HCS
	HCSdist = calc_dist(CME.cone[1,1], CME.cone[1,2]) 

	# determine SW speed
	global SW_v, SW_rho, varCd
	SW_rho, SW_v = calc_SW(CME.points[CC.idcent][1,0]-CME.rr, HCSdist)

	# get total CME velocity vector (def+drag, not propagation)
	vr = np.sqrt(np.sum(CME.vels[0,:]**2))
	#CME_v = CME.vels[1,:] + CME.vels[2,:]
	colat = (90. - CME.cone[1,1]) * dtor
	lon = CME.cone[1,2] * dtor
	colathat = np.array([np.cos(lon) * np.cos(colat), np.sin(lon) * np.cos(colat), -np.sin(colat)]) 
	lonhat = np.array([-np.sin(lon), np.cos(lon), 0.])
	# scale the value to the new timestep
	CME_v = (CME.vdefLL[0] * colathat + CME.vdefLL[1] * lonhat + CME.vdragLL[0] * colathat + CME.vdragLL[1] * lonhat)  / (CME.points[CC.idcent][1,0] + vr * CME.dt * 60 / rsun) 

	# remove any radial component
	CMEv_nr = CME_v - np.dot(CME_v, CME.rhat) * CME.rhat	
	magdifvec = np.sqrt(CMEv_nr[0]**2 + CMEv_nr[1]**2 + CMEv_nr[2]**2)

	# use a variable form of Cd = tanh(beta)
	# this means using some approx for beta -> fit to Aschwanden Physics of Solar Corona figure 
	H = np.maximum(CME.cone[1,0] - 1., 0.01)
	#print H
	beta = 2.515 * np.power(H, 1.382) 
	varCd = Cd * math.tanh(beta)
	# determine drag force
	Fd = - (2. * varCd * SW_rho / CME.rp  / rsun / math.pi) * CMEv_nr * magdifvec
	return Fd

def calc_dist(lat, lon):
# copied over from pickleB

# Use a similar slerping method as calcB but for the distance pickle.  We use the distances
# determined at the source surface height so that we must perform three slerps but no linterp.
# The distance pickle has dimensions [180, 360, 3], not the half degree resolution used in
# the B pickle
	# determine the nearest grid indices
	latidx = int(lat) + 89 # lower index for lat (upper is just + 1)
	lonidx = int(lon) % 360      # lower index for lon
	lonidx2 = (lonidx + 1) % 360 # make wrap at 0/360
	p1 = dists[latidx+1, lonidx]    
	p2 = dists[latidx+1, lonidx2]   
	p3 = dists[latidx, lonidx]      
	p4 = dists[latidx, lonidx2]     
	angdist = trislerp(lat, lon, p1, p2, p3, p4, 1.)
	return angdist


def trislerp(lat_in, lon_in, q1, q2, q3, q4, delta):
# copied over from pickleB

# This function assumes the spacing between points is delta. It uses the standard slerp formula 
#to slerp twice in longitude and then uses those results for one slerp in latitude.  This 
#function works fine for either scalar or vector qs.
	f_lat = (lat_in % delta) / delta  # contribution of first point in lat (0->1)
	f_lon = (lon_in % delta) / delta  # contribution of first point in lon (0->1)
	omega = delta * 3.14159 / 180.  # angular spacing
	# two lon slerps
	qa = (q1 * np.sin((1-f_lon) * omega) + q2 * np.sin(f_lon * omega)) / np.sin(omega) 
	qb = (q3 * np.sin((1-f_lon) * omega) + q4 * np.sin(f_lon * omega)) / np.sin(omega) 
	# one lat slerp
	qf = (qb * np.sin((1-f_lat) * omega) + qa * np.sin(f_lat * omega)) / np.sin(omega) 
	return qf



def calc_SW(r_in, HCSang):
	# Guhathakurta values
	fluxes = [2.5e3, 1.6e3] # SB and CH flux
	width_coeffs = [64.6106, -29.5795, 5.68860, 2.5, 26.3156]
	Ncs_coeffs = [2.6e5, 5.5986, 5.4155, 0.82902, -5.6654, 3.9784]
	Np_coeffs  = [8.6e4, 4.5915, 2.4406, -0.95714, -3.4846, 5.6630]
	
	#  mass flux at 1 AU
	scale = 1.
	SBflux = scale * fluxes[0] * 215.**2 #v in km; 1 Au nv = 2500*1e8 /s or Mdot= 1.86e-14 Msun/yr
	CHflux = scale * fluxes[1] * 215.**2 # Mdot = 1.19e-14
	## determine width of SB (from MHD simulation) !probably changes w/solar cycle... explore later
	my_w = width_coeffs[0] - width_coeffs[1] * r_in + width_coeffs[2] * r_in **2
	if r_in > width_coeffs[3]: my_w =  width_coeffs[4] # value at maximum
	## calculate CS and CH polynomial values
	ri = 1. / r_in
	ri = np.min([1., ri])
	#print ri
	##multiplied in the mysterious 1e8 G06 doesnt mention but includes
	Ncs = Ncs_coeffs[0] * np.exp(Ncs_coeffs[1] *ri +Ncs_coeffs[2] *ri**2) * ri**2 * (1. + Ncs_coeffs[3]  * ri + Ncs_coeffs[4]  * ri**2 + Ncs_coeffs[5]  * ri**3)
	Np  = Np_coeffs[0] * np.exp(Np_coeffs[1]*ri + Np_coeffs[2]*ri**2) * ri**2 * (1. + Np_coeffs[3] * ri + Np_coeffs[4] * ri**2 + Np_coeffs[5] * ri**3)

	# determine relative contributions of SB and CH polys
	exp_factor = np.exp(-HCSang**2 / my_w**2 / 2.) #2 from method of getting my_w
	my_dens = (Np + (Ncs - Np) * exp_factor)  # cm^-3

	# Chen density
	#my_dens = 3.99e8 * (3. * ri**12 + ri**4) + 2.3e5 * ri**2

	# determine velocity from flux and density
	my_vel  = 1.e5 * (CHflux + (SBflux - CHflux) * exp_factor)/ my_dens / r_in**2  #cm/s

	return my_dens * 1.6727e-24, my_vel

# Geometry programs
def SPH2CART(sph_in):
	r = sph_in[0]
	colat = (90. - sph_in[1]) * dtor
	lon = sph_in[2] * dtor
	x = r * np.sin(colat) * np.cos(lon)
	y = r * np.sin(colat) * np.sin(lon)
	z = r * np.cos(colat)
	return np.array([x, y, z])

def CART2SPH(x_in):
# calcuate spherical coords from 3D cartesian
# output lat not colat
	r_out = np.sqrt(x_in[0]**2 + x_in[1]**2 + x_in[2]**2)
	colat = np.arccos(x_in[2] / r_out) * 57.29577951
	lon_out = np.arctan2(x_in[1] , x_in[0]) * 57.29577951
	return np.array([r_out, 90. - colat, lon_out])

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