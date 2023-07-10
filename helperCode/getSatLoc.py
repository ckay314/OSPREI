import numpy as np
import sys
import datetime

# OSPREI wants format:
#Time                            R [AU]          lat [deg]       lon [deg]
#2022-01-26 00:00:00             0.69215         3.81529         -99.39884
#where lon can be any inertial coord sys

# helioWeb files are in
#YEAR DAY  RAD_AU  HGI_LAT  HGI_LON
#1990   1    0.98    -3.0    24.5



def getSatLoc(satName, start, stop, pad=30, fracYr=True, outName=None):
    satpath0 = 'helioweb/'
    planetlist = ['earth', 'jupiter', 'mars', 'mercury', 'neptune', 'pluto', 'saturn', 'uranus', 'venus']
    satlist = ['bepicolombo', 'cassini', 'dawn', 'galileo', 'helios1', 'helios2', 'juno', 'maven', 'messenger', 'msl', 'newhorizons', 'osirisrex', 'psp', 'rosetta', 'solo', 'spitzer', 'stereoa', 'stereob', 'ulysses']
    
    if satName in planetlist:
        satpath = satpath0 + 'planets/'
    elif satName in satlist:
        satpath = satpath0 + 'spacecraft/'
    else:
        sys.exit('Unknown satellite name')
    
    # read in appropriate helioweb file    
    hdata = np.genfromtxt(satpath+satName+'.lst', dtype=float, skip_header=1)
    
    # get year and doy est
    if fracYr:
        yr1 = int(start)
        yr2 = int(stop)
        doy1 = (start-yr1)*365 
        doy2 = (stop-yr2)*365
        stDT = datetime.datetime(year=yr1, month=1, day=1) + datetime.timedelta(days=doy1-1)
        endDT = datetime.datetime(year=yr1, month=1, day=1) + datetime.timedelta(days=doy1-1)
    # string yr
    else:
        lenSt = len(start)
        if (start[4] == '/') and (lenSt in [10,16]):
            if len(start) == 10:
                fmt = '%Y/%m/%d'
            elif len(start) == 16:
                fmt = '%Y/%m/%d %H:%M'
        elif (start[4] == ':') and (lenSt in [10,16]):
            if len(start) == 10:
                fmt = '%Y-%m-%d'
            elif len(start) == 16:
                fmt = '%Y-%m-%d %H:%M'
        else:
            sys.exit('Unknown format for start/stop time. Use either YYYY/MM/DD or YYYY-MM-DD (can also include HH:MM)')
        try:
            stDT = datetime.datetime.strptime(start, fmt)
            endDT = datetime.datetime.strptime(stop, fmt)
        except:
            sys.exit('Unknown format for start/stop time. Use either YYYY/MM/DD or YYYY-MM-DD')
    
    
    stDTp = stDT + datetime.timedelta(days=-pad)
    endDTp = endDT + datetime.timedelta(days=pad)
    
    fracyr = hdata[:,0] + hdata[:,1]/365.
    fracSt = stDTp.year + stDTp.timetuple().tm_yday/365.
    fracEnd = endDTp.year + endDTp.timetuple().tm_yday/365.
    idx = np.where((fracyr >= fracSt) & (fracyr <= fracEnd))[0]
    
    fname = satName+'_temp.sat', 'w'
    if outName:
        fname = satName+'_'+outName+'.sat'
    f1 = open(fname, 'w')
    
    f1.write('#Time    R [AU]   HGIlat [deg]   HGIlon [deg] \n')
    for i in idx:    
        thisDT = datetime.datetime(year=int(hdata[i,0]), month=1, day=1) + datetime.timedelta(hdata[i,1]-1)
        outstr = thisDT.strftime('%Y-%m-%d %H:%M:%S') + ' ' + str(hdata[i,2]) + ' '  + str(hdata[i,3])+ ' '  + str(hdata[i,4]) + '\n'
        f1.write(outstr)
    f1.close()
    
    
    
    # get Carrington longitude at start time, needed for ForeCAT
    clonEarth = getEarthClon(stDT)
    if satName == 'earth':
        clon = clonEarth
        
    # need to comp HGIlon of sat and earth to get sat clon    
    else:
        # frac DoY at start time
        fracDoY = (stDT - datetime.datetime(year=stDT.year, month=1, day=1)).total_seconds()/24/3600 +1
        
        # get earth lon at start
        edata = np.genfromtxt(satpath0+'planets/earth.lst', dtype=float, skip_header=1)
        idx = np.min(np.where((edata[:,1] >= fracDoY) & (edata[:,0] == stDT.year)))
        if edata[idx,1] == fracDoY:
            elonHGI = edata[idx,4]
        else:
            # linear interpolation
            elonHGI = edata[idx,4] + (fracDoY - edata[idx,1]) * (edata[idx-1,4] - edata[idx,4]) / (edata[idx-1,1] - edata[idx,1])
            
        # get sat lon
        idx = np.min(np.where((hdata[:,1] >= fracDoY) & (hdata[:,0] == stDT.year)))
        if hdata[idx,1] == fracDoY:
            hlonHGI = hdata[idx,4]
        else:
            # linear interpolation
            hlonHGI = hdata[idx,4] + (fracDoY - hdata[idx,1]) * (hdata[idx-1,4] - hdata[idx,4]) / (hdata[idx-1,1] - hdata[idx,1])
        #print (hlonHGI)
        # convert to sat clon
        clon = (clonEarth + (hlonHGI - elonHGI))%360
    
    return fname, clon
    
def getEarthClon(date):
    # Takes in a date time object
    date0 = datetime.datetime(year=1995, month=1, day=1)
    diff = (date-date0).total_seconds()/86400. # difference in days
    # based on https://space.umd.edu/pm/crn/carrtime.html
    est1 = (349.03 - (360 * diff / 27.2753)) % 360.
    # calcuate the correction term
    aa,bb,cc,dd,ee,ff,gg,hh,ii = 1.91759, -0.130809, -0.0825392, -0.175091, 365.271, 0.267977, -24812.2, -0.00529779, -0.00559338
    cor = ff + diff/gg + aa * np.sin(2*np.pi*diff/ee) + bb * np.sin(4*np.pi*diff/ee) + hh * np.sin(6*np.pi*diff/ee) + cc * np.cos(2*np.pi*diff/ee) + dd * np.cos(4*np.pi*diff/ee) + ii * np.cos(6*np.pi *diff/ee)
    return (est1+cor)%360
    
#fname, clon =print(getSatLoc('psp', 2022.123, 2022.156, outName='temp'))
fname, clon = getSatLoc('solo', '2022/03/11 01:06', '2022/03/14 00:00', outName='20220310', fracYr=False)
print (fname, clon)
    
    