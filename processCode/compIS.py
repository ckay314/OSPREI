import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import datetime
import matplotlib.dates as mdates


def plotIS(obsIn, starts=None, end=None, fileDir=None, stretch=None, isSat=True, outName=None, frontpad=1, scales=None, cols=['#882255', '#88CCEE', '#332288',  '#44AA99']):
    # Obs In should have format
    # 0 Year 	1 DOY 	2 hour	3 Btot	4 Br/x	5 Bt/y	6 Bn/z	7 Np 8 Vtot	9 Tp

    nSat = len(obsIn)
    
    if not scales:
        scales = [[1,1,1,1,1,1,1] for i in range(nSat)]
    
    # Read in the data files
    data = []
    obsDTs = []
    for i in range(nSat):
        aFile = filesIn[i]
        if fileDir:
            data.append(np.genfromtxt(fileDir+aFile, dtype=float))
        else:
            data.append(np.genfromtxt(fileDir+aFile, dtype=float))
            
        thisData = data[i]
        base = datetime.datetime(int(thisData[0,0]), 1, 1, 0, 0)
        obsDT = np.array([base + datetime.timedelta(days=int(thisData[i,1])-1, seconds=int(thisData[i,2]*3600)) for i in range(len(thisData[:,0]))])
        obsDTs.append(obsDT)
    
    # check if given start times, align relative to first one
    if starts:
        st1 = starts[0]
        startplot = datetime.datetime(st1[0], st1[1], st1[2], st1[3], st1[4])
        for i in range(nSat-1):
            nextst = starts[i+1]
            startplot2 = datetime.datetime(nextst[0], nextst[1], nextst[2], nextst[3], nextst[4])
            timedelta = startplot2 - startplot
            obsDTs[i+1] = obsDTs[i+1] - timedelta
            print ('Shifting by: ', timedelta)
    
    # set up fig    
    fig, axes = plt.subplots(7, 1, sharex=True, figsize=(6,8))
    
    # plot things
    pltidxs = [3,4,5,6,8,9,7]
    #cols = ['#882255', '#88CCEE', '#332288',  '#44AA99']
    norms = [1,1,1,1,1,1e6,1]
    for i in range(nSat):
        for j in range(7):
            axes[j].plot(obsDTs[i], data[i][:,pltidxs[j]]/norms[j], '--', lw=2, c=cols[i])
            axes[j].plot(obsDTs[i], data[i][:,pltidxs[j]]/norms[j] * scales[i][j], lw=3, c=cols[i])
            
    axes[0].set_ylabel('B (nT)')
    if isSat:
        axes[1].set_ylabel('B$_R$ (nT)')
        axes[2].set_ylabel('B$_T$ (nT)')
        axes[3].set_ylabel('B$_N$ (nT)')
    else:
        axes[1].set_ylabel('B$_x$ (nT)')
        axes[2].set_ylabel('B$_y$ (nT)')
        axes[3].set_ylabel('B$_z$ (nT)')
    axes[4].set_ylabel('v (km/s)')
    axes[5].set_ylabel('T (MK)')
    axes[6].set_ylabel('n (cm$^{-3}$)')
    
    if starts:
        st1 = starts[0]
        st = datetime.datetime(st1[0], st1[1], st1[2], st1[3], st1[4])
        for i in range(7):
            yl = axes[i].get_ylim()
            axes[i].plot([st, st], yl, 'k--', zorder=0)
            
    if starts and end:
        pad = 6
        st1 = starts[0]
        startplot = datetime.datetime(st1[0], st1[1], st1[2], st1[3], st1[4]) - datetime.timedelta(hours=frontpad*pad)
        endplot = datetime.datetime(end[0], end[1], end[2], end[3], end[4]) + datetime.timedelta(hours=pad)
        axes[6].set_xlim([startplot, endplot])
    
    plt.subplots_adjust(hspace=0.1,left=0.15,right=0.95,top=0.95,bottom=0.15) 
    myFmt = mdates.DateFormatter('%Y %b %d %H:%M ')
    axes[6].xaxis.set_major_formatter(myFmt)
    plt.setp( axes[6].xaxis.get_majorticklabels(), rotation=20, horizontalalignment='right')
    
    if outName:
        if fileDir:
            plt.savefig(fileDir+outName)
        else:
            plt.savefig(outName)
    else:            
        plt.show()
    


fileDir = '/Users/ckay/OSPREI/MEOWHiSS/'
filesIn = ['20190910_pspdata.dat', '20190915_stadata.dat']    
plotIS(filesIn, fileDir=fileDir, starts=[[2019,9,19,16,38],[2019,9,21,10,17]], end=[2019,9,25,0,0], outName='HSScomp.png', frontpad=4, cols=['#7D3C98', 'r'])

fileDir = '/Users/ckay/OSPREI/runFiles/'
filesIn = ['20220310_soldata.dat', '20220310_omnidataSWAP.dat']    
plotIS(filesIn, fileDir=fileDir, starts=[[2022,3,11,22,12],[2022,3,13,22,1]], end=[2022,3,12,15,40], frontpad=2, scales=([1,1,1,1,1,1,1], [4,5,5,3,1,1,1]), outName='CMEcomp.png', cols=['#3498DB', 'g'])