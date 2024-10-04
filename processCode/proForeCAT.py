import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.stats import norm, pearsonr
from scipy import ndimage
import datetime
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import warnings
warnings.filterwarnings("ignore")

global dtor
dtor = math.pi / 180.


# make label text size bigger
plt.rcParams.update({'font.size':14})

# Set up the path variable
# I like keeping all the code in a single folder called code
# but you do you (and update this to match whatever you do)
import OSPREI as OSP
mainpath = OSP.mainpath
sys.path.append(os.path.abspath(OSP.codepath)) #MTMYS

import processOSPREI as pO
import setupPro as sP

# |------------------------------------------------------------|
# |------------------------ CPA plot --------------------------|    
# |------------------------------------------------------------| 
def makeCPAplot(ResArr, nEns, BFs=[None], satCols=None, satNames=None):
    # |----------- Set up figure -----------|
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10,10))
    maxr = ResArr[0].FCrs[-1]
    
    # |----------- If ensemble plot the ranges -----------|
    if nEns > 1:
        fakers = np.linspace(1.1,maxr+0.05,100, endpoint=True)
        splineVals = np.zeros([nEns, 100, 3])
        means = np.zeros([100, 3])
        stds  = np.zeros([100, 3])
        lims  = np.zeros([100, 2, 3])
    
        i = 0
        # |----------- Repackage profiles -----------|
        for key in ResArr.keys():
            # Fit a spline to data since may be different lengths since take different times
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClats,bc_type='natural')
            splineVals[i,:, 0] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FClonsS,bc_type='natural')
            splineVals[i,:, 1] = thefit(fakers)
            thefit = CubicSpline(ResArr[key].FCrs,ResArr[key].FCtilts,bc_type='natural')
            splineVals[i,:, 2] = thefit(fakers)    
            i += 1
        # |----------- Eval range of profiles at each time -----------|    
        for i in range(3):
            means[:,i]  = np.mean(splineVals[:,:,i], axis=0)
            stds[:,i]   = np.std(splineVals[:,:,i], axis=0)
            lims[:,0,i] = np.max(splineVals[:,:,i], axis=0) 
            lims[:,1,i] = np.min(splineVals[:,:,i], axis=0)
            axes[i].fill_between(fakers, lims[:,0,i], lims[:,1,i], color='LightGray') 
            axes[i].fill_between(fakers, means[:,i]+stds[:,i], means[:,i]-stds[:,i], color='DarkGray') 
        

    # |----------- Plot the seed profile -----------|
    axes[0].plot(ResArr[0].FCrs, ResArr[0].FClats, linewidth=4, color='k', label='Seed', zorder=20)
    axes[1].plot(ResArr[0].FCrs, ResArr[0].FClonsS, linewidth=4, color='k', zorder=20)
    axes[2].plot(ResArr[0].FCrs, ResArr[0].FCtilts, linewidth=4, color='k', zorder=20)
   
    # |----------- Add in the best fit lines if given -----------|
    if BFs[0] or (BFs[0] == 0):
        for i in range(len(BFs)):
            BFidx = np.where(BFs ==BFs[i])[0]
            mycol = satCols[i]
            if len(BFidx) > 1:
                mycol = satCols[BFidx[0]]
            idx = BFs[i]
            axes[0].plot(ResArr[idx].FCrs, ResArr[idx].FClats, linewidth=3, color=mycol, label=satNames[i])
            axes[1].plot(ResArr[idx].FCrs, ResArr[idx].FClonsS, linewidth=3, color=mycol)
            axes[2].plot(ResArr[idx].FCrs, ResArr[idx].FCtilts, linewidth=2, color=mycol)
            
    # |----------- Add a legend -----------|
    if nEns > 1:
        ncol = 1
        if BFs[0] or (BFs[0] == 0): ncol = len(BFs)+1
        fig.legend(loc='upper center', fancybox=True, fontsize=13, labelspacing=0.4, handletextpad=0.4, framealpha=0.5, ncol=ncol)
    
    # |----------- Add in final position as text -----------|
    # |----------- If ensemble give uncertainty -----------|
    degree = '$^\\circ$'
    if nEns > 1:
        all_latfs, all_lonfs, all_tiltfs = [], [], []
        for key in ResArr.keys():
            all_latfs.append(ResArr[key].FClats[-1])
            all_lonfs.append(ResArr[key].FClonsS[-1])
            all_tiltfs.append(ResArr[key].FCtilts[-1])
        fitLats = norm.fit(all_latfs)
        fitLons = norm.fit(all_lonfs)
        fitTilts = norm.fit(all_tiltfs)
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(fitLats[0])+'$\\pm$'+'{:4.1f}'.format(fitLats[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(fitLons[0])+'$\\pm$'+'{:4.1f}'.format(fitLons[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(fitTilts[0])+'$\\pm$'+'{:4.1f}'.format(fitTilts[1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
        
    # |----------- Single value for single run -----------|    
    else:
        axes[0].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClats[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[0].transAxes)
        axes[1].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FClonsS[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[1].transAxes)
        axes[2].text(0.97, 0.05, '{:4.1f}'.format(ResArr[0].FCtilts[-1])+degree, horizontalalignment='right', verticalalignment='center', transform=axes[2].transAxes)
                  
    # |----------- Axes labels and saving -----------|
    axes[0].set_ylabel('Latitude ('+degree+')')
    axes[1].set_ylabel('Longitude ('+degree+')')
    axes[2].set_ylabel('Tilt ('+degree+')')
    axes[2].set_xlabel('Distance (R$_S$)')
    axes[0].set_xlim([1.01,maxr+0.15])
    plt.subplots_adjust(hspace=0.1,left=0.13,right=0.95,top=0.95,bottom=0.1)
    plt.savefig(OSP.Dir+'/fig_'+str(ResArr[0].name)+'_CPA.'+pO.figtag)
    plt.close() 
