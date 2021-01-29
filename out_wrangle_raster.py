#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:12:34 2021

@author: mallen
"""

import rasterio as rio
import os
import numpy as np
import pandas as pd
from scipy.stats import sem
import geopandas as gpd
import matplotlib.pyplot as plt
import glob

os.chdir("/Users/mallen/Documents/ecostress_p2/code")
city = 'la'
seed = 2021
time = 'aft'
xvab = 'lamcr'
threshold = 0.5

# import shapefile for zonal aggregation, convert land cover to to df
if city == "la":
    classes = ["tree", "grass", "soil", "water", "building", "road", "other_imp", "tallshrub"]
    shp = gpd.read_file("../data/shp/Los Angeles Neighborhood Map/geo_export_3b62483e-90a4-4acf-9359-a830f3086143_utm_clip_clean.shp")
elif city == "nyc":
    classes = ["tree", "grass", "soil", "water", "building", "road", "other_imp", "rail"]
    shp = gpd.read_file("../data/shp/nynta_20d/nynta_utm.shp")

# import env variables
lcr = rio.open("../data/lc/" + city + "_fractionalcover.tif").read() # use wall fraction instead? Does the sign of wall temperatures flip????
lampr = rio.open("../data/lc/" + city + "_lambdap.tif").read()
lamcr = rio.open("../data/lc/" + city + "_lambdac.tif").read()
dsmr = rio.open("../data/dsm/" + city + "_dsm_mean_70m.tif").read()
elevr = rio.open("../data/dsm/" + city + "_elevation_70m.tif").read() 
dtcr = rio.open("../data/dtc/" + city + "_grid_join_rasterize_clip.tif").read()

# list files
lfn =  glob.glob("../data/rf_output/" + time + "/*" + city + "_output.tif")

# import - try other standardizations
complst = []
runmeta = []
for f in lfn:
    fi = rio.open(f)
    complst.append(fi.read()[0])
    # import metadata
    runmeta.append(np.array(pd.read_csv(f[:42] + "vabparams.txt").iloc[:,1]))
complst = np.array(complst)
runmeta = pd.DataFrame(runmeta).T
runmeta.columns = lfn

# =============================================================================
# analysis
# =============================================================================
# plot eg
#fig, ax0 = plt.subplots(1, 1)
#p0 = ax0.imshow(complst[1], vmin = -2, vmax = 2)

### difference against lp/lc?
# reshape
lcrs = pd.DataFrame(lcr.reshape([lcr.shape[0], lcr.shape[1]*lcr.shape[2]]).T, columns = classes)
lamprs = pd.DataFrame(lampr.reshape([lampr.shape[1]*lampr.shape[2]]).T, columns = ['lampr'])
lamcrs = pd.DataFrame(lamcr.reshape([lamcr.shape[1]*lamcr.shape[2]]).T, columns = ['lamcr'])
dsmrs = pd.DataFrame(dsmr.reshape([dsmr.shape[1]*dsmr.shape[2]]).T, columns = ['dsmr'])
complsts = pd.DataFrame(complst.reshape([complst.shape[0], complst.shape[1]*complst.shape[2]]).T, columns = [lfn])
# concat
merge = pd.concat([complsts, lcrs, lamprs, lamcrs, dsmrs], axis = 1)

# filter for bulding fraction, nas
merge = merge.dropna()
#mergeb = merge[merge['building'] > threshold]
mergeb = merge[merge['tree'] + merge['grass'] < threshold]
#plt.scatter(mergeb['lamcr'], mergeb['lst_diff'], alpha = 0.05)

### bin based on lc, dsm??
# label based on nadir, west view, east view (check lfn, these get a random from glob)
if city == 'la':
    if time == 'aft':
        tlab = 'Afternoon'
        nadir = 1
        west = 0
        east = 2
    elif time == 'mor':
        tlab = "Morning"
        nadir = 0
        west = 2
        east = 1
elif city == 'nyc':
    if time == 'aft':
        tlab = 'Afternoon'
        nadir = 2
        west = 1
        east = 0
    elif time == 'mor':
        tlab = "Morning"
        nadir = 0
        west = 0
        east = 0

# set xvab and bins
if xvab == 'lamcr':
    bins = [1, 2, 4, 6, 9, 12, 50]
elif xvab == 'lampr':
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
elif xvab == 'dsmr':
    bins = [1, 2, 4, 6, 9, 12, 50]
    
# bin yvab
bin_centers = []
n = []
w = []
e = []
nse = []
wse = []
ese = []
for b in range(len(bins)-1):
    n.append(np.mean(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[nadir])
    w.append(np.mean(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[west])
    e.append(np.mean(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[east])
    nse.append(sem(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[nadir])
    wse.append(sem(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[west])
    ese.append(sem(mergeb[(mergeb[xvab] > bins[b]) & (mergeb[xvab] < bins[b+1])])[east])
    bin_centers.append((bins[b+1]+bins[b])/2)

# plot a scatter
fig, ax0 = plt.subplots(1, 1)
ax0.errorbar(bin_centers, n, yerr = nse, color = 'black', label = '')
ax0.errorbar(bin_centers, w, yerr = wse, color = 'indianred', label = '')
ax0.errorbar(bin_centers, e, yerr = ese, color = 'dodgerblue', label = '')

ax0.scatter(bin_centers, n, edgecolor = 'black', facecolor = 'white', label = 'Near-Nadir', zorder = 10)
ax0.scatter(bin_centers, w, edgecolor = 'indianred', facecolor = 'white', label = 'Off-Nadir Position West', zorder = 10)
ax0.scatter(bin_centers, e, edgecolor = 'dodgerblue', facecolor = 'white', label = 'Off-Nadir Position East', zorder = 10)

ax0.legend(frameon = False)
ax0.set_ylabel("Z-Score of LST")
ax0.set_xlabel("$\lambda_{C}$ (Bin Centers)")
ax0.text(0.02, 0.04, tlab, transform = ax0.transAxes)

plt.savefig("../plots/det_anis/" + city + "_" + time + "_" + xvab + "_z-score.png", dpi = 1000)