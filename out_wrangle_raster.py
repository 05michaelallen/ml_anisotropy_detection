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
import geopandas as gpd
import matplotlib.pyplot as plt
import glob

os.chdir("/Users/mallen/Documents/ecostress_p2/code")
city = 'la'
seed = 2021

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
lfn =  glob.glob("../data/rf_output/*.tif")

# import - try other standardizations
complst = []
for f in lfn:
    fi = rio.open(f)
    complst.append(fi.read()[0])
    
complst = np.array(complst)

# =============================================================================
# analysis
# =============================================================================
# plot eg
fig, ax0 = plt.subplots(1, 1)
p0 = ax0.imshow(complst[2], vmin = -2, vmax = 2)


# plot difference? west - east
complst_d = complst[0] - complst[1]
fig, ax0 = plt.subplots(1, 1)
p0 = ax0.imshow(complst_d, cmap = plt.cm.bwr)
plt.colorbar(p0)

### difference against lp/lc?
# reshape
lcrs = pd.DataFrame(lcr.reshape([lcr.shape[0], lcr.shape[1]*lcr.shape[2]]).T, columns = classes)
lamcrs = pd.DataFrame(lamcr.reshape([lamcr.shape[1]*lamcr.shape[2]]).T, columns = ['lamcr'])
complsts = pd.DataFrame(complst_d.reshape([complst_d.shape[0]*complst_d.shape[1]]).T, columns = ['lfn'])
# concat
merge = pd.concat([lcrs, lamcrs, complsts], axis = 1)

# filter for bulding fraction, nas
merge = merge.dropna()
mergeb = merge[merge['building'] > 0.5]
plt.scatter(mergeb['lamcr'], mergeb['lst_diff'], alpha = 0.05)

# bin based on lc, dsm??
bins = [1, 2, 4, 6, 8, 15, 50]
binned = []
bin_centers = []
for b in range(6):
    binned.append(np.mean(mergeb[(mergeb['lamcr'] > bins[b]) & (mergeb['lamcr'] < bins[b+1])]['lst_diff']))
    bin_centers.append((bins[b+1]+bins[b])/2)

plt.scatter(bin_centers, binned)
