#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:47:29 2021

@author: mallen
"""

# this script is a continuation from nyc_explore_v1.py

### model temperatures at nadir using the suite of environmental variables (similar to the RF previous in ep1)
# use departure to indicate anisotropy. departures should be large in high lc rough areas
# and low in flat areas

# per-pixel confidence indicates how well we can model each pixel with the given suite of variables. 
# what we look for is a drop in r2 for off nadir views, the drop should correlate with 
# increased surface structure, lp, lc, dsm, etc. 
# can quantify the drop using z score as well. e.g. how many standard deviations did we change?

# check erin's paper as well - i think she used a similar setup

# if we can set up "normal", we can then model the distribution for a given day 
# by plugging in mean + sd, then look at error
# may need to model each burough separately
# need to be particularly careful about ocean bleed into the polygon


# import packages
import os 
import rasterio as rio
from rasterio import features
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterstats as rs
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime

os.chdir("/Users/mallen/Documents/ecostress_p2/code")
city = 'la'
rtype = 'standardize'
seed = 2021

# flags
imgcheck = False
diagnostic_plots = False
ext_shp = False
sub_manhattan = False
output = True
extract_output_to_shp = True

# =============================================================================
# functions
# =============================================================================
# standardize, normalize or rescale to quartiles
def rescale(img, rtype):
    # compute mean, sd, and quartiles
    imgmean = np.nanmean(img)
    imgsd = np.nanstd(img)
    imgiqr = np.nanquantile(img, 0.75, axis = (0)) - np.nanquantile(img, 0.25, axis = (0))
    # compute selected standardization method
    img_stn = []
    if rtype == "standardize":
        img_stn = (img - imgmean)/imgsd # standarize
    elif rtype == "normalize":
        img_stn = (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img)) # normalize
    elif rtype == "rescale":
        img_stn = (img - np.nanquantile(img, 0.5))/imgiqr # rescale
    elif rtype == "none":
        img_stn = img
    else:
        print("Error: rtype must be one of the following \n 'standardize', 'normalize', 'rescale'")
    return img_stn



### note: the azimuth/solar zenith angles in ECOSTRESS metadata files are North = 0 (-180 to 180), and horizon = 90
# we adjust them below so the math is easier to interpret
def a_adj(meta):
    # adjust azimuth angles
    adj = []
    for i in range(len(meta)):
        if meta['solar_azimuth'][i] > 180:
            adj.append(meta['solar_azimuth'][i] - 180)
        else:
            adj.append(meta['solar_azimuth'][i] + 180)
    meta['solar_azimuth_adj'] = pd.Series(adj)
    adj = []
    for i in range(len(meta)):
        if meta['view_azimuth'][i] > 180:
            adj.append(meta['view_azimuth'][i] - 180)
        else:
            adj.append(meta['view_azimuth'][i] + 180)
    meta['view_azimuth_adj'] = pd.Series(adj)
    # adjust solar zenith
    meta['solar_zenith_adj'] = 90 - meta['solar_zenith'] 
    return meta



def import_reshape(fn):
    img = rio.open(fn)
    imgr = img.read()
    imgr = np.reshape(imgr, [imgr.shape[0], imgr.shape[1]*imgr.shape[2]])
    return (img, imgr)



def extract_to_shapefile(shp, rast, band, stats):
    zs = rs.zonal_stats(shp, 
                        rast.read()[band],
                        affine = rast.transform,
                        stats = stats)
    zss = []
    for stat in stats:
        zss.append([np.asarray([x[attribute] for x in zs]) for attribute in [stat]][0])
    return zss



def rmse(pred, valid):
    return np.sqrt(np.mean((pred - valid)**2))

# =============================================================================
# bring in data, pre-processing
# =============================================================================
# import metadata
meta = pd.read_csv("../data/" + city + "meta/" + "ecostress_p2_" + city + "_combinedmetadata_shift_v1.csv")

# adjust azimuth angles
meta = a_adj(meta)

### import ensemble of environmental variables
lc, lcr = import_reshape("../data/lc/" + city + "_fractionalcover.tif")
lamp, lampr = import_reshape("../data/lc/" + city + "_lambdap.tif")
lamc, lamcr = import_reshape("../data/lc/" + city + "_lambdac.tif")
dsm, dsmr = import_reshape("../data/dsm/" + city + "_dsm_mean_70m.tif")
elev, elevr = import_reshape("../data/dsm/" + city + "_elevation_70m.tif") 
dtc, dtcr = import_reshape("../data/dtc/" + city + "_grid_join_rasterize_clip.tif") 

# import shapefile for zonal aggregation, convert land cover to to df
if city == "la":
    classes = ["tree", "grass", "soil", "water", "building", "road", "other_imp", "tallshrub"]
    shp = gpd.read_file("../data/shp/Los Angeles Neighborhood Map/geo_export_3b62483e-90a4-4acf-9359-a830f3086143_utm_clip_clean.shp")
elif city == "nyc":
    classes = ["tree", "grass", "soil", "water", "building", "road", "other_imp", "rail"]
    shp = gpd.read_file("../data/shp/nynta_20d/nynta_utm.shp")

#### extract to shapefile
stats = ['mean']
if ext_shp:
    # single band
    lampr = extract_to_shapefile(shp, lamp, 0, stats)[0]
    lamcr = extract_to_shapefile(shp, lamc, 0, stats)[0]
    dsmr = extract_to_shapefile(shp, dsm, 0, stats)[0]
    elevr = extract_to_shapefile(shp, elev, 0, stats)[0]
    dtcr = extract_to_shapefile(shp, dtc, 0, stats)[0]
    # multi band 
    lcr = []
    for b in range(lc.count):
        lcr.append(extract_to_shapefile(shp, lc, b, stats)[0])
    lcr = np.stack(lcr)
    lstr = []
    for f in range(len(meta)):
        lstr.append(extract_to_shapefile(shp, 
                                         rio.open("../data/" + city + "lst/ECO2LSTE." + meta['filename'][f] + "_utm_shift_clip.tif"), 
                                         0, 
                                         stats)[0])
    lst = np.stack(lstr)
else:
    ### import temperature rasters
    lst = []
    for f in range(len(meta)):
        lstf = rio.open("../data/" + city + "lst/ECO2LSTE." + meta['filename'][f] + "_utm_shift_clip.tif")
        # read and filter imgf 
        lstfr = lstf.read()[0,:,:].astype(float)
        lstfr[lstfr == 0] = np.nan
        lst.append(lstfr)
    # stack into array
    lst = np.array(lst)
    # reshape into row = px, col = date
    lst = np.reshape(lst, [lst.shape[0], lst.shape[1]*lst.shape[2]])

# grab metadata
out_meta = rio.open("../data/" + city + "lst/ECO2LSTE." + meta['filename'][f] + "_utm_shift_clip.tif").meta.copy()
out_meta.update({"dtype": 'float64',
                 "nodata": -999, 
                 "count": 3})

### filter metadata to retrieve image indices 
#meta_vz = meta[(meta['view_zenith'] > 12) & (meta['view_azimuth_adj'] > 180) & (meta['view_azimuth_adj'] < 360)] # afternoon off nadir, west side of sky
#meta_vz = meta[(meta['view_zenith'] > 12) & (meta['view_azimuth_adj'] < 180)] # afternoon off nadir, east side of sky
meta_vz = meta[(meta['view_zenith'] < 12)] # afternoon nadir
meta_vza = meta_vz[(meta['solar_zenith_adj'] > -10) & (meta['hourfrac'] > 12)]

### rescale each image
# none for LA standardize seems to do best
ar = []
for i in meta_vza.index:
    ar.append(rescale(lst[i,:] * 0.02, rtype))

# check images
if imgcheck:
    if ext_shp:
        for i in range(len(ar)):
            pd.concat([pd.Series(ar[i], name = 'lst'), shp], axis = 1).plot(column = 'lst', legend = True)
    else:
        for i in range(len(ar)):
            print(meta_vza.reset_index(drop = True)['filename'][i])
            p = plt.imshow(ar[i].reshape(lstfr.shape[0], lstfr.shape[1])) #rm 1
            plt.colorbar(p)
            plt.show()
        

# take averages
# check counts for each, nullify areas with incomplete count
arm = np.stack(ar).mean(axis = 0)

# =============================================================================
# create model 
# =============================================================================
# merge all of the data together
merge = pd.concat([pd.DataFrame(arm.T, columns = ['lst']),
                   pd.DataFrame(lcr.T, columns = classes),
                   pd.DataFrame(lampr.T, columns = ['lamp']),
                   pd.DataFrame(lamcr.T, columns = ['lamc']),
                   pd.DataFrame(dsmr.T, columns = ['dsm']),
                   pd.DataFrame(elevr.T, columns = ['elev']),
                   pd.DataFrame(dtcr.T, columns = ['dtc']),
                   ], axis = 1)

if sub_manhattan:
    if ext_shp:
        merge = pd.concat([merge, shp['BoroName']], axis = 1)
        merge = merge[merge['BoroName'] == "Manhattan"]
        merge = merge.drop('BoroName', axis = 1)
    else:
        # rasterize 
        shapes = ((geom, value) for geom, value in zip(shp.geometry, shp.BoroCode))
        burned = features.rasterize(shapes = shapes, fill = -999, out = dtcr, transform = dtc.transform)
        

# remove nas
merge[merge == -999] = np.nan
mergena = merge.dropna()
mergena_idx = mergena.index

# yvab
yvab = np.array(mergena['lst'])

# split train/test
# note: we are training using the entire set
# then compare model output to off-nadir views
X_train, X_test, y_train, y_test = train_test_split(np.array(mergena.iloc[:,1:]), 
                                                    yvab,
                                                    test_size = 0.2,
                                                    random_state = seed)

# initialize model
model = RandomForestRegressor(n_estimators = 50, 
                              oob_score = True, 
                              random_state = seed, 
                              verbose = True, 
                              n_jobs = -1)
model.fit(X_train, y_train)

# grab feature importance
fi = model.feature_importances_
# merge with labels
cols = list(merge.iloc[:,1:].columns)
fic = pd.DataFrame([fi.T], columns = [cols]).T
print("feature importance:")
print(fic)

# get oob and post r2
oob = model.oob_score_ 
print("oob: " + str(np.round(oob, 3)))

# post score (goodness of fit)
r2 = model.score(X_test, y_test)
print("post-test r2: " + str(np.round(r2, 3)))
#rr = np.corrcoef(pred, y_test)[0,1]**2 # these are the same? 

# predict full set 
pred_test = model.predict(X_test)
pred_full = model.predict(np.array(mergena.iloc[:,1:]))

# compute post error metrics
rmse0 = rmse(pred_test, y_test)
# residuals 
resid = y_test - pred_test
resid_full = yvab - pred_full
#resid_stud = (pred_test - y_test)/np.std(yvab)

# =============================================================================
# post-processing
# =============================================================================
pred_full = pd.DataFrame(pred_full, index = mergena.index, columns = ['pred'])
resid_full = pd.DataFrame(resid_full, index = mergena.index, columns = ['resid'])
merge_pred = pd.concat([merge, pred_full, resid_full], axis = 1)

### diagnostic plots
if diagnostic_plots:
    if ext_shp:
        if sub_manhattan:
            shp = shp[shp['BoroName'] == "Manhattan"]
        merge_pred = pd.concat([merge_pred, shp], axis = 1)
        merge_pred.plot(column = "pred", legend = True)
        merge_pred.plot(column = "lst", legend = True)
        merge_pred['diff'] = merge_pred['lst'] - merge_pred['pred']
        merge_pred.plot(column = "dif", legend = True)
    else:
        p = plt.imshow(np.array(merge_pred['pred']).reshape(lstfr.shape[0],lstfr.shape[1]))
        plt.colorbar(p)
        p = plt.imshow(np.array(merge['lst']).reshape(lstfr.shape[0],lstfr.shape[1]))
        plt.colorbar(p)
        p = plt.imshow(np.array(merge['lst']).reshape(lstfr.shape[0],lstfr.shape[1]) - np.array(merge_pred['pred']).reshape(lstfr.shape[0],lstfr.shape[1]))
        plt.colorbar(p)
    
    # plot them against each other
    fig, ax0 = plt.subplots(1, 1, figsize = [4, 4])
    
    #ax0.scatter(merge_pred['lamc'], merge['lst'] - merge_pred['pred'])
    #ax0.plot([, 5], [-5, 5])

    ### residuals
    #fig, ax0 = plt.subplots(1, 1)
    #ax0.scatter(pred_test, resid, alpha = 0.025, c = 'dodgerblue')

# output rasters and metadata
if output:
    # record date/time
    t = datetime.now()
    t = t.strftime("%d_%m_%Y_%H_%M_%S")
    # parameters and vabs
    outparam = pd.DataFrame.from_dict(model.get_params(), orient = 'index')
    outparam.to_csv("../data/rf_output/" + t + "_modelparams.txt")
    outvab = list(merge.columns)
    outvab.append("view zenith," + str(np.round(meta_vza['view_zenith'].mean(), 3)))
    outvab.append("time," + str(np.round(meta_vza['hourfrac'].mean(), 3)))
    outvab.append("view azimuth," + str(np.round(meta_vza['view_azimuth_adj'].mean(), 3)))
    outvab.append("solar azimuth," + str(np.round(meta_vza['solar_azimuth_adj'].mean(), 3)))
    outvab.append("oob," + str(np.round(oob, 3)))
    outvab.append("test_r2," + str(np.round(r2, 3)))
    outvab.append("rmse," + str(np.round(rmse0, 3)))
    outvab.append("count," + str(np.round(len(meta_vza), 3)))
    pd.Series(outvab).to_csv("../data/rf_output/" + t + "_vabparams.txt")
    
    # output rasters
    targeto = np.array(merge_pred['lst']).reshape(lstfr.shape[0],lstfr.shape[1])
    predo = np.array(merge_pred['pred']).reshape(lstfr.shape[0],lstfr.shape[1])
    resido = np.array(merge_pred['resid']).reshape(lstfr.shape[0],lstfr.shape[1])
    
    # filename
    fn_out = "../data/rf_output/" + t + "_va" + str(np.round(meta_vza['view_azimuth_adj'].mean(), 1)) + "_vz" + str(np.round(meta_vza['view_zenith'].mean(), 1)) + "_sa" + str(np.round(meta_vza['solar_azimuth_adj'].mean(), 1)) + "_" + city + "_output.tif"
    # output
    with rio.open(fn_out, 'w', **out_meta) as dst:
        dst.write(np.stack([targeto, predo, resido]))

# =============================================================================
# extract to shapefile
# =============================================================================
    out_shp = []
    out_stats = ['mean']
    for i in range(3):
        out_shpi = rs.zonal_stats(shp, 
                                  rio.open(fn_out).read()[i], 
                                  affine = rio.open(fn_out).transform, 
                                  stats = out_stats)
        for s in range(len(out_stats)):
            out_shpis = [np.asarray([x[attribute] for x in out_shpi]) for attribute in [out_stats[s]]][0]
            out_shpis[out_shpis == None] = np.nan # tag unfilled as nan
            out_shp.append(out_shpis.astype(float))
    
    # convert to array
    out_shp = pd.concat([shp, pd.DataFrame(np.array(out_shp).T, columns = ['target', 'predict', 'residual'])], axis = 1)
    #out_shp.plot(column = 0, legend = True)
    # output
    out_shp.to_file("../data/rf_output/" + t + "_va" + str(np.round(meta_vza['view_azimuth_adj'].mean(), 1)) + "_vz" + str(np.round(meta_vza['view_zenith'].mean(), 1)) + "_sa" + str(np.round(meta_vza['solar_azimuth_adj'].mean(), 1)) + "_" + city + "_output.shp")
