#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:58:23 2021

@author: KonstantinosChr
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import ecco_v4_py as ecco



def plot_panarctic_map_timestep(start_time,end_time,variable, min_value, max_value):
    # The function plots/saves in png-format a given variable for all timesteps between start_time and end_time
    # start_time = start date for plotting e.g., '2005-01-07T18'
    # end_time   = end date for plotting e.g., '2005-01-09T18'
    # min_value = mininum variable value
    # max_value = maximum variable value 
    # Overview of the NORAE3 wave variables is given in: 
    # https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html
    
    url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    ds = xr.open_dataset(url).sel(time=slice(start_time, end_time))
    for i in range(len(ds.time)):
        print(ds.time.values[i])
        fig, ax = plt.subplots()
        plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
        ecco.plot_proj_to_latlon_grid(ds.longitude, ds.latitude, \
                                  ds[variable].loc[ds.time[i]], \
                                  projection_type='stereo',\
                                  plot_type = 'contourf', \
                                  show_colorbar=True,
                                  cmap = 'jet',
                                  dx=1, dy=1,cmin=min_value, cmax=max_value,\
                                  lat_lim=50);    
        plt.title(str(ds.time.values[i]).split(':')[0]+'UTC')
        plt.savefig(variable+str(ds.time.values[i]).split(':')[0]+'.png',bbox_inches = 'tight')
        plt.close()


