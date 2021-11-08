#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 08:58:23 2021

@author: KonstantinChri
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import ecco_v4_py as ecco
import pandas as pd
from subprocess import Popen


def plot_panarctic_map_mean(start_time, end_time, variable):
    # The function plots/saves in png-format the mean of a given variable between start_time and end_time
    # start_time = start date for plotting e.g., '2005-01-07T18'
    # end_time   = end date for plotting e.g., '2005-01-09T18'
    # Overview of the NORAE3 wave variables is given in:
    # https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html

    url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    ds = xr.open_dataset(url).sel(time=slice(start_time, end_time))
    min_value = ds[variable].mean('time').min()
    max_value = ds[variable].mean('time').max()
    fig, ax = plt.subplots()
    plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
    ecco.plot_proj_to_latlon_grid(ds.longitude, ds.latitude,
                                  ds[variable].mean('time'),
                                  projection_type='stereo',
                                  plot_type='contourf',
                                  show_colorbar=True,
                                  cmap='jet',
                                  dx=1, dy=1, cmin=min_value, cmax=max_value,
                                  lat_lim=50)
    plt.title('Mean:'+start_time + '--' + end_time)
    plt.savefig(variable+'avg'+start_time + '-'
                + end_time+'.png', bbox_inches='tight')
    plt.close()


def estimate_WEF(hs, tp):
    WEF = 0.5*(hs**2)*(0.85*tp)
    return WEF


def estimate_WPD(wnd):
    WPD = 0.5*1.225*(wnd**3)
    return WPD


def plot_panarctic_map(start_time, end_time, product, variable, method):
    # Plots in a panarctic map a given variable
    # start_time = start date for plotting e.g., '2005-01-07T18'
    # end_time   = end date for plotting e.g., '2005-01-09T18'
    # Product: 'nora3' or 'wam4'
    # variable: e.g., 'hs', 'tp', 'ff' for wind
    # method: 'timestep' for plotting all timesteps for given period or 'mean'
    # Overview of the NORAE3 wave variables is given in:
    # https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html
    if product == 'nora3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    elif product == 'wam4':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_be'
    elif product == 'wam4c47':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_c47_be'
    date_list = pd.date_range(start=start_time, end=end_time, freq='H')
    #var = xr.open_dataset(url)[variable].sel(time=slice(start_time, end_time))
    var = xr.open_dataset(url)[variable]
    var = var.sel(time=~var.get_index("time").duplicated())
    var = var.sel(time=slice(start_time, end_time))
    lon = xr.open_dataset(url).longitude
    lat = xr.open_dataset(url).latitude
    if method == 'timestep':
        min_value = var.min()
        max_value = var.max()
        print(max_value)
        for i in range(len(date_list)):
            print(date_list[i])
            fig, ax = plt.subplots()
            plt.axes(projection=ccrs.NorthPolarStereo(
                true_scale_latitude=70))
            ecco.plot_proj_to_latlon_grid(lon, lat,
                                          var.loc[date_list[i]],
                                          projection_type='stereo',
                                          plot_type='contourf',
                                          show_colorbar=True,
                                          cmap='jet',
                                          dx=1, dy=1, cmin=min_value, cmax=max_value,
                                          lat_lim=50)
            plt.title(str(var.time.values[i]).split(':')[0]+'UTC')
            plt.savefig(
                variable+str(var.time.values[i]).split(':')[0]+'.png', bbox_inches='tight')
            plt.close()
    elif method == 'mean':
        fig, ax = plt.subplots()
        plt.axes(projection=ccrs.NorthPolarStereo(true_scale_latitude=70))
        ecco.plot_proj_to_latlon_grid(lon, lat,
                                      var.mean('time'),
                                      projection_type='stereo',
                                      plot_type='contourf',
                                      show_colorbar=True,
                                      cmap='jet',
                                      dx=1, dy=1, cmin=var.mean('time').min(),
                                      cmax=var.mean('time').max(),
                                      lat_lim=50)
        plt.title('Mean:'+start_time + '--' + end_time)
        plt.savefig(variable+'avg'+start_time + '-'
                    + end_time+'.png', bbox_inches='tight')
        plt.close()
