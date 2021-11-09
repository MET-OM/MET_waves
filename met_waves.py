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
import numpy as np


def estimate_WEF(hs, tp):
    WEF = 0.5*(hs**2)*(0.85*tp)
    return WEF


def estimate_WPD(wnd):
    WPD = 0.5*1.225*(wnd**3)
    return WPD


def distance_2points(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c  # in km
    return distance


def find_nearest(lon_model, lat_model, lat0, lon0):
    #print('find nearest point...')
    dx = distance_2points(lat0, lon0, lat_model, lon_model)
    rlat0 = dx.where(dx == dx.min(), drop=True).rlat
    rlon0 = dx.where(dx == dx.min(), drop=True).rlon
    return rlon0, rlat0


def plot_timeseries(start_time, end_time, lon, lat, product, variable):
    if product == 'NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    elif product == 'WAM4':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_be'
    elif product == 'WAM4C47':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_c47_be'
    date_list = pd.date_range(start=start_time, end=end_time, freq='H')
    #var = xr.open_dataset(url)[variable].sel(time=slice(start_time, end_time))
    lon_model = xr.open_dataset(url).longitude
    lat_model = xr.open_dataset(url).latitude
    var = xr.open_dataset(url)[variable]
    var = var.sel(time=~var.get_index("time").duplicated())
    var = var.sel(time=slice(start_time, end_time))
    rlon, rlat = find_nearest(lon_model, lat_model, lat, lon)
    var = var.sel(rlat=rlat, rlon=rlon)
    fig, ax = plt.subplots()
    var.plot()
    plt.grid()
    plt.title('Coordinates:'+str(lon)+','+str(lat), fontsize=16)
    plt.savefig(variable+'_ts_'+start_time + '-'
                + end_time+'.png', bbox_inches='tight')


def plot_panarctic_map(start_time, end_time, product, variable, method):
    # Plots in a panarctic map a given variable
    # start_time = start date for plotting e.g., '2005-01-07T18'
    # end_time   = end date for plotting e.g., '2005-01-09T18'
    # Product: 'nora3' or 'wam4'
    # variable: e.g., 'hs', 'tp', 'ff' for wind
    # method: 'timestep' for plotting all timesteps for given period or 'mean'
    # Overview of the NORAE3 wave variables is given in:
    # https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html
    if product == 'NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    elif product == 'WAM4':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_be'
    elif product == 'WAM4C47':
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
            plt.title(product+','+str(var.time.values[i]).split(':')[0]+'UTC')
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
        plt.title(product+','+'Mean:'+start_time + '--' + end_time)
        plt.savefig(variable+'avg'+start_time + '-'
                    + end_time+'.png', bbox_inches='tight')
        plt.close()
