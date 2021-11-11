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
import time


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


def url_agg(product):
    if product == 'NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml'
    elif product == 'WAM4':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_be'
    elif product == 'WAM4C47':
        url = 'https://thredds.met.no/thredds/dodsC/sea/mywavewam4/mywavewam4_c47_be'
    return url


def plot_timeseries(start_time, end_time, lon, lat, product, variable, write_csv):
    start = time.time()
    date_list = pd.date_range(start=start_time, end=end_time, freq='H')
    df = pd.DataFrame({'time': date_list, variable: np.zeros(len(date_list))})
    df = df.set_index('time')
    url = url_agg(product=product)
    ds = xr.open_dataset(url)
    units = ds[variable].units
    print('Find nearest point to lon.='+str(lon)+','+'lat.='+str(lat))
    rlon, rlat = find_nearest(ds.longitude, ds.latitude, lat, lon)
    lon_near = ds.longitude.sel(rlat=rlat, rlon=rlon).values[0][0]
    lat_near = ds.latitude.sel(rlat=rlat, rlon=rlon).values[0][0]
    print('Found nearest: lon.='+str(lon_near)+',lat=' + str(lat_near))
    print('Extract time series...:'+str(start_time)+'--->'+str(end_time))
    for i in range(len(date_list)):
        df[variable][i] = ds[variable].sel(
            rlat=rlat, rlon=rlon).loc[date_list[i]].values[0][0]
    if write_csv is True:
        print('Write data to file...')
        df.to_csv(variable+'_'+product+'_lon'
                  + str(lon_near)+'_lat'+str(lat_near)+start_time + '-'
                  + end_time+'.csv')
    else:
        pass
    print('Plot time series...')
    fig, ax = plt.subplots()
    df.plot()
    plt.grid()
    plt.ylabel('['+units+']', fontsize=14)
    plt.title(product+',lon.='
              + str(lon_near)+',lat.='+str(lat_near), fontsize=16)
    plt.savefig(variable+'_'+product+'_lon'
                + str(lon_near)+'_lat'+str(lat_near)+start_time + '-'
                + end_time+'.png', bbox_inches='tight')
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    return


def plot_panarctic_map(start_time, end_time, product, variable, method):
    """
    Plots in a panarctic map a given variable
    start_time = start date for plotting e.g., '2005-01-07T18'
    end_time   = end date for plotting e.g., '2005-01-09T18'
    Product: 'nora3' or 'wam4'
    variable: e.g., 'hs', 'tp', 'ff' for wind
    method: 'timestep' for plotting all timesteps for given period or 'mean'
    Overview of the NORAE3 wave variables is given in:
    https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html
    """
    url = url_agg(product=product)
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


def get_url(day):
    url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/' + \
        day.strftime('%Y') + '/'+day.strftime('%m') + \
        '/SPC'+day.strftime('%Y%m%d')+'00.nc'
    return url
