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


def plot_NorthPolarStereo(product, var,lon, lat, min_value,max_value,method,ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    plt.axes(projection=ccrs.NorthPolarStereo(
        true_scale_latitude=70))
    ecco.plot_proj_to_latlon_grid(lon, lat,
                                  var,
                                  projection_type='stereo',
                                  plot_type='contourf',
                                  show_colorbar=True,
                                  cmap='jet',
                                  dx=1, dy=1, cmin=min_value, cmax=max_value,
                                  lat_lim=50)     
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_timeseries(start_time, end_time, lon, lat, product, variable, write_csv, **plotargs):
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
    fig, ax = plt.subplots(**plotargs)
    df.plot(ax=ax)
    ax.grid()
    ax.set_ylabel('['+units+']', fontsize=14)
    ax.set_title(product+',lon.='
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
            ax = plot_NorthPolarStereo(product=product, 
                                  var=var.loc[date_list[i]],lon=lon, lat=lat, 
                                  min_value=min_value,max_value=max_value,
                                  method=method, ax=ax)
            plt.title(product+','+str(var.time.values).split(':')[0]+'UTC')
            plt.savefig(variable+str(var.time.values).split(':')[0]+'.png', bbox_inches='tight')
            plt.close()
    elif method == 'mean':
        ax = plot_NorthPolarStereo(product=product, 
                                  var=var.mean('time'),lon=lon, lat=lat, 
                                  min_value=var.mean('time').min(),
                                  max_value=var.mean('time').max(),
                                  method=method)
        plt.title(product+',Mean:'+start_time+'--'+end_time)
        plt.savefig(variable+'_Mean_'+start_time+'-'+end_time+'.png', bbox_inches='tight')
        plt.close()

def get_url(product, day):
    if product == 'NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/' + \
            day.strftime('%Y') + '/'+day.strftime('%m') + \
            '/SPC'+day.strftime('%Y%m%d')+'00.nc'
    return url


def plot_2D_spectra(start_time, end_time, lon, lat, product):
    data = []
    date_list = pd.date_range(start=start_time, end=end_time, freq='D')
    for k in range(len(date_list)):  # loop over days
        url = get_url(product, date_list[k])
        ds = xr.open_dataset(url)
        # Find the nearest grid point.
        abslat = np.abs(ds.latitude-lat)
        abslon = np.abs(ds.longitude-lon)
        c = np.maximum(abslon, abslat)
        ([xloc], [yloc]) = np.where(c == np.min(c))
        print('Nearest point lon, lat:'
              + str(ds.longitude.values[xloc, yloc])+','+str(ds.latitude.values[xloc, yloc]))
        data.append(ds.SPEC[:, xloc, yloc, :, :])
        SPEC = xr.concat(data, dim='time')
    for i in range(SPEC.time.shape[0]):  # loop over SPEC time
        z = SPEC[i, :, :].values
        nrows, ncols = z.shape
        y = ds.freq.values
        x = ds.direction.values
        ([x_peak], [y_peak]) = np.where(z == np.max(z))
        x, y = np.meshgrid(x, y)
        fig = plt.figure(figsize=(8, 6))
        ax = plt.subplot(111, projection='3d')
        ax.xaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.fill = False
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.fill = False
        ax.zaxis.pane.set_edgecolor('white')
        ax.grid(False)
        ax.w_zaxis.line.set_lw(0.)
        ax.set_zticks([])
        plot = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='white',
                               vmin=SPEC.min(), vmax=SPEC.max(), alpha=0.8)
        ceb = ax.contourf(x, y, z, zdir='x', vmin=SPEC.min(),
                          vmax=SPEC.max(), cmap='viridis', offset=0)
        ceb = ax.contourf(x, y, z, zdir='y', vmin=SPEC.min(),
                          vmax=SPEC.max(), cmap='viridis', offset=0.55)
        ax.set_zlim(SPEC.min(), SPEC.max())
        cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel('m**2 s')
        ax.set_ylabel(r'$\mathregular{f[Hz]}$', labelpad=20)
        ax.set_xlabel(r'$\mathregular{\theta}[deg]$', labelpad=20)
        ax.set_title(product+"\n"+'lon.='+str(ds.longitude.values[xloc, yloc])+',lat='+str(ds.latitude.values[xloc, yloc])
                     + "\n"+str(SPEC['time'].values[i]).split(':')[0]
                     + "\n"
                             + r'$\theta_{p}$='
                     + str(ds.direction.values[x_peak].round(2))+'deg'
                     + "\n"+r'$T_{p}$='+str((1/ds.freq.values[y_peak]).round(2))+'s', fontsize=16)
        plt.savefig('SPEC_lon'+str(ds.longitude.values[xloc, yloc])+'lat'+str(ds.latitude.values[xloc, yloc])
                    + str(SPEC['time'].values[i]).split(':')[0]+'.png', bbox_inches='tight')
        plt.close()


def plot_topaz(start_time, end_time, variable, method):
    url = 'https://thredds.met.no/thredds/dodsC/topaz/dataset-ran-arc-day-myoceanv2-be'
    date_list = pd.date_range(start=start_time, end=end_time, freq='D')
    product = 'Arctic Ocean Physics Reanalysis'
    if variable == 'ice_speed':
        var = xr.Dataset({"ice_speed": ((xr.open_dataset(url)[
                         'uice'].sel(time=slice(start_time, end_time)))**2 + (xr.open_dataset(url)['vice'].sel(time=slice(start_time, end_time)))**2)**0.5})
        var = var[variable]
        var = var.assign_attrs(units='m s-1')
    else:
        var = xr.open_dataset(url)[variable].sel(time=slice(start_time, end_time))
        
    if method == 'timestep':
        min_value = var.min()
        max_value = var.max()
        print(max_value)
        for i in range(len(date_list)):
            print(date_list[i])
            plot_NorthPolarStereo(product=product, 
                                  var=var.loc[date_list[i]],lon=var.longitude,
                                  lat=var.latitude, 
                                  min_value=min_value,max_value=max_value,
                                  method=method)
            plt.title(product+'\n '+str(date_list[i]))
            plt.savefig(variable + str(date_list[i]) +'.png', bbox_inches='tight')
            plt.close
    elif method == 'mean':
        plot_NorthPolarStereo(product=product, 
                                  var=var.mean('time'),lon=var.longitude,
                                  lat=var.latitude, 
                                  min_value=var.mean('time').min(),
                                  max_value=var.mean('time').max(),
                                  method=method)
        plt.title(product+'\n Mean:'+start_time+'--'+end_time)
        plt.savefig(variable+'_Mean_'+start_time+'-'+end_time+'.png', bbox_inches='tight')
        plt.close()
        

