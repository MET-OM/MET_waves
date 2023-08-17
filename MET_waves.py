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
import os
from nco import Nco
from cdo import Cdo



def estimate_WEF(hs, tp):
    WEF = 0.5*(hs**2)*(0.85*tp) # KW/m
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

def plot_grid_spec_points(url,s=0.5,color='red'):
    """
    Plot all grid points where spectrum is available in NORA3/WAM4/WW3 datasets (https://thredds.met.no)
    url : link or nc-file for NORA3/WAM4/WW3 dataset
    e.g., url for NORA3: https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/2020/12/SPC2020123100.nc
          url for WAM4:  https://thredds.met.no/thredds/dodsC/fou-hi/mywavewam4archive/2022/05/24/MyWave_wam4_SPC_20220524T00Z.nc
          url for WW3:   https://thredds.met.no/thredds/dodsC/ww3_4km_latest_files/ww3_POI_SPC_20220524T12Z.nc
    s : marker size
    color: marker color
    """
    data = xr.open_dataset(url)
    fig, ax = plt.subplots()
    ax = plt.axes(projection=ccrs.Orthographic(-10, 45))
    ax.stock_img()
    #ax.add_feature(cfeature.LAND,color='darkkhaki')
    #ax.add_feature(cfeature.LAKES)
    #ax.coastlines(resolution='50m', color='darkkhaki', linewidth=0.5)
    ax.scatter(data.longitude,data.latitude,s=s, marker='.', color=color, transform=ccrs.PlateCarree())
    plt.title('Grid Points (2D Spectrum)',fontsize=18)
    plt.show()
    plt.savefig('spec_points.png',dpi=300)
    return fig, ax


def plot_NorthPolarStereo(product, var,lon, lat, min_value,max_value,cmap,ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    plt.axes(projection=ccrs.NorthPolarStereo(
        true_scale_latitude=70))
    ecco.plot_proj_to_latlon_grid(lon, lat,
                                  var,
                                  projection_type='stereo',
                                  plot_type='contourf',
                                  show_colorbar=True,
                                  cmap=cmap,
                                  dx=1, dy=1, cmin=min_value, cmax=max_value,
                                  lat_lim=50)
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_timeseries(start_time, end_time, lon, lat, product, variable, write_csv, ts_obs, **plotargs):
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
    if ts_obs is not None:
        ts_obs.plot(ax=ax)
        ax.legend([product,'obs.'])
    else:
        ax.legend([product])
    ax.grid()
    ax.set_ylabel(variable+' ['+units+']', fontsize=14)
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


def plot_panarctic_map(start_time, end_time, product, variable, method, cmap='jet'):
    """
    Plots in a panarctic map a given variable
    start_time = start date for plotting e.g., '2005-01-07T18'
    end_time   = end date for plotting e.g., '2005-01-09T18'
    Product: 'nora3' / 'wam4'
    variable: e.g., 'hs', 'tp', 'ff' for wind
    method: 'timestep' for plotting all timesteps for given period or 'mean'
    Overview of the NORAE3 wave variables is given in:
    https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_agg/wam3kmhindcastaggregated.ncml.html
    """
    url = url_agg(product=product)
    date_list = pd.date_range(start=start_time, end=end_time, freq='H')
    #var = xr.open_dataset(url)[variable].sel(time=slice(start_time, end_time))
    if variable == 'WEF':
       var = estimate_WEF(xr.open_dataset(url)['hs'].sel(time=slice(start_time, end_time)), xr.open_dataset(url)['tp'].sel(time=slice(start_time, end_time)))
    else:
    	var = xr.open_dataset(url)[variable]
    	var = var.sel(time=~var.get_index("time").duplicated())
    	var = var.sel(time=slice(start_time, end_time))
    lon = xr.open_dataset(url).longitude
    lat = xr.open_dataset(url).latitude
    if method == 'timestep':
        min_value = var.min()
        max_value = var.max()
        #print(max_value)
        for i in range(len(date_list)):
            print(date_list[i])
            fig, ax = plt.subplots()
            ax = plot_NorthPolarStereo(product=product,
                                  var=var.loc[date_list[i]],lon=lon, lat=lat,
                                  min_value=min_value,max_value=max_value,
                                  cmap=cmap, ax=ax)
            plt.title(variable+'['+var.units+']'+','+product+','+str(date_list[i])+'UTC')
            plt.savefig(variable+str(date_list[i])+'.png', bbox_inches='tight')
            plt.close()
    elif method == 'mean':
        ax = plot_NorthPolarStereo(product=product,
                                  var=var.mean('time'),lon=lon, lat=lat,
                                  min_value=var.mean('time').min(),
                                  max_value=var.mean('time').max(),
                                  cmap=cmap)
        plt.title(product+',Mean:'+start_time+'--'+end_time)
        plt.savefig(variable+'_Mean_'+start_time+'-'+end_time+'.png', bbox_inches='tight',dpi=300)
        plt.close()


def get_url(product, day):
    if product == 'SPEC_NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_spectra/' + \
            day.strftime('%Y') + '/'+day.strftime('%m') + \
            '/SPC'+day.strftime('%Y%m%d')+'00.nc'
    elif product == 'SPEC_WW3':
        url = 'https://thredds.met.no/thredds/dodsC/ww3_4km_latest_files/' + \
            'ww3_POI_SPC_'+day.strftime('%Y%m%d')+'T00Z.nc'
    elif product == 'NORA3':
        url = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/' + \
            day.strftime('%Y') + '/'+day.strftime('%m') +  '/' + \
            day.strftime('%Y%m%d')+'_MyWam3km_hindcast.nc'
    return url


def plot_2D_spectra(start_time, end_time, lon, lat, product):
    """
    Plot 2D spectra for a given time period
    start_time : start date for plotting e.g., '2005-01-07T18'
    end_time   : end date for plotting e.g., '2005-01-09T18'
    lon : longtitude
    lat   latitude
    product: 'SPEC_NORA3' / 'SPEC_WW3'
    """
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
        ([y_peak], [x_peak]) = np.where(z == np.max(z))
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
        ax.contourf(x, y, z, zdir='x', vmin=SPEC.min(),
                          vmax=SPEC.max(), cmap='viridis', offset=0)
        ax.contourf(x, y, z, zdir='y', vmin=SPEC.min(),
                          vmax=SPEC.max(), cmap='viridis', offset=0.55)
        ax.set_zlim(SPEC.min(), SPEC.max())
        cbar = fig.colorbar(plot, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel('m**2 s')
        ax.set_ylabel(r'$\mathregular{f[Hz]}$', labelpad=20)
        ax.set_xlabel(r'$\mathregular{\theta}[deg]$', labelpad=20)
        ax.set_title(product+"\n"+'lon.='+str(ds.longitude.values[xloc, yloc])+',lat='+str(ds.latitude.values[xloc, yloc])
                     + "\n"+str(SPEC['time'].values[i]).split(':')[0]
                     #+ "\n"
                     #        + r'$\theta_{p}$='
                     #+ str(ds.direction.values[x_peak].round(2))+'deg'
                     #+ "\n"+r'$T_{p}$='+str((1/ds.freq.values[y_peak]).round(2))+'s'
                     ,fontsize=16)
        plt.savefig('SPEC_lon'+str(ds.longitude.values[xloc, yloc])+'lat'+str(ds.latitude.values[xloc, yloc])
                    + str(SPEC['time'].values[i]).split(':')[0]+'.png', bbox_inches='tight')
        plt.close()


def extract_2D_spectra(start_time, end_time, lon, lat, product):
    """
    Extract and Save (in netcdf format) 2D spectra for a given time period
    start_time : start date for plotting e.g., '2005-01-07T18'
    end_time   : end date for plotting e.g., '2005-01-09T18'
    lon : longtitude
    lat   latitude
    product: 'SPEC_NORA3' / 'SPEC_WW3'
    """
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
    SPEC = SPEC.sel(time=slice(start_time, end_time))
    SPEC.to_netcdf('SPEC_lon'+str(ds.longitude.values[xloc, yloc])+'_lat'+str(ds.latitude.values[xloc, yloc])
                    +'_'+ str(SPEC['time'].values[0]).split(':')[0]+'_'+ str(SPEC['time'].values[-1]).split(':')[0]+'.nc')

    return SPEC


def plot_topaz(start_time, end_time, variable, method,save_data):
    """
    info about the hindcast: https://thredds.met.no/thredds/myocean/ARC-MFC/arc-topaz-ran-arc.html
    """
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
                                  min_value=min_value,max_value=max_value,cmap='ocean')
            plt.title(product+'\n '+str(date_list[i]))
            plt.savefig(variable + str(date_list[i]) +'.png', bbox_inches='tight')
            plt.close
        if save_data == True:
            var.to_netcdf(variable+'_'+start_time+'-'+end_time+'.nc')
    elif method == 'mean':
        plot_NorthPolarStereo(product=product,
                                  var=var.mean('time'),lon=var.longitude,
                                  lat=var.latitude,
                                  min_value=var.mean('time').min(),
                                  max_value=var.mean('time').max(),cmap='ocean')
        plt.title(product+'\n Mean:'+start_time+'--'+end_time)
        plt.savefig(variable+'_Mean_'+start_time+'-'+end_time+'.png', bbox_inches='tight')
        plt.close()
        if save_data == True:
            var.mean('time').to_netcdf(variable+'_Mean_'+start_time+'-'+end_time+'.nc')




def extract_ts_point(start_date,end_date,variable, lon, lat, product ='NORA3'):
    """
    Extract times series of  the nearest gird point (lon,lat) from
    nora3 wave hindcast and save it as netcdf.
    """
    nco = Nco()
    date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    outfile = 'lon'+str(lon)+'_'+'lat'+str(lat)+'_'+date_list.strftime('%Y%m%d')[0]+'_'+date_list.strftime('%Y%m%d')[-1]+'.nc'

    if os.path.exists(outfile):
        os.remove(outfile)
        print(outfile, 'already exists, so it will be deleted and create a new....')

    else:
        print("....")


    tempfile = [None] *len(date_list)
    # Create directory
    dirName = 'temp'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    # extract point and create temp files
    for i in range(len(date_list)):
        tempfile[i] = 'temp/temp'+date_list.strftime('%Y%m%d')[i]+'.nc'
        if product == 'NORA3':
             infile = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/'+date_list.strftime('%Y')[i]+'/'+date_list.strftime('%m')[i]+'/'+date_list.strftime('%Y%m%d')[i]+'_MyWam3km_hindcast.nc'
             #infile_http = 'https://thredds.met.no/thredds/fileServer/windsurfer/mywavewam3km_files/'+date_list.strftime('%Y')[i]+'/'+date_list.strftime('%m')[i]+'/'+date_list.strftime('%Y%m%d')[i]+'_MyWam3km_hindcast.nc'
             print(infile)
             if i==0:
                 ds = xr.open_dataset(infile)
                 print('Find nearest point to lon.='+str(lon)+','+'lat.='+str(lat))
                 rlon, rlat = find_nearest(ds.longitude, ds.latitude, lat, lon)
                 lon_near = ds.longitude.sel(rlat=rlat, rlon=rlon).values[0][0]
                 lat_near = ds.latitude.sel(rlat=rlat, rlon=rlon).values[0][0]
                 print('Found nearest: lon.='+str(lon_near)+',lat.=' + str(lat_near))

        opt = ['-O -v '+",".join(variable)+' -d rlon,'+str(rlon.values[0])+' -d rlat,'+str(rlat.values[0])]
        for x in range(0, 6):  # try 6 times
            try:
                nco.ncks(input=infile , output=tempfile[i], options=opt)
            except:
                print('......Retry'+str(x)+'.....')
                time.sleep(10)  # wait for 10 seconds before re-trying
            else:
                break

    #merge temp files
    nco.ncrcat(input=tempfile, output=outfile)

    # add lon, lat info in nc-file
    ds = xr.open_dataset(outfile)
    ds['lon'] =  lon_near
    ds['lat'] =  lat_near
    ds.to_netcdf(outfile)
    
    #remove temp files
    for i in range(len(date_list)):
        os.remove(tempfile[i])

    return

def extract_variable(start_date,end_date,variable,mean_method, product ='NORA3'):
    """
    Extract a variable for the whole NORA3 domain and save it as netcdf.
    """
    nco = Nco()
    date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    outfile = "-".join(variable)+'_'+date_list.strftime('%Y%m%d')[0]+'_'+date_list.strftime('%Y%m%d')[-1]+'.nc'

    if os.path.exists(outfile):
        os.remove(outfile)
        print(outfile, 'already exists, so it will be deleted and create a new....')

    else:
        print("....")


    tempfile = [None] *len(date_list)
    # Create directory
    dirName = 'temp'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    # extract point and create temp files
    for i in range(len(date_list)):
        tempfile[i] = 'temp/temp'+date_list.strftime('%Y%m%d')[i]+'.nc'
        if product == 'NORA3':
             infile = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/'+date_list.strftime('%Y')[i]+'/'+date_list.strftime('%m')[i]+'/'+date_list.strftime('%Y%m%d')[i]+'_MyWam3km_hindcast.nc'
             print(infile)


        for x in range(0, 6):  # try 6 times
            try:
                if mean_method == None:
                    #opt = ['-O -v '+",".join(variable)]
                    #nco.ncks(input=infile , output=tempfile[i], options=opt)
                    if variable[0] == 'WEF':
                        opt = ['-s '+"WEF=0.5*hs*hs*tmp"]
                        nco.ncap2(input=infile , output=tempfile[i], options=opt)
                elif mean_method == 'daily':
                    opt = ['-O -v '+",".join(variable)+' -d time,,,24,24']
                    nco.ncra(input=infile , output=tempfile[i], options=opt)
            except:
                print('......Retry'+str(x)+'.....')
                time.sleep(10)  # wait for 10 seconds before re-trying
            else:
                break

    #merge temp files
    nco.ncrcat(input=tempfile, output=outfile)

    #remove temp files
    for i in range(len(date_list)):
        os.remove(tempfile[i])

    return

def plot_swan_spec2D(start_time, end_time,infile, site=0):
    from wavespectra import read_ncswan
    ds = read_ncswan(infile).sel(time=slice(start_time, end_time))
    hs_spec = ds.isel(site=0).efth.spec.hs()
    # tp_spec = ds.isel(site=0).efth.spec.tp()
    # dp_spec = ds.isel(site=0).efth.spec.dp()
    vmax=ds.efth.max()
    vmin=ds.efth.min()
    for i in range(ds.time.shape[0]):
        ax = plt.subplot(111, polar=True)
        ds.isel(site=site,time=i).efth.spec.split(fmin=0.04).spec.plot(kind="contourf", cmap="ocean_r", as_period=True,normalised=True)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.title('$'+'lon.:'+str(ds.lon.values[0].round(2))+','+'lat.:'+str(ds.lat.values[0].round(2))+','+str(ds.time.values[i]).split(':')[0]+', H_{m0}:'+str(hs_spec.values[i].round(1))+
                  'm'+'$')
        plt.savefig(str(ds.time.values[i]).split(':')[0]+'_site'+str(site)+'.png',dpi=300)
        plt.close()



def download_WEF(start_date,end_date, product ='NORA3'):
    """
    Create WEF from NORA3 data and save it as netcdf (using cdo).
    """
    cdo = Cdo()
    date_list = pd.date_range(start=start_date , end=end_date, freq='D')
    outfile = "WEF"+'_'+date_list.strftime('%Y%m%d')[0]+'_'+date_list.strftime('%Y%m%d')[-1]+'.nc'

    if os.path.exists(outfile):
        os.remove(outfile)
        print(outfile, 'already exists, so it will be deleted and create a new....')

    else:
        print("....")


    outfile = [None] *len(date_list)
    # Create directory
    dirName = 'download'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    except FileExistsError:
        print("Directory " , dirName ,  " already exists")

    # extract point and create temp files
    for i in range(len(date_list)):
        outfile[i] = 'download/WEF'+date_list.strftime('%Y%m%d')[i]+'.nc'
        if product == 'NORA3':
             infile = 'https://thredds.met.no/thredds/dodsC/windsurfer/mywavewam3km_files/'+date_list.strftime('%Y')[i]+'/'+date_list.strftime('%m')[i]+'/'+date_list.strftime('%Y%m%d')[i]+'_MyWam3km_hindcast.nc'
             print(infile)


        for x in range(0, 6):  # try 6 times
            try:
                #cdo_command = ['cdo', 'expr,WEF=0.5*hs*hs*tmp', infile, ]
                #call(cdo_command)
                cdo.expr("'WEF=0.5*hs*hs*tmp'",input=infile, output=outfile[i], options="-f nc")
            except:
                print('......Retry'+str(x)+'.....')
                time.sleep(10)  # wait for 10 seconds before re-trying
            else:
                break

    return

def plot_MET_forecast(url):
    "url: opendap link of WW3_4km or WAM800 models from thredds.met.no (MET Norway)"
    ds = xr.open_dataset(url)
    print(url)
    cut=8
    print('Hs',ds.hs.max())
    cmap_waves = create_cmap(type='wave')

    for i in range(ds.time.shape[0]):
        fig,ax = plt.subplots(figsize=(8,6), subplot_kw={"projection": ccrs.Orthographic(-15,60)})
        levels = np.round(np.linspace(0,int(ds.hs.max()+1),int(ds.hs.max())*10),1)
        im = ax.contourf(ds.longitude, ds.latitude,ds.hs.loc[ds.time[i]],levels = levels, transform = ccrs.PlateCarree(),cmap=cmap_waves) #ocean_r, coolwarm
        plt.title(ds.title+' '+str(ds.time.values[i])[:13] +'UTC')
        #ax.coastlines('50m')
        #ax.stock_img()
        ax.set_facecolor('lightgrey')
        cbar_ax = fig.add_axes([0.8, 0.12, 0.05, 0.7])
        cbar_ax.set_title('$'+'H_s[m]'+'$')
        fig.colorbar(im, cax=cbar_ax)
        ax.gridlines(xlocs=range(-180,180,10),ylocs=range(-90, 90, 10),color='black',linestyle='dotted',draw_labels=True)
        plt.savefig(ds.title.split(' ')[0] +'_' + str(ds.time.values[i])[:13]+'_Hs.png',bbox_inches = 'tight')
        plt.close()

def plot_SWAN_map(url='Ianos500m_st6_20200914.nc',var='hs', start_time='2020-09-14T12', end_time='2020-09-20T16', point=None, cmap='jet'):
    ds = xr.open_dataset(url).sel(time=slice(start_time, end_time))
    if var=='wind':
        ds['wind'] = (ds['xwnd']**2 + ds['ywnd']**2)**(1/2)
        ds['wind'].attrs['units'] = 'm/s'
    print('max '+var,ds[var].max())
    for i in range(ds.time.shape[0]):
        fig, ax = plt.subplots()
        levels = np.round(np.linspace(0,int(ds[var].max()+1),int(ds[var].max()+1)*10),1)
        im = ax.contourf(ds.longitude, ds.latitude,ds[var].loc[ds.time[i]],levels = levels,cmap=cmap) #, transform = ccrs.PlateCarree(),cmap='coolwarm') # coolwarm  
        if var=='wind': 
            ax.quiver(ds.longitude[::50], ds.latitude[::50],ds['xwnd'].loc[ds.time[i]][::50,::50],ds['ywnd'].loc[ds.time[i]][::50,::50],linewidths=0.01)       
        if point is not None:
            for j in range(len(point)):
                p = point[j]
                plt.plot(p[0],p[1],marker ='^',color='magenta', markersize=8)
        plt.title('DNORA/SWAN,'+str(ds.time.values[i])[:13] +'UTC')
        cbar_ax = plt.colorbar(im) 
        cbar_ax.ax.set_title('$['+ds[var].units+']$')
        ax.set_facecolor('beige')
        plt.savefig(str(ds.time.values[i])[:13]+'_'+var+'.png',bbox_inches = 'tight',dpi=300)
        plt.close()

def new_func():
    return 50

def create_cmap(type = 'wave'):
    from matplotlib.colors import ListedColormap
    from skimage.transform import resize
    if type == 'wind':
        colors0 = np.array([
           [0.62352941, 0.7254902 , 0.74901961, 1.        ],
           [0.18823529, 0.61568627, 0.7254902 , 1.        ],
           [0.21960784, 0.40784314, 0.74901961, 1.        ],
           [0.3372549 , 0.35294118, 0.77254902, 1.        ],
           [0.18823529, 0.38431373, 0.55294118, 1.        ],
           [0.18823529, 0.38431373, 0.55294118, 1.        ],
           [0.        , 0.46666667, 0.43921569, 1.        ],
           [0.        , 0.46666667, 0.43921569, 1.        ],
           [0.        , 0.6       , 0.47843137, 1.        ],
           [0.        , 0.6       , 0.47843137, 1.        ],
           [0.        , 0.68235294, 0.40784314, 1.        ],
           [0.        , 0.68235294, 0.40784314, 1.        ],
           [0.65882353, 0.78823529, 0.26666667, 1.        ],
           [0.65882353, 0.78823529, 0.26666667, 1.        ],
           [0.78431373, 0.25882353, 0.05098039, 1.        ],
           [0.78431373, 0.25882353, 0.05098039, 1.        ],
           [0.84313725, 0.        , 0.19607843, 1.        ],
           [0.84313725, 0.        , 0.19607843, 1.        ],
           [0.68627451, 0.31372549, 0.53333333, 1.        ],
           [0.68627451, 0.31372549, 0.53333333, 1.        ],
           [0.60392157, 0.18823529, 0.59215686, 1.        ]])
    elif type == 'wave':
        colors0 = np.array([
           [0.62109375, 0.72265625, 0.74609375, 1.        ],
           [0.1875    , 0.61328125, 0.72265625, 1.        ],
           [0.1875    , 0.3828125 , 0.55078125, 1.        ],
           [0.21875   , 0.40625   , 0.74609375, 1.        ],
           [0.22265625, 0.234375  , 0.5546875 , 1.        ],
           [0.73046875, 0.3515625 , 0.74609375, 1.        ],
           [0.6015625 , 0.1875    , 0.58984375, 1.        ],
           [0.51953125, 0.1875    , 0.1875    , 1.        ],
           [0.74609375, 0.19921875, 0.37109375, 1.        ],
           [0.74609375, 0.40234375, 0.33984375, 1.        ],
           [0.74609375, 0.74609375, 0.74609375, 1.        ],
           [0.6015625 , 0.49609375, 0.60546875, 1.        ]])

    colors0_resized = resize(colors0,(256,4),order=1)
    cmap_new = ListedColormap(colors0_resized)
    return cmap_new
