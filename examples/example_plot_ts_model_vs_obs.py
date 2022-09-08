import MET_waves
import xarray as xr

start_time='2016-12-25T12'; end_time='2016-12-29T00'
#url_obs = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/Aggregated_BUOY_observations/D_Breisundet_wave.ncml.html'
url_obs = 'https://thredds.met.no/thredds/dodsC/obs/buoy-svv-e39/2016/12/201612_E39_D_Breisundet_wave.nc'
df_obs = xr.open_dataset(url_obs).sel(time=slice(start_time, end_time)).to_dataframe()

MET_waves.plot_timeseries(start_time=start_time, end_time=end_time, 
                                    lon=5.93,lat=62.45, product='NORA3', 
                                    variable='hs', ts_obs = df_obs['Hm0'],   
                                    write_csv=False)




