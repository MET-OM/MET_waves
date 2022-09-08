import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
#import os 

url = 'Tromso250_19930131.nc'
point = None # [18.1277, 69.8303] # None
start_time='1993-02-01T19'; end_time='1993-02-01T20'
ds = xr.open_dataset(url).sel(time=slice(start_time, end_time))
print(url)

print('Hs',ds.hs.max())


#print('wsp',ds.ff.max())

for i in range(ds.time.shape[0]):
    fig, ax = plt.subplots()
    #levels = np.round(np.linspace(0,int(np.ceil(ds.hs.max())),int(np.ceil(ds.hs.max()))*2),1)
    levels = np.round(np.linspace(0,17,170),1)
    #cs = ax.contour(ds.longitude, ds.latitude,ds.depth.loc[ds.time[i]],levels = np.round(np.linspace(0,460,8),1),cmap='binary') #, transform = ccrs.PlateCarree(),cmap='coolwarm') # coolwarm 
    #ax.clabel(cs, inline=1, fontsize=8, fmt='%1.0f')
    im = ax.contourf(ds.longitude, ds.latitude,ds.hs.loc[ds.time[i]],levels = levels,cmap='jet') #, transform = ccrs.PlateCarree(),cmap='coolwarm') # coolwarm 
    if point is not None:
        plt.plot(point[0],point[1],marker ='^',color='magenta', markersize=8)
    plt.title('DNORA/SWAN,'+str(ds.time.values[i])[:13] +'UTC')
    #cbar_ax = plt.colorbar(im) 
    #cbar_ax.ax.set_title('$'+'H_s[m]'+'$')
    ax.set_facecolor('beige')
    plt.savefig('png/'+str(ds.time.values[i])[:13]+'_Hs.png',bbox_inches = 'tight',dpi=300)
    plt.close()

#terminal command:
#convert -delay 20 -loop 0 *png Hs.gif

#reduce size
#gifsicle -i Hs_winter_2019_2020.gif  -O3 --colors 256 -o Hs_winter_2019_2020_opt.gif
