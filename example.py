from met_waves import plot_panarctic_map, plot_timeseries, plot_2D_spectra, plot_topaz

#plot_panarctic_map(start_time='1997-01-08T22', end_time='1997-01-08T23',
#                   product='TOPAZ', variable='ssh', method='timestep')

#plot_timeseries(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20,
#                lat=56.53, product='NORA3', variable='hs', write_csv=True, figsize=(30, 9))


#plot_2D_spectra(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20,
#                lat=56.53, product='NORA3')

plot_topaz(date='1999-02-02', variable='fice')
