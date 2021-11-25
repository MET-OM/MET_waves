from met_waves import plot_panarctic_map, plot_timeseries, plot_2D_spectra, plot_topaz

# plot_panarctic_map(start_time='1997-01-08T22', end_time='1997-01-08T23',
#                     product='NORA3', variable='hs', method='mean')

#plot_timeseries(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20,
#                lat=56.53, product='NORA3', variable='hs', write_csv=True)


#plot_2D_spectra(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20,
#                lat=56.53, product='NORA3')

plot_topaz(start_time='1997-01-01T00', end_time='1997-12-31T23', variable='fice', method = 'mean')
