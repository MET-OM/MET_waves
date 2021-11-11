from met_waves import plot_panarctic_map, plot_timeseries, plot_2D_spectra

#plot_panarctic_map(start_time='2020-11-08T23', end_time='2020-11-12T00',
#                   product='NORA3', variable='hs', method='timestep')

#plot_timeseries(start_time='2020-12-31T00', end_time='2020-12-31T23', lon=3.20,
#                lat=56.53, product='NORA3', variable='hs', write_csv=True)


plot_2D_spectra(start_time='2020-12-31T00', end_time='2020-12-31T23', lon=3.20,
                lat=56.53, product='NORA3')
