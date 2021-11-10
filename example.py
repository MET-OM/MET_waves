from met_waves import plot_panarctic_map, plot_timeseries

#plot_panarctic_map(start_time='2020-11-08T23', end_time='2020-11-12T00',
#                   product='NORA3', variable='hs', method='timestep')

plot_timeseries(start_time='2019-01-01T00', end_time='2019-12-31T23', lon=3.20,
                lat=56.53, product='NORA3', variable='hs', write_csv=True)
