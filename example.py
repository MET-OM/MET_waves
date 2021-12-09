import met_waves 

met_waves.plot_panarctic_map(start_time='2007-11-08T12', end_time='2007-11-10T23',
                    product='NORA3', variable='hs', method='mean')


met_waves.plot_timeseries(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20,
                lat=56.53, product='NORA3', variable='hs', write_csv=True)


met_waves.plot_2D_spectra(start_time='2007-11-08', end_time='2007-11-09', lon=3.20,
                lat=56.53, product='SPEC_NORA3')

#met_waves.plot_topaz(start_time='1997-01-01', end_time='1997-12-31', variable='ice_speed', method = 'mean',save_data =True)

