import MET_waves


MET_waves.plot_timeseries(start_time='1993-01-31T10:00', end_time='1993-01-31T23:00', lon=4.93,
                lat=62.17, product='NORA3', variable='hs', write_csv=True, ts_obs=None)
