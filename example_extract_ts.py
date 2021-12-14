import met_waves
import time

start = time.time()

met_waves.extract_ts_point(start_date ='2019-01-01', 
                              end_date= '2019-01-31',
                    variable=['hs','tp','hs_swell','tp_swell'],
                    lon = 5, lat = 60,
                    product='NORA3')
    
end = time.time()
print(end - start)
    