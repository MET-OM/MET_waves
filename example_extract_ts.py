import met_waves
import time

start = time.time()

met_waves.extract_point_nora3(start_date ='2019-01-01', 
                    end_date= '2019-12-31',
                    variable='hs', lon = 5, lat = 60)
    
end = time.time()
print(end - start)
    