from nora3_plt import plot_panarctic_map_timestep, plot_panarctic_map_mean 

# plot timesteps
plot_panarctic_map_timestep(start_time =  '2005-01-07T18',end_time = '2005-01-09T18',variable ='hs')
#terminal command for gif: convert -delay 20 -loop 0 *png hs.gif


# plot mean
#plot_panarctic_map_mean(start_time =  '2005-01-07T18',end_time = '2005-01-09T18',variable ='hs')
