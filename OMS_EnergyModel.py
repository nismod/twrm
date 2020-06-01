# This model is developed for Oxford Martin School Transboundary Resource Management
# Jordan River Basin Project Energy Model
# First created by Gokhan Cuceloglu, 3/5/2020, Istanbul

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

#keeps the simulation time
start_time = time.time()

#reads inputs such as hourly consumption, and renewable (solar and wind) generation for 1MW installed capacity
df2              = pd.read_csv("data/EnergyModel_Inputs2.csv", sep = ",")

hours            = np.array(list(range(1,len(df2)+1)))
demand           = np.array(df2.loc[:, 'Hourly Consumption'])
generated_solar  = np.array(df2.loc[:, 'Generated Solar'])
generated_wind   = np.array(df2.loc[:, 'Generated Wind'])
water_rel_demand = np.zeros(len(df2)) # if there will be time series for that it can be replaced. Currently none
num_hours        = len(hours)

#installed capacities
base_load_capacity  = 4800.0
naturalgas_capacity = 12000.0
diesel_capacity     = 0.0
solar_capacity      = 9500.0
wind_capacity       = 21.0

#coeffs
fac_reserve         = 0.2
base_load_fac       = 0.5
energy_loss_grid    = 0.0
energy_loss_storage = 0.0
max_ramp_rate       = 1000.0 #MWh
grid_connection     = 1000.0 #MW
storage_capacity    = 5000.0 #MWh

#production capacities depending on the installed capacity
base_load       = np.ones(len(df2)) * base_load_capacity * base_load_fac
naturalgas      = np.ones(len(df2)) * naturalgas_capacity
diesel          = np.ones(len(df2)) * diesel_capacity
generated_solar = generated_solar   * solar_capacity
generated_wind  = generated_wind    * wind_capacity

#derived series
generated_renewable   = np.zeros(num_hours)
used_baseload         = np.zeros(num_hours)
used_wind             = np.zeros(num_hours)
used_solar            = np.zeros(num_hours)
used_renewable        = np.zeros(num_hours)
used_naturalgas       = np.zeros(num_hours)
used_diesel           = np.zeros(num_hours)
net_demand            = np.zeros(num_hours)
excess_energy         = np.zeros(num_hours)
energy_deficit        = np.zeros(num_hours)
energy_storage        = np.zeros(num_hours)
used_storage          = np.zeros(num_hours)

#Core Model 
demand                = demand + water_rel_demand - base_load
generated_renewable   = generated_wind + generated_solar
used_wind             = np.minimum(generated_wind, demand)
used_solar            = np.minimum(generated_solar, (demand - used_wind))
used_renewable        = used_wind + used_solar
net_demand            = demand - used_renewable

for idx in range(1,num_hours):
    if idx == 0:
        used_storage[idx]    = 0.0
        used_naturalgas[idx] = np.minimum(naturalgas[idx] * (1.0-energy_loss_grid), net_demand[idx])
        used_diesel[idx]     = np.minimum(diesel[idx], (net_demand[idx]-used_naturalgas[idx]))
        excess_energy[idx]   = generated_renewable[idx] - used_renewable[idx]
        energy_deficit[idx]  = demand[idx] - (used_renewable[idx] + used_naturalgas[idx] + used_diesel[idx])
        energy_storage[idx]  = excess_energy[idx]
    else:
        used_storage[idx]    = np.minimum(energy_storage[idx-1] * (1.0-energy_loss_grid) , net_demand[idx])
        used_naturalgas[idx] = np.minimum(naturalgas[idx], (net_demand[idx] - used_storage[idx]))
        used_diesel[idx]     = np.minimum(diesel[idx], (net_demand[idx] - used_naturalgas[idx] - used_storage[idx]))
        excess_energy[idx]   = generated_renewable[idx] - used_renewable[idx]
        energy_deficit[idx]  = demand[idx] - (used_renewable[idx] + used_naturalgas[idx] + used_diesel[idx] + used_storage[idx])
        energy_storage[idx]  = energy_storage[idx-1] + excess_energy[idx] - (used_storage[idx]/(1.0-energy_loss_storage))
#end of core model
        
#other calculations
total_demand            = sum(demand)
total_used_wind         = sum(used_wind)
total_used_solar        = sum(used_solar)
total_used_natural_gas  = sum(used_naturalgas)
total_used_diesel       = sum(used_diesel)
total_conventional      = total_used_natural_gas + total_used_diesel
total_renewable         = total_used_wind + total_used_solar
total_energy_deficit    = sum(energy_deficit)
required_capacity       = max(net_demand) * (1+fac_reserve)


df = pd.DataFrame({
    "Hours"                 : hours,
    "Total Demand"          : demand,
    "Generated Wind "       : generated_wind,
    "Generated Solar"       : generated_solar,
    "Generated Renewable"   : generated_renewable,
    "Used_Wind"			    : used_wind,
    "Used Solar"			: used_solar,
    "Used Renewable"		: used_renewable,
    "Net Demand"			: net_demand,
    "Used Storage"          : used_storage,
    "Natural Gas Used"	    : used_naturalgas,
    "Diesel Used"		    : used_diesel,
    "Excess Energy"		    : excess_energy,
    "Energy Deficit"		: energy_deficit,
    "Storage"		    	: energy_storage,
    })

#Processing Time
finish_time = time.time()
print("--- %s seconds ---" % (finish_time - start_time))

#Output File
df.to_csv('results/EnergyModel_Output.csv', index=False)

#Plot
plot = 1
if  plot==1:
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=False)
    pal = sns.color_palette("Set1")
    #axes.stackplot(hours, used_wind, used_solar, used_naturalgas, used_diesel, labels=['wind','solar','natural','diesel'])
    axes[0,0].stackplot(hours, base_load, used_wind, used_solar, used_naturalgas, used_diesel, energy_deficit, excess_energy, labels=['base load','wind','solar','natural','diesel', 'energy deficit', 'excess energy'], colors=pal, alpha=0.4)
    axes[0,0].set(xlabel='hours', ylabel='GW', title='Energy Demand and Supply')

    axes[0,1].plot(net_demand)
    axes[1,0].plot(energy_storage)
    axes[1,0].plot(used_storage)
    axes[1,1].plot(excess_energy)

    #plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()