# This model is developed for Oxford Martin School Transboundary Resource Management
# Jordan River Basin Project Energy Model
# First created by Gokhan Cuceloglu, 3/5/2020, Istanbul

#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

start_time = time.time()


df2             = pd.read_csv("data/EnergyModel_Inputs.csv", sep = ",")

hours           = np.array(list(range(1,len(df2)+1)))
demand          = np.array(df2.loc[:, 'Hourly Consumption'])
generated_solar = np.array(df2.loc[:, 'Generated Solar'])
generated_wind  = np.array(df2.loc[:, 'Generated Wind'])
other_demand    = np.ones(len(df2))
base_load       = np.ones(len(df2))

num_hours           = len(hours)
naturalgas_capacity = np.ones(num_hours) * 2740.0
diesel_capacity     = np.ones(num_hours) * 814.0

#coeffs
fac_solar        = 1.0
fac_wind         = 1.0
fac_reserve      = 0.2
fac_natural      = 1.0
fac_diesel       = 1.0
fac_baseload     = 0.0
fac_otherdemands = 0.0
energy_loss      = 0.1

#derived series
generated_renewable   = np.zeros(num_hours)
used_wind             = np.zeros(num_hours)
used_solar            = np.zeros(num_hours)
used_renewable        = np.zeros(num_hours)
used_wind             = np.zeros(num_hours)
used_naturalgas       = np.zeros(num_hours)
used_diesel           = np.zeros(num_hours)
net_demand            = np.zeros(num_hours)
excess_energy         = np.zeros(num_hours)
energy_deficit        = np.zeros(num_hours)
energy_storage        = np.zeros(num_hours)
used_storage          = np.zeros(num_hours)

#factors
generated_solar       = generated_solar     * fac_solar
generated_wind        = generated_wind      * fac_wind
naturalgas_capacity   = naturalgas_capacity * fac_natural
diesel_capacity       = diesel_capacity     * fac_diesel
other_demand          = other_demand        * fac_otherdemands
base_load             = base_load           * fac_baseload

#Core Model 
demand                = demand + other_demand - base_load
generated_renewable   = generated_wind + generated_solar
used_wind             = np.minimum(generated_wind, demand)
used_solar            = np.minimum(generated_solar, (demand - used_wind))
used_renewable        = used_wind + used_solar
net_demand            = demand - used_renewable

for idx in range(1,num_hours):
    if idx == 0:
        used_storage[idx]    = 0.0
        used_naturalgas[idx] = np.minimum(naturalgas_capacity[idx] * (1.0-energy_loss), net_demand[idx])
        used_diesel[idx]     = np.minimum(diesel_capacity[idx], (net_demand[idx]-used_naturalgas[idx]))
        excess_energy[idx]   = generated_renewable[idx] - used_renewable[idx]
        energy_deficit[idx]  = demand[idx] - (used_renewable[idx] + used_naturalgas[idx] + used_diesel[idx])
        energy_storage[idx]  = excess_energy[idx]
    else:
        used_storage[idx]    = np.minimum(energy_storage[idx-1] * (1.0-energy_loss) , net_demand[idx])
        used_naturalgas[idx] = np.minimum(naturalgas_capacity[idx], (net_demand[idx] - used_storage[idx]))
        used_diesel[idx]     = np.minimum(diesel_capacity[idx], (net_demand[idx] - used_naturalgas[idx] - used_storage[idx]))
        excess_energy[idx]   = generated_renewable[idx] - used_renewable[idx]
        energy_deficit[idx]  = demand[idx] - (used_renewable[idx] + used_naturalgas[idx] + used_diesel[idx] + used_storage[idx])
        energy_storage[idx]  = energy_storage[idx-1] + excess_energy[idx] - (used_storage[idx]/(1.0-energy_loss))
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
plot = 0
if  plot==1:
    fig, axes = plt.subplots(nrows=1, ncols=1, sharex=False)
    pal = sns.color_palette("Set1")
    #axes.stackplot(hours, used_wind, used_solar, used_naturalgas, used_diesel, labels=['wind','solar','natural','diesel'])
    axes.stackplot(hours, base_load, used_wind, used_solar, used_naturalgas, used_diesel, excess_energy, labels=['base load','wind','solar','natural','diesel','excess energy'], colors=pal, alpha=0.4)
    axes.set(xlabel='hours', ylabel='GW', title='Energy Demand and Supply')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()