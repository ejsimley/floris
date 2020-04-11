# Copyright 2020 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See read the https://floris.readthedocs.io for documentation

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyoptsparse
import os

import floris.tools as wfct
import floris.tools.wind_rose as rose
import floris.tools.power_rose as pr

# Define wind farm coordinates and layout
wf_coordinate = [39.8283, -98.5795]

# set min and max yaw offsets for optimization
min_yaw = 0.0
max_yaw = 25.0

# Define minimum and maximum wind speed for optimizing power. 
# Below minimum wind speed, assumes power is zero.
# Above maximum_ws, assume optimal yaw offsets are 0 degrees
minimum_ws = 3.0
maximum_ws = 15.0

# Instantiate the FLORIS object
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = wfct.floris_interface.FlorisInterface(
    os.path.join(file_dir, '../../../example_input.json')
)

# Set wind farm to N_row x N_row grid with constant spacing 
# (2 x 2 grid, 5 D spacing)
D = fi.floris.farm.turbines[0].rotor_diameter
N_row = 2
spc = 5
layout_x = []
layout_y = []
for i in range(N_row):
	for k in range(N_row):
		layout_x.append(i*spc*D)
		layout_y.append(k*spc*D)
N_turb = len(layout_x)

fi.reinitialize_flow_field(
	layout_array=(layout_x, layout_y),
	wind_direction=[270.0],
	wind_speed=[8.0]
)
fi.calculate_wake()

# ==============================================================================
print('Plotting the FLORIS flowfield...')
# ==============================================================================

# Initialize the horizontal cut
hor_plane = fi.get_hor_plane(
    height=fi.floris.farm.turbines[0].hub_height
)

# Plot and show
fig, ax = plt.subplots()
wfct.visualization.visualize_cut_plane(hor_plane, ax=ax)
ax.set_title('Baseline flow for U = 8 m/s, Wind Direction = 270$^\circ$')

# ==============================================================================
print('Importing wind rose data...')
# ==============================================================================

# Create wind rose object and import wind rose dataframe using WIND Toolkit
# HSDS API. Alternatively, load existing file with wind rose information.
calculate_new_wind_rose = False

wind_rose = rose.WindRose()

if calculate_new_wind_rose:

	wd_list = np.arange(0,360,5)
	ws_list = np.arange(0,26,1)

	df = wind_rose.import_from_wind_toolkit_hsds(
		wf_coordinate[0],
	    wf_coordinate[1],
	    ht = 100,
	    wd = wd_list,
	    ws = ws_list,
	    limit_month = None,
	    st_date = None,
	    en_date = None
	)

else:
	df = wind_rose.load(os.path.join(
		file_dir, '../../scipy/windtoolkit_geo_center_us.p'
	))

# plot wind rose
wind_rose.plot_wind_rose()

def objective_function(varDict, **kwargs):
    # Parse the variable dictionary
    yaw = varDict['yaw']

    # Compute the objective function
    funcs = {}
    funcs['obj'] = -1*fi.get_farm_power_for_yaw_angle(yaw)/1e3

    fail = False
    return funcs, fail

# =============================================================================
print('Finding baseline and optimal yaw angles in FLORIS...')
# =============================================================================

solutions = []
base_result_dict = dict()
opt_result_dict = dict()

for i in range(len(df.wd)):
    print('Computing wind speed, wind direction pair ' + str(i) \
		+ ' out of ' + str(len(df.wd)) + ': wind speed = ' \
		+ str(df.ws[i]) + ' m/s, wind direction = ' \
		+ str(df.wd[i])+' deg.')

    wd_itr = df.wd[i]
    ws_itr = df.ws[i]

    if (ws_itr >= minimum_ws) & (ws_itr <= maximum_ws):
        fi.reinitialize_flow_field(
            wind_direction=[wd_itr],wind_speed=[ws_itr])

        # calculate baseline power
        fi.calculate_wake(yaw_angles=0.0)
        power_base = fi.get_turbine_power()

        # calculate power for no wake case
        fi.calculate_wake(no_wake=True)
        power_no_wake = fi.get_turbine_power(no_wake=True)

        # Setup the optimization problem
        optProb = pyoptsparse.Optimization('yaw_opt_wd_' + str(wd_itr),
                                        objective_function)

        # Add the design variables to the optimization problem
        optProb.addVarGroup('yaw', N_turb, 'c', lower=min_yaw,
                                                upper=max_yaw,
                                                value=0.)

        # Add the objective to the optimization problem
        optProb.addObj('obj')

        # Setup the optimization solver
        # Note: pyOptSparse has other solvers available; some may require additional
        #   licenses/installation. See https://github.com/mdolab/pyoptsparse for
        #   more information. When ready, they can be invoked by changing 'SLSQP'
        #   to the solver name, for example: 'opt = pyoptsparse.SNOPT(fi=fi)'.
        opt = pyoptsparse.SLSQP()

        # Run the optimization with finite-differencing
        solution = opt(optProb, sens='FD')

        opt_yaw_angles = solution.getDVs()['yaw']

        # optimized power
        fi.calculate_wake(yaw_angles=opt_yaw_angles)
        power_opt = fi.get_turbine_power()
    elif ws_itr >= maximum_ws:
        fi.reinitialize_flow_field(
            wind_direction=[wd_itr], wind_speed=[ws_itr])

        # calculate baseline power
        fi.calculate_wake(yaw_angles=0.0)
        power_base = fi.get_turbine_power()

        # calculate power for no wake case
        fi.calculate_wake(no_wake=True)
        power_no_wake = fi.get_turbine_power(no_wake=True)

        opt_yaw_angles = N_turb*[0.0]
        power_opt = power_base
    else:
        power_base = N_turb*[0.0]
        power_no_wake = N_turb*[0.0]
        opt_yaw_angles = N_turb*[0.0]
        power_opt = N_turb*[0.0]

    base_result_dict[i] = {
        'ws':ws_itr,
        'wd':wd_itr,
        'power_baseline':np.sum(power_base),
        'turbine_power_baseline':power_base,
        'power_no_wake':np.sum(power_no_wake),
        'turbine_power_no_wake':power_no_wake
    }

    opt_result_dict[i] = {
        'ws':ws_itr,
        'wd':wd_itr,
        'power_opt':np.sum(power_opt),
        'turbine_power_opt':power_opt,
        'yaw_angles':opt_yaw_angles
    }

df_base = pd.DataFrame.from_dict(base_result_dict, "index")
df_base.reset_index(drop=True,inplace=True)

df_opt = pd.DataFrame.from_dict(opt_result_dict, "index")
df_opt.reset_index(drop=True,inplace=True)

# Initialize power rose
case_name = 'Example '+str(N_row)+' x '+str(N_row)+ ' Wind Farm'
power_rose = pr.PowerRose()
power_rose.make_power_rose_from_user_data(
	case_name,
	df,
	df_base['power_no_wake'],
	df_base['power_baseline'],
	df_opt['power_opt']
)

# Summarize using the power rose module
fig, axarr = plt.subplots(3, 1, sharex=True, figsize=(6.4, 6.5))
power_rose.plot_by_direction(axarr)
power_rose.report()

plt.show()