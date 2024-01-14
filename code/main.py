# Run the MPC 
import importlib.util
import do_mpc
from math import floor
from model import *
from controller import template_mpc
from simulator import template_simulator
from estimator import template_estimator
import matplotlib.pyplot as plt
import sys
import os

# Set Project Parameters and Initialize
timescale = 1
HOURS = 3
N_iterations = int(floor(HOURS*60/timescale))

n_G_delays = int(floor(5/timescale))
n_I_delays = int(floor(15/timescale))
n_Idose_delays = int(floor(20/timescale))
n_Gdose_delays = int(floor(15/timescale))

model = template_model(time_scale=timescale,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)
mpc = template_mpc(model,time_scale=timescale)
simulator = template_simulator(model,time_scale=timescale)
estimator = do_mpc.estimator.StateFeedback(model)

# Set Initial State
Gs = [200 *100]*(n_G_delays+1)
Is = [0.001 *50]*(n_I_delays+1)
Ids = [0 *50]*(n_Idose_delays)
Gds = [0 *100]*(n_Gdose_delays)
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)

# Setup Visualization
# graphics = do_mpc.graphics.Graphics(mpc.data)

# fig, ax = plt.subplots(5, sharex=True)
# graphics.add_line(var_type='_x', var_name='G_t', axis=ax[0])
# graphics.add_line(var_type='_x', var_name='I_t', axis=ax[1])
# graphics.add_line(var_type='_u', var_name='G_u', axis=ax[2])
# graphics.add_line(var_type='_x', var_name='G_load', axis=ax[2]) #+str(n_Gdose_delays-1)
# graphics.add_line(var_type='_u', var_name='I_u', axis=ax[3])
# graphics.add_line(var_type='_x', var_name='I_load', axis=ax[3]) #+str(n_Idose_delays-1)
# # Fully customizable:
# ax[0].set(ylabel='BG [mg/dl]',ylim=[0,200])
# ax[1].set(ylabel='I [mU/ml]')
# ax[2].set(ylabel='G [mg/dl per min]')
# ax[3].set(ylabel='CSII [mU/ml per min]')


# Main Loop
import contextlib
for k in range(N_iterations):
	# sys.stdout = open(os.devnull, "w")
	u0 = mpc.make_step(x0)
	if k==75: u0[0] = 100 *100
	y_next = simulator.make_step(u0)
	x0 = estimator.make_step(y_next)
	# sys.stdout = sys.__stdout__


x_obs = mpc.data['_x']
u_obs = mpc.data['_u']

fig,ax = plt.subplots(4,1,figsize=(10,8))
ax[0].plot(x_obs[:,0]/100,label='G')
ax[0].plot(x_obs[:,n_G_delays]/100,label='G tau')
[ax[0].axhline(a,color='black',linestyle='--') for a in [60,140]]
ax[1].plot(x_obs[:,n_G_delays+1]/50,label='I')
ax[1].plot(x_obs[:,n_G_delays+1+n_I_delays]/50,label='I tau')
ax[2].plot(u_obs[:,0]/100,label='G in')
ax[3].plot(u_obs[:,1]/50,label='I in')
ax[3].plot(x_obs[:,n_G_delays+1+n_I_delays+n_Idose_delays]/50,label='I act')

ax[0].set(title='Glucose Profile',ylabel='mg/dl',ylim=[0,250])
ax[1].set(title='Insulin Profile',ylabel='mU/ml')
ax[2].set(title='Glucose Infusion Profile',ylabel='mg/dl per min')
ax[3].set(title='Insulin Infusion Profile',ylabel='mU/ml per min')
[ax[i].legend(loc='best') for i in range(4)]

plt.tight_layout()
plt.show()

# graphics.plot_results(mpc.data)



