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
timescale = 5
HOURS = 2
N_iterations = int(floor(HOURS*60/timescale))

g = 100 #NEED TO CHANGE THESE IN controller.py TOO
i = 200 #NEED TO CHANGE THESE IN controller.py TOO

n_G_delays = int(floor(5/timescale))
n_I_delays = int(floor(15/timescale))
n_Idose_delays = int(floor(20/timescale))
n_Gdose_delays = int(floor(15/timescale))

model = template_model(time_scale=timescale,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)
mpc = template_mpc(model,time_scale=timescale,g=g,i=i)
simulator = template_simulator(model,time_scale=timescale)
estimator = do_mpc.estimator.StateFeedback(model)


# Set Initial State
Gs = [200 *g]*(n_G_delays+1) #*100
Is = [0 *i]*(n_I_delays+1) #*50
Ids = [0 *i]*(n_Idose_delays) #*50
Gds = [0 *g]*(n_Gdose_delays) #*100
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)
estimator.x0 = x0
mpc.set_initial_guess()

# Main Loop
import contextlib
for k in range(N_iterations):
	# sys.stdout = open(os.devnull, "w")
	u0 = mpc.make_step(x0)
	# if (k>=70) & (k<=80): u0[0] = (k-70)*1
	y_next = simulator.make_step(u0)
	x0 = estimator.make_step(y_next)
	# sys.stdout = sys.__stdout__

x_obs = mpc.data['_x']
u_obs = mpc.data['_u']

fig,ax = plt.subplots(4,1,figsize=(10,8))
ax[0].plot(x_obs[:,0]/g,label='G')
ax[0].plot(x_obs[:,n_G_delays]/g,label='G tau')
[ax[0].axhline(a,color='black',linestyle='--') for a in [80,120]]
ax[1].plot(x_obs[:,n_G_delays+1]/i,label='I')
ax[1].plot(x_obs[:,n_G_delays+1+n_I_delays]/i,label='I tau')
ax[2].plot(u_obs[:,0]/g,label='G in')
ax[3].plot(u_obs[:,1]/i,label='I in')
ax[3].plot(x_obs[:,n_G_delays+1+n_I_delays+n_Idose_delays]/i,label='I act')

ax[0].set(title='Glucose Profile',ylabel='mg/dl',ylim=[0,250])
ax[1].set(title='Insulin Profile',ylabel='mU/ml',ylim=[0,np.max(x_obs[:,n_G_delays+1])/i])
ax[2].set(title='Glucose Infusion Profile',ylabel='mg/dl per min')
ax[3].set(title='Insulin Infusion Profile',ylabel='mU/ml per min')
[ax[i].legend(loc='best') for i in range(4)]

plt.tight_layout()
plt.show()



