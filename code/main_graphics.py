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
HOURS = 10 #7am to 9pm
N_iterations = int(floor(HOURS*60/timescale))

g = 100 
i = 50 

n_G_delays = int(floor(5/timescale))
n_I_delays = int(floor(15/timescale))
n_Idose_delays = int(floor(20/timescale))
n_Gdose_delays = int(floor(15/timescale))

model = template_model(time_scale=timescale,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)
mpc = template_mpc(model,time_scale=timescale,g=g,i=i)
simulator = template_simulator(model,time_scale=timescale)
# estimator = do_mpc.estimator.StateFeedback(model)
estimator = template_estimator(model,time_scale=timescale,g=g,i=i,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)


# Set Initial State
Gs = [150 *g]*(n_G_delays+1)
Is = [1 *i]*(n_I_delays+1)
Ids = [0 *i]*(n_Idose_delays)
Gds = [0 *g]*(n_Gdose_delays)
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)
estimator.x0 = x0
estimator.p_est0 = 0.5
mpc.set_initial_guess()
estimator.set_initial_guess()

# Main Loop
# bfast_start = int(floor(2*60/timescale)) #
# bfast_end = int(floor(2.5*60/timescale))
lunch_start = int(floor(5*60/timescale))
lunch_end = int(floor(6*60/timescale))
# dinner_start = int(floor(7.5*60/timescale))
# dinner_end = int(floor(8*60/timescale))

import contextlib
for k in range(N_iterations):
	sys.stdout = open(os.devnull, "w")
	u0 = mpc.make_step(x0)
	# if (k>=bfast_start) & (k<=bfast_end): u0[0] += (1/(bfast_end-bfast_start))*50 *g # 300 mg/dl of glucose infused of 15 mins
	if (k>=lunch_start) & (k<=lunch_end): u0[0] += (1/(lunch_end-lunch_start))*100 *g # 
	# if (k>=dinner_start) & (k<=dinner_end): u0[0] += (1/(dinner_end-dinner_start))*150 *g
	v0 = 1*g*np.random.randn(model.n_v,1) # measurement noise
	y_next = simulator.make_step(u0, v0=v0)
	x0 = estimator.make_step(y_next)
	sys.stdout = sys.__stdout__

# Graphics
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
mhe_graphics = do_mpc.graphics.Graphics(estimator.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# %%capture
fig, ax = plt.subplots(5, sharex=True, figsize=(20,12))
plt.ion()
plt.suptitle('Testing MHE Implementation with Measurement Noise \n '+str(HOURS)+' hour simulation, low insulin sensitivity (0.1)')
mpc_graphics.add_line(var_type='_x', var_name='G_t', axis=ax[0])
sim_graphics.add_line(var_type='_x', var_name='G_t', axis=ax[0])
# mhe_graphics.add_line(var_type='_x', var_name='G_t', axis=ax[0])
# mhe_graphics.add_line(var_type='_x', var_name='G_t_meas', axis=ax[0])

mpc_graphics.add_line(var_type='_x', var_name='I_t', axis=ax[1])
sim_graphics.add_line(var_type='_x', var_name='I_t', axis=ax[1])
# mhe_graphics.add_line(var_type='_x', var_name='I_t', axis=ax[1])
# mhe_graphics.add_line(var_type='_x', var_name='I_t_meas', axis=ax[1])

mpc_graphics.add_line(var_type='_u', var_name='G_u', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='I_u', axis=ax[3])
mhe_graphics.add_line(var_type='_p', var_name='gamma', axis=ax[4])

for line_i in mhe_graphics.result_lines.full: line_i.set_linestyle('--')
for line_i in sim_graphics.result_lines.full: line_i.set_linestyle('--')

ax[0].axhline(80 *g,color='black',linestyle='--',linewidth=0.75)
ax[0].axhline(120 *g,color='black',linestyle='--',linewidth=0.75)
ax[3].axhline(0, color='white',linestyle='--',linewidth=0.1)
ax[4].axhline(0, color='white',linestyle='--',linewidth=0.1)
ax[4].axhline(1, color='white',linestyle='--',linewidth=0.1)

ax[4].axhline(0.1,color='black',linestyle='--',linewidth=0.75)
ax[0].set(title='Blood Glucose',ylabel='BG [mg/dl]')
ax[1].set(title='Insulin on Board',ylabel='IOB [mU/ml]')
ax[2].set(title='Controller Glucose Infusion',ylabel='G Infusion [mg/dl min]')
ax[3].set(title='Controller Insulin Infusion',ylabel='I Infusion [mU/ml min]')
ax[4].set(title='MHE Estimate of Insulin Sensitivity',ylabel='Gamma 0-1',xlabel='hours')

import matplotlib.ticker as ticker
ticks_g = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/g))
ticks_i = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/i))
ticks_x = ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/60/60))
ax[0].yaxis.set_major_formatter(ticks_g)
ax[1].yaxis.set_major_formatter(ticks_i)
ax[2].yaxis.set_major_formatter(ticks_g)
ax[3].yaxis.set_major_formatter(ticks_i)
ax[4].xaxis.set_major_formatter(ticks_x)

fig.align_ylabels()

from matplotlib.animation import FuncAnimation, ImageMagickWriter

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mhe_graphics.plot_results(t_ind=t_ind)
    sim_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.reset_axes()
    mhe_graphics.reset_axes()
    sim_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]

anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

gif_writer = ImageMagickWriter(fps=5)
anim.save('/Users/kcoffey/Documents/ClosedLoop/output/mhe_test_1meal_90est_1noise.gif')#, writer=gif_writer)




