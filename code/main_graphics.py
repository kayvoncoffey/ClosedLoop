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
HOURS = 6
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
g = 100 #NEED TO CHANGE THESE IN controller.py TOO
i = 100 #NEED TO CHANGE THESE IN controller.py TOO
Gs = [200 *g]*(n_G_delays+1)
Is = [0 *i]*(n_I_delays+1)
Ids = [0 *i]*(n_Idose_delays)
Gds = [0 *g]*(n_Gdose_delays)
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)
estimator.x0 = x0
mpc.set_initial_guess()

# Main Loop
a=40
b=45
meal = [0]*a+[i*3 for i in range(b-a)]+[0]*int(floor(N_iterations-b))
import contextlib
for k in range(N_iterations):
	sys.stdout = open(os.devnull, "w")
	u0 = mpc.make_step(x0)
	if (k>=a) & (k<=b): u0[0] += (k-a)*3 *g
	y_next = simulator.make_step(u0)
	x0 = estimator.make_step(y_next)
	sys.stdout = sys.__stdout__

# Graphics
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

# %%capture
fig, ax = plt.subplots(4, sharex=True, figsize=(16,12))
plt.ion()
plt.suptitle('Correction of Hyperglycemia with Unobserved Meal \n '+str(HOURS)+' hour simulation')
mpc_graphics.add_line(var_type='_x', var_name='G_t', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='I_t', axis=ax[1])
mpc_graphics.add_line(var_type='_u', var_name='G_u', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='I_u', axis=ax[3])

ax[0].axhline(80 *g,color='black',linestyle='--',linewidth=0.75)
ax[0].axhline(120 *g,color='black',linestyle='--',linewidth=0.75)
ax[0].set(title='Blood Glucose',ylabel='BG [mg/dl]')
ax[1].set(title='Insulin on Board',ylabel='IOB [mU/ml]')
ax[2].set(title='Controller Glucose Infusion',ylabel='G Infusion [mg/dl min]')
ax[3].set(title='Controller Insulin Infusion',ylabel='I Infusion [mU/ml min]',xlabel='seconds')

fig.align_ylabels()

from matplotlib.animation import FuncAnimation, ImageMagickWriter

def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]

anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

gif_writer = ImageMagickWriter(fps=5)
anim.save('/Users/kcoffey/Documents/ClosedLoop/output/test_Vmax70.gif')#, writer=gif_writer)




