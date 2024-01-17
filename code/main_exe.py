# Run one step of MPC controller and print output

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
timescale=1
n_G_delays = int(floor(5/timescale))
n_I_delays = int(floor(15/timescale))
n_Idose_delays = int(floor(20/timescale))
n_Gdose_delays = int(floor(15/timescale))

model = template_model(time_scale=timescale,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)
mpc = template_mpc(model,time_scale=timescale)
simulator = template_simulator(model,time_scale=timescale)
estimator = do_mpc.estimator.StateFeedback(model)

# Set Initial State
BG = 100 # mg/dl
BG_tau = [100, 98, 96, 94, 90] # mg/dl 5 minute history
IOB = 0.2 # mU/ml
IOB_tau = [0.2,0.21,0.24,0.25,0.27,0.29,0.31,0.32,0.33,0.34,0.37,0.4,0.44,0.5,0.55] # mU/ml 15 minute history

g = 100 #NEED TO CHANGE THESE IN controller.py TOO
i = 100 #NEED TO CHANGE THESE IN controller.py TOO
Gs = [BG *g]+[j*100 for j in BG_tau] #[BG *100]+[BG_tau *100]*(n_G_delays)
Is = [IOB *i]+[j*100 for j in IOB_tau] #[IOB *100]+[IOB_tau *100]*(n_I_delays)
Ids = [0 *100]*(n_Idose_delays)
Gds = [0 *100]*(n_Gdose_delays)
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)
mpc.set_initial_guess()

# sys.stdout = open(os.devnull, "w")
u0 = mpc.make_step(x0)
y_next = simulator.make_step(u0)
x0 = estimator.make_step(y_next)
# sys.stdout = sys.__stdout__

preds_G = mpc.data.prediction(('_x','G_t'), t_ind=0)
preds_I = mpc.data.prediction(('_x','I_t'), t_ind=0)
comp_Gu = mpc.data.prediction(('_u','G_u'), t_ind=0)
comp_Iu = mpc.data.prediction(('_u','I_u'), t_ind=0)

fig, ax = plt.subplots(4,sharex=True,figsize=(10,8))
plt.suptitle('Correction of Hyperglycemia with Unobserved Meal')
hist = np.arange(-np.maximum(n_G_delays,n_I_delays),1)
ax[0].plot(hist,[np.nan]*(len(hist)-n_G_delays-1)+BG_tau[::-1]+[BG])
ax[1].plot(hist,IOB_tau[::-1]+[IOB])
for j in [0,1,2]:
	ax[0].plot(np.arange(0,len(preds_G[0,:,j])),preds_G[0,:,j]/g,linestyle='--',color='orange')
	ax[1].plot(np.arange(0,len(preds_I[0,:,j])),preds_I[0,:,j]/i,linestyle='--',color='orange')
	ax[2].plot(np.arange(0,len(comp_Gu[0,:,j])),comp_Gu[0,:,j]/g,linestyle='--',color='orange')
	ax[3].plot(np.arange(0,len(comp_Iu[0,:,j])),comp_Iu[0,:,j]/i,linestyle='--',color='orange')


ax[0].axhline(80 ,color='black',linestyle='--',linewidth=0.75)
ax[0].axhline(120 ,color='black',linestyle='--',linewidth=0.75)
ax[0].set(title='Blood Glucose',ylabel='BG [mg/dl]')
ax[1].set(title='Insulin on Board',ylabel='IOB [mU/ml]')
ax[2].set(title='Controller Glucose Infusion',ylabel='G Infusion [mg/dl min]')
ax[3].set(title='Controller Insulin Infusion',ylabel='I Infusion [mU/ml min]',xlabel='seconds')

plt.tight_layout()
plt.show()

print('==================================================================== \n')
print('Glucose 5 minutes ago: '+str(BG_tau)+' mg/dl')
print('Glucose current: '+str(BG)+' mg/dl')
print('Insulin on board 15 minutes ago: '+str(IOB_tau)+' mU/ml')
print('Insulin on board currently: '+str(IOB)+' mU/ml')
print('-------------------------------------------------------------')
print('Controller Glucose infusion rate: '+str(u0[0]/100)+' mg/dl per min')
print('Controller Insulin infusion rate: '+str(u0[1]/100)+' mU/ml per min \n')
print('==================================================================== \n')
