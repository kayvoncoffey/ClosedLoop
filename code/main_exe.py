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

g = 100 #NEED TO CHANGE THESE IN controller.py TOO
i = 200 #NEED TO CHANGE THESE IN controller.py TOO

model = template_model(time_scale=timescale,nG=n_G_delays,nI=n_I_delays,nGd=n_Gdose_delays,nId=n_Idose_delays)
mpc = template_mpc(model,time_scale=timescale,g=g,i=i)
simulator = template_simulator(model,time_scale=timescale)
estimator = do_mpc.estimator.StateFeedback(model)

# Set Initial State
BG = 200 # mg/dl
BG_tau = [199, 197, 194, 190, 185] # mg/dl 5 minute history
IOB = 1 # mU/ml
IOB_tau = [1.2,1.21,1.24,1.25,1.271,1.29,1.31,1.32,1.33,1.34,1.37,1.4,1.44,1.5,1.55] # mU/ml 15 minute history

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
	ax[0].plot(np.arange(0,len(preds_G[0,:,j])),preds_G[0,:,j]/g,linestyle='--',color='orange',linewidth=0.75)
	ax[1].plot(np.arange(0,len(preds_I[0,:,j])),preds_I[0,:,j]/i,linestyle='--',color='orange',linewidth=0.75)
	ax[2].plot(np.arange(0,len(comp_Gu[0,:,j])),comp_Gu[0,:,j]/g,linestyle='--',color='orange',linewidth=0.75)
	ax[3].plot(np.arange(0,len(comp_Iu[0,:,j])),comp_Iu[0,:,j]/i,linestyle='--',color='orange',linewidth=0.75)


ax[0].axhline(80 ,color='black',linestyle='--',linewidth=0.75)
ax[0].axhline(120 ,color='black',linestyle='--',linewidth=0.75)
ax[0].set(title='Blood Glucose',ylabel='BG [mg/dl]')
ax[1].set(title='Insulin on Board',ylabel='IOB [mU/ml]')
ax[2].set(title='Controller Glucose Infusion',ylabel='G Infusion [mg/dl '+str(timescale)+'min]')
ax[3].set(title='Controller Insulin Infusion',ylabel='I Infusion [mU/ml '+str(timescale)+'min]',xlabel='minutes')

plt.tight_layout()
plt.show()

print('==================================================================== \n')
print('Glucose 5 minutes ago: '+str(BG_tau)+' mg/dl')
print('Glucose current: '+str(BG)+' mg/dl')
print('Insulin on board 15 minutes ago: '+str(IOB_tau)+' mU/ml')
print('Insulin on board currently: '+str(IOB)+' mU/ml')
print('-------------------------------------------------------------')
print('Controller Glucose infusion rate: '+str(u0[0]/100)+' mg/dl per '+str(timescale)+'min')
print('Controller Insulin infusion rate: '+str(u0[1]/100)+' mU/ml per '+str(timescale)+'min \n')
print('==================================================================== \n')
