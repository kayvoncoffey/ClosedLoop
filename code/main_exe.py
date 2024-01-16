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
BG_tau = 150 # mg/dl 5 minutes ago
IOB = 0.2 # mU/ml
IOB_tau = 0.3 # mU/ml 15 minute ago


Gs = [BG *100]+[BG_tau *100]*(n_G_delays)
Is = [IOB *100]+[IOB_tau *100]*(n_I_delays)
Ids = [0 *100]*(n_Idose_delays)
Gds = [0 *100]*(n_Gdose_delays)
x0 = np.array(Gs+Is+Ids+Gds).reshape(-1,1)

mpc.x0 = x0 #mpc.set_initial_state(x0, reset_history=True)
simulator.x0 = x0 #simulator.set_initial_state(x0, reset_history=True)
mpc.set_initial_guess()

sys.stdout = open(os.devnull, "w")
u0 = mpc.make_step(x0)
y_next = simulator.make_step(u0)
x0 = estimator.make_step(y_next)
sys.stdout = sys.__stdout__

print('==================================================================== \n')
print('Glucose 5 minutes ago: '+str(BG_tau)+' mg/dl')
print('Glucose current: '+str(BG)+' mg/dl')
print('Insulin on board 15 minutes ago: '+str(IOB_tau)+' mU/ml')
print('Insulin on board currently: '+str(IOB)+' mU/ml')
print('-------------------------------------------------------------')
print('Controller Glucose infusion rate: '+str(u0[0]/100)+' mg/dl per min')
print('Controller Insulin infusion rate: '+str(u0[1]/100)+' mU/ml per min \n')
print('==================================================================== \n')
