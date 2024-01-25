import do_mpc
import numpy as np
from math import floor

# Setup the state estimator
def template_estimator(model,time_scale,g,i,nG,nI,nGd,nId):
	time_step = 1*60*time_scale

	mhe = do_mpc.estimator.MHE(model, ['gamma'])

	# Set parameters:
	setup_mhe = {
	    'n_horizon': int(floor(90/time_scale)),#int(floor(20/time_scale)),
	    't_step': time_step,
	    'store_full_solution': True,
	    'meas_from_data': True,
	}
	mhe.set_param(**setup_mhe)


	# Set objective weighting matrices
	n_states = len(model.x.labels())

	P_v = np.diag(np.array([2]))
	P_x = 5*np.eye(n_states)
	P_p = 1*np.eye(1)

	mhe.set_default_objective(P_x, P_v, P_p)

	# Define function to return struct with non-estimated parameters
	p_template_mhe = mhe.get_p_template()

	def p_fun_mhe(t_now):
	    return p_template_mhe

	mhe.set_p_fun(p_fun_mhe)

	mhe.bounds['lower', '_x', 'I_t'] = 0
	mhe.bounds['lower', '_x', 'I_tau'] = np.array([0]*nI).reshape(-1,1)
	mhe.bounds['lower', '_x', 'I_load'] = np.array([0]*nId).reshape(-1,1)
	mhe.bounds['lower', '_x', 'G_t'] = 0 #*100
	mhe.bounds['lower', '_x', 'G_tau'] = np.array([0]*nG).reshape(-1,1)
	mhe.bounds['lower', '_x', 'G_load'] = np.array([0]*nGd).reshape(-1,1)

	# # upper bounds of the states

	# # lower bounds of the inputs
	mhe.bounds['lower', '_u', 'G_u'] = 0
	mhe.bounds['lower', '_u', 'I_u'] = 0

	# # upper bounds of the inputs
	mhe.bounds['upper', '_u', 'G_u'] = 0 *g
	mhe.bounds['upper', '_u', 'I_u'] = 5 *i

	mhe.bounds['lower','_p_est', 'gamma'] = 0.01
	mhe.bounds['upper','_p_est', 'gamma'] = 1

	# # [Optional] Set measurement function.
	# # Measurements are read from data object by default.

	mhe.setup()

	# mhe = do_mpc.estimator.StateFeedback(model)

	return mhe