import do_mpc
import numpy as np 
from math import floor

# Setup the Controller
def template_mpc(model,time_scale,g,i):
	n_G_delays = int(floor(5/time_scale))
	n_I_delays = int(floor(15/time_scale))
	n_Idose_delays = int(floor(20/time_scale))
	n_Gdose_delays = int(floor(15/time_scale))
	time_step = 1*60*time_scale
	# Obtain an instance of the do-mpc MPC class
	# and initiate it with the model:
	mpc = do_mpc.controller.MPC(model)

	# Set parameters:

	setup_mpc = {
		'n_robust': 1,
		'n_horizon': int(floor(70/time_scale)),
		't_step': time_step,
		'state_discretization': 'discrete',
		'store_full_solution':True,
		# Use MA27 linear solver in ipopt for faster calculations:
		#'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
	}
	mpc.set_param(**setup_mpc)

	# Configure objective function:
	_x = model.x
	_u = model.u
	mterm = (_x['G_t'] - 100 *g)**2 # terminal cost *100
	lterm = (_x['G_t'] <= 60 *i)*10000 + (_u['G_u']) + (_u['I_u']) #*100
	# lterm = (1/(_x['G_t'] - 10 *100)) # stage cost
	mpc.set_rterm(G_u=2.0, I_u=2.0)

	mpc.set_objective(mterm=mterm, lterm=lterm)
	# mpc.set_rterm(G_u=0.1, Q_dot = 1e-3) # Scaling for quad. cost.

	# State and input bounds:
	# # lower bounds of the states
	mpc.bounds['lower', '_x', 'I_t'] = 0
	mpc.bounds['lower', '_x', 'I_tau'] = np.array([0]*n_I_delays).reshape(-1,1)
	mpc.bounds['lower', '_x', 'I_load'] = np.array([0]*n_Idose_delays).reshape(-1,1)
	mpc.bounds['lower', '_x', 'G_t'] = 0 #*100
	mpc.bounds['lower', '_x', 'G_tau'] = np.array([0]*n_G_delays).reshape(-1,1)
	mpc.bounds['lower', '_x', 'G_load'] = np.array([0]*n_Gdose_delays).reshape(-1,1)

	# # upper bounds of the states

	# # lower bounds of the inputs
	mpc.bounds['lower', '_u', 'G_u'] = 0
	mpc.bounds['lower', '_u', 'I_u'] = 0

	# # upper bounds of the inputs
	mpc.bounds['upper', '_u', 'G_u'] = 0 *g
	mpc.bounds['upper', '_u', 'I_u'] = 500 *i

	# mpc.scaling['_x', 'G_t'] = 1
	# mpc.scaling['_x', 'G_tau'] = 1
	# mpc.scaling['_u', 'G_u'] = 1
	# mpc.scaling['_x', 'I_t'] = 100
	# mpc.scaling['_x', 'I_tau'] = 100
	# mpc.scaling['_u', 'I_u'] = 100

	mpc.set_uncertainty_values(gamma = np.array([0.3,0.5,0.7]))

	mpc.setup()

	return mpc



