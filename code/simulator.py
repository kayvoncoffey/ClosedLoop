import do_mpc
import numpy as np

# Setup the system simulator
def template_simulator(model,time_scale):
	# Obtain an instance of the do-mpc simulator class
	# and initiate it with the model:
	time_step = 1*60*time_scale
	simulator = do_mpc.simulator.Simulator(model)

	# Set parameter(s):
	simulator.set_param(t_step = time_step)

	# Uncertain parameters
	p_num = simulator.get_p_template()
	def p_fun(t_now):
		p_num['gamma'] = 0.6 #np.random.uniform(0.2,0.7)
		return p_num

	simulator.set_p_fun(p_fun)

	# Setup simulator:
	simulator.setup()

	return simulator