import do_mpc

# Setup the system simulator
def template_simulator(model,time_scale):
	# Obtain an instance of the do-mpc simulator class
	# and initiate it with the model:
	time_step = 1*60*time_scale
	simulator = do_mpc.simulator.Simulator(model)

	# Set parameter(s):
	simulator.set_param(t_step = time_step)

	# Optional: Set function for parameters and time-varying parameters.

	# Setup simulator:
	simulator.setup()

	return simulator