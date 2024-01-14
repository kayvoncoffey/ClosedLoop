# Setup the state estimator
def template_estimator(model):
	# # Obtain an instance of the do-mpc MHE class
	# # and initiate it with the model.
	# # Optionally pass a list of parameters to be estimated.
	# mhe = do_mpc.estimator.MHE(model)

	# # Set parameters:
	# setup_mhe = {
	#     'n_horizon': 10,
	#     't_step': 0.1,
	#     'meas_from_data': True,
	# }
	# mhe.set_param(**setup_mhe)

	# # Set custom objective function
	# # based on:
	# y_meas = mhe._y_meas
	# y_calc = mhe._y_calc

	# # and (for the arrival cost):
	# x_0 = mhe._x
	# x_prev = mhe._x_prev

	# ...
	# mhe.set_objective(...)

	# # Set bounds for states, parameters, etc.
	# mhe.bounds[...] = ...

	# # [Optional] Set measurement function.
	# # Measurements are read from data object by default.

	# mhe.setup()
	estimator = do_mpc.estimator.StateFeedback(model)

	return estimator