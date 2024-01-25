import do_mpc
from math import floor
import numpy as np
from casadi import *


# Define dynamic model
def f1(I,time_scale):
	Rg = 180 *time_scale # scaled by 5 to convert to 5 minute time increments
	alpha = 0.29
	Vp = 3
	C5 = 26
	return Rg/(1+np.exp(alpha*(I/Vp - C5)))

def f2(G,time_scale):
	Ub = 72 *time_scale # scaled by 5 to convert to 5 minute time increments
	C2 = 144
	Vg = 10
	return Ub*(1-np.exp(-G/(C2*Vg)))
	
def f3(G,time_scale):
	C3 = 1000
	Vg = 10
	return G/(C3*Vg)

def f4(I,time_scale):
	U0 = 40 *time_scale #scaled by 5 to convert to 5 minute time increments
	Um = 940 *time_scale #scaled by 5 to convert to 5 minute time increments
	beta = 1.77
	C4 = 80
	Vi = 11
	E = 0.2 *time_scale # scaled by 5 to convert to 5 minute time increments
	ti = 100 /time_scale # scaled by 1/5 to convert to 5 minute time increments
	return U0 + (Um-U0)/(1+np.exp(-beta*np.log(0.0000001+I/C4*(1/Vi+1/(E*ti)))));

def f5(G,time_scale):
	Rm = 210 *time_scale # scaled by 5 to convert to 5 minute time increments
	C1 = 2000
	Vg = 10
	a1 = 300
	return Rm/(1+np.exp((C1-G/Vg)/a1))

def template_model(time_scale,nG,nI,nGd,nId):
	# Obtain an instance of the do-mpc model class
	model_type = 'discrete'
	model = do_mpc.model.Model(model_type)
	# and select time discretization:
	# Parameters struct (known plant & system parameters): Kissler system parameters
	# time_scale = tscale
	time_step_use = 1*60*time_scale

	# Uncertain parameters:
	GAMMA = model.set_variable('_p', 'gamma')
	# GAMMA = 0.5
	BETA = 0
	Km = 2300
	m = 60 /time_scale # scaled by 1/5 to convert to 5 minute time increments
	mb = 60 /time_scale # scaled by 5 to convert to 5 minute time increments
	s = 0.0072 *time_scale # scaled by 5 to convert to 5 minute time increments
	Vmax = 50 *time_scale #150 *time_scale # scaled by 5 to convert to 5 minute time increments

	# States struct (optimization variables): glucose and insulin at current time step and delays
	G_t = model.set_variable(var_type='_x', var_name='G_t', shape=(1,1))
	n_G_delays = nG #int(floor(5/time_scale))
	G_tau = model.set_variable(var_type='_x', var_name='G_tau', shape=(n_G_delays,1))

	I_t = model.set_variable(var_type='_x', var_name='I_t', shape=(1,1))
	n_I_delays = nI #int(floor(15/time_scale))
	I_tau = model.set_variable(var_type='_x', var_name='I_tau', shape=(n_I_delays,1))

	n_Idose_delays = nId #int(floor(20/time_scale))
	I_load = model.set_variable(var_type='_x', var_name='I_load', shape=(n_Idose_delays,1))

	n_Gdose_delays = nGd #int(floor(15/time_scale))
	G_load = model.set_variable(var_type='_x', var_name='G_load', shape=(n_Gdose_delays,1))

	# Inputs struct (operated variables): glucose infusion (meals or dual hormone), insulin infusion (CSII)
	G_u = model.set_variable(var_type='_u', var_name='G_u', shape=(1,1))
	I_u = model.set_variable(var_type='_u', var_name='I_u', shape=(1,1))

	# Set Measurement variables and process + measurement noise
	G_meas = model.set_meas('G_t_meas', G_t, meas_noise=True)
	I_u_meas = model.set_meas('I_u_meas', I_u, meas_noise=False)
	G_u_meas = model.set_meas('G_u_meas',G_u, meas_noise=False)


	# Set RHS
	# G_t_scale = model.set_expression(expr_name='G_tscale', expr=G_t/100)
	# I_t_scale = model.set_expression(expr_name='I_tscale', expr=I_t/100)
	model.set_rhs('G_t', G_t+G_u+f1(I_tau[n_I_delays-1],time_scale)-f2(G_t,time_scale)-GAMMA*(1+s*(m-mb))*f3(G_t,time_scale)*f4(I_t,time_scale))
	model.set_rhs('I_t', I_t+I_load[n_Idose_delays-1]+BETA*f5(G_tau[n_G_delays-1],time_scale)-Vmax*I_t/(Km+I_t)) 

	G_next = vertcat(G_t)
	for i in range(n_G_delays-1):
		G_next = vertcat(G_next,G_tau[i])
	model.set_rhs('G_tau', G_next)

	I_next = vertcat(I_t)
	for i in range(n_I_delays-1): 
		I_next = vertcat(I_next,I_tau[i])
	model.set_rhs('I_tau', I_next)

	I_load_next = vertcat(I_u)
	for i in range(n_Idose_delays-1): 
		I_load_next = vertcat(I_load_next,I_load[i])
	model.set_rhs('I_load', I_load_next)

	G_load_next = vertcat(G_u)
	for i in range(n_Gdose_delays-1): 
		G_load_next = vertcat(G_load_next,G_load[i])
	model.set_rhs('G_load', G_load_next)

	# Setup model:
	model.setup()

	return model



