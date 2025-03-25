# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:43:15 2025

@author: liqunwei
"""
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import matplotlib.pyplot as plt
from math import floor
from sklearn.svm import OneClassSVM

from scipy.stats import gamma

class system_model:
    def __init__(self,time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA):
        self.meal = True
        self.time_scale = time_scale
        self.N = N
        self.G_t = G_t
        self.G_t_past = G_t
        self.I_t_list = I_t_list
        self.n_I_delays = n_I_delays
        self.I_u_list = I_u_list
        self.n_Idose_delays = n_Idose_delays
        self.GAMMA = GAMMA
        self.pred_next_g = G_t
        self.calib = 0
        
        self.BETA = 0
        self.Km = 2300
        # self.m = 60 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        # self.mb = 60 /self.time_scale # scaled by 5 to convert to 5 minute time increments
        # self.s = 0.0072 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        # self.Vmax = 100 *self.time_scale #150 *time_scale # scaled by 5 to convert to 5 minute time increments
        self.m = 60 # scaled by 1/5 to convert to 5 minute time increments
        self.mb = 60 # scaled by 5 to convert to 5 minute time increments
        self.s = 0.0072  # scaled by 5 to convert to 5 minute time increments
        self.Vmax = 150  #150 *time_scale # scaled by 5 to convert to 5 minute time increments

    

    def f1(self,I):
        # Rg = 180 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Rg = 180
        alpha = 0.29
        Vp = 3
        C5 = 26
        x = alpha*(I/Vp - C5)
        # y = x.copy()
        # y[x>=0] = Rg*np.exp(-x[x>=0]) / (np.exp(-x[x>=0]) + 1)
        # y[x<0] = Rg/(1+np.exp(x[x<0]))
        if x>=0:
            y = Rg*np.exp(-x) / (np.exp(-x) + 1)
        else:
            y = Rg/(1+np.exp(x))
        return y
    
    def f2(self,G):
        # Ub = 72 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Ub = 72
        C2 = 144
        Vg = 10
        return Ub*(1-np.exp(-G/(C2*Vg)))
    
    def f3(self,G):
        C3 = 1000
        Vg = 10
        return G/(C3*Vg)
    
    def f4(self,I):
        # U0 = 40 *self.time_scale #scaled by 5 to convert to 5 minute time increments
        U0 = 40
        # Um = 940 *self.time_scale #scaled by 5 to convert to 5 minute time increments
        Um = 940
        beta = 1.77
        C4 = 80
        Vi = 11
        # E = 0.2 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        E = 0.2
        # ti = 100 /self.time_scale # scaled by 1/5 to convert to 5 minute time increments
        ti = 100
        return U0 + (Um-U0)/(1+np.exp(-beta*np.log(0.0000001+I/C4*(1/Vi+1/(E*ti)))));
    
    def f5(self,G):
        # Rm = 210 *self.time_scale # scaled by 5 to convert to 5 minute time increments
        Rm = 210
        C1 = 2000
        Vg = 10
        a1 = 300
        return Rm/(1+np.exp((C1-G/Vg)/a1))
    
    # iLoad is the effective infused insulin at the moment(infusion could take time)
    def iLoad(self,I_u_list):
        return I_u_list[self.n_Idose_delays-1]
    
    # iTau is the previous inlusion dose in the body that makes liver produce glucose(Tau: the process takes time)
    def iTau(self,I_t_list):
        return I_t_list[self.n_I_delays-1]
    
    def t_next(self,tau):
        #update current glucose level g_t
        self.calib = self.G_t - self.pred_next_g
        self.G_t_past = self.G_t
         # dI_t = self.iLoad(self.I_u_list)-0.06*self.I_t_list[0]*self.time_scale
         # [dI_t+self.I_t_list[0]]+self.I_t_list[:-1]
        #update injected insulin level
        GAMMA = (np.random.rand()-self.GAMMA)/self.GAMMA*0.2+self.GAMMA
        dI_t = self.iLoad(self.I_u_list)-self.time_scale*self.Vmax*self.I_t_list[0]/(self.Km+self.I_t_list[0])
        dG_t = self.f1( self.iTau(self.I_t_list))-self.f2(self.G_t)-GAMMA*(1+self.s*(self.m-self.mb))*self.f3(self.G_t)*self.f4(self.I_t_list[0])
        
        self.pred_next_g = self.G_t + self.f1( self.iTau(self.I_t_list))-self.f2(self.G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(self.G_t)*self.f4(self.I_t_list[0])
        
        self.I_t_list = np.insert(self.I_t_list, 0,  dI_t+self.I_t_list[0])[:-1]
        self.I_u_list = np.insert(self.I_u_list,0,tau[0])[:-1]
        
        self.G_t += dG_t * self.time_scale
        
    def predicted_g(self,tau,extended_len=0):
        G_N = np.zeros(self.N+extended_len)
        I_t_list = self.I_t_list
        G_t = self.G_t
        
        I_u_list = self.I_u_list
        for i in range(self.N):
            I_t = I_t_list[0]
            dG_t = self.f1( self.iTau(I_t_list))-self.f2(G_t)-self.GAMMA*(1+self.s*(self.m-self.mb))*self.f3(G_t)*self.f4(I_t)
            
            # dI_t =I_load[n_Idose_delays-1]+BETA*f5(G_tau[n_G_delays-1],time_scale)-Vmax*I_t/(Km+I_t)
            dI_t = self.iLoad(I_u_list)-self.time_scale*self.Vmax*I_t/(self.Km+I_t)
            # dI_t = self.iLoad(I_u_list)-self.time_scale*0.06*I_t
            G_t = G_t + dG_t * self.time_scale
            I_t_list = np.insert(I_t_list, 0, dI_t+I_t_list[0])[:-1]
            
            I_u_list = np.insert(I_u_list, 0,tau[i])[:-1]
            
            G_N[i] = G_t + self.calib
        
        return G_N
    def discrete_tau(self,tau):
        return tau-np.array(tau)
# Step of the system

def l_cost(tau, system_model, g, target=105, safe_high=120, safe_low=90, alert_low=60, alert_high=150):
    gt = system_model.predicted_g(tau)
    l_safe_high = np.square(gt-target*g)*(gt>safe_high*g)
    l_safe_low = np.square(gt-target*g)*(gt<safe_low*g)
    l_alert_high = np.square(gt-target*g)*(gt>alert_high*g)
    l_alert_low = np.square(gt-target*g)*( gt<alert_low*g)
    return np.sum(l_safe_high+l_safe_low+l_alert_high+l_alert_low)

def max_cost(tau, system_model,g, G_t_max=140):
    G_t_N = system_model.predicted_g(tau)
    return np.max(np.square(np.array(G_t_N)-G_t_ref*g))



def mpc_cost(tau, system_model,g, G_t_ref=105):
    G_t_N = system_model.predicted_g(tau)
    return np.sum(np.square(np.array(G_t_N)-G_t_ref*g))/len(tau)

def m_cost(tau, system_model,g, G_t_ref=105, G_low=70):
    G_t_N = system_model.predicted_g(tau)-system.calib
    mse = np.mean(np.square(np.array(G_t_N)-G_t_ref*g))
    # max = np.max(np.square(np.array(G_t_N)-G_t_ref*g))
    # G_t_N_extended = system_model.predicted_g(tau,12)-system.calib
    # extended_max = np.max(np.square(np.array(G_t_N_extended)-G_t_ref*g))
    # extended_low = np.sum(np.square(np.array(G_t_N_extended)-G_t_ref*g)*(G_t_N_extended<G_low*g))
    safe_penalty = np.sum((G_t_N > 140*g) | (G_t_N < 70*g)) * 1e4
    return mse+safe_penalty

def solve_mpc(tau_ini,N,system_model,g,G_t_ref):

    # Linear constraints on the rate of change of tau: -delta_tau_max <= tau[i+1] - tau[i] <= delta_tau_max for all i in the N - 1
    # Implemented using LinearConstraint as -delta_tau_max <= delta_tau_matrix * tau <= delta_tau_max
    delta_tau_max = 500 #200
    tau_max = 1000 #300
    delta_tau_matrix = np.eye(N) - np.eye(N, k=1)
    constraint1 = LinearConstraint(delta_tau_matrix, -delta_tau_max, delta_tau_max)

    # # # We need a constraint on the rate of change of tau[0] respect to its previous value, which is tau_ini[0]
    first_element_matrix = np.zeros([N, N])
    first_element_matrix[0, 0] = 1
    constraint2 = LinearConstraint(first_element_matrix, 0, tau_ini[0]+delta_tau_max)
    # # G_t_constraint = predicted_g(N,tau,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA, time_scale)
    # system_model = system_model(time_scale,N,G_t,I_t_list,n_I_delays,I_u_list,n_Idose_delays, GAMMA)
    G_t_constraint = NonlinearConstraint(system_model.predicted_g, 0.0, 200*g)
    # discrete_constraint = NonlinearConstraint(system_model.discrete_tau, 0.0, 0.0)

    # Add constraints
    delta_tau_constraint = [constraint1,constraint2,G_t_constraint]
    # delta_tau_constraint = [G_t_constraint]
    # Bounds --> -tau_max <= tau[idx] <= tau_max for idx = 0 to N-1
    bounds = [(0, tau_max) for idx in range(N)]

    # Starting optimisation point for theta and dtheta are the current measurements

    # Minimization
    result = minimize(m_cost, tau_ini, args=(system_model, g), bounds=bounds, constraints=delta_tau_constraint)

    # Extract the optimal control sequence
    tau_mpc = result.x

    return tau_mpc

# Cost function to be minimized


# ---------- SIMULATION INITIALISATION ----------
timescale = 5
HOURS = 24 #7am to 9pm
N_iterations = int(floor(HOURS*60/timescale))

# n_G_delays = int(floor(5/timescale)) removed for typeI
n_I_delays = int(floor(15/timescale))  #tau2
n_Idose_delays = int(floor(5/timescale)) #tau1

g=100
# Arrays for logging
tau = np.zeros(N_iterations)
# ---------- CONTROL SYSTEM CALIBRATION ----------
# Model predictive control horizon
N = 10
tau_ini = np.zeros(N)
# Array for values of theta and tau for each simulation

G_t_start = 100*g
I_t_record = np.ones(N)*50

I_u_record = np.ones(N)*50

GAMMA = 0.5
# ---------- SIMULATION ----------
system = system_model(timescale,N,G_t_start,I_t_record,n_I_delays,I_u_record,n_Idose_delays, GAMMA)
# Init time

G_t_ref=100
G_t = np.zeros(N_iterations)
G_predicted = np.zeros(N_iterations)
calib = np.zeros(N_iterations)
calib_anamoly =np.zeros(N_iterations)
moving_average = np.empty(0)
compensation = 0

bfast_start = int(floor(2*60/timescale)) #
bfast_end = int(floor(6*60/timescale))
x = np.arange(bfast_end-bfast_start)
alpha, beta = 4, 1/4
bfast = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)

lunch_start = int(floor(5*60/timescale))
lunch_end = int(floor(10*60/timescale))
x = np.arange(lunch_end-lunch_start)
alpha, beta = 4, 1/4
lunch = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)

dinner_start = int(floor(8*60/timescale))
dinner_end = int(floor(14*60/timescale))
x = np.arange(dinner_end-dinner_start)
alpha, beta = 5, 1/5
dinner = gamma.cdf(x + 1, a=alpha, scale=1/beta) - gamma.cdf(x, a=alpha, scale=1/beta)



model = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# Define the window size for the moving average
window_size = 5
meal_compensation = 200
threshold = 500

compensation_times = 0
compensation_times_thrd = 2
compensation_gap = 0
compensation_gap_sleep = 12
for idx in range(N_iterations):
    # tau_mpc = np.zeros(N)+23.5 # constant infusion test: 21 mU/min = 105mU/5min => 30U/day
    tau_mpc = solve_mpc(tau_ini,N,system,g,G_t_ref)
    # Use first element of control input optimal solution
    
    
    # Initial solution for next step = current solution
    
    if idx in (bfast_start, lunch_start, dinner_start):
        tau_mpc[0] += compensation
    # if (system.G_t < 4.5*g*18 and system.G_t-system.G_t_past <0) or system.G_t < 4*g*18:
    #     tau_mpc[0] = 0
    # if system.G_t < 4.5*g*18:
    #     tau_mpc[0] = 0
    
    
    #manual manipulation
    # if idx > 10 and idx -window_size <= 10:
    moving_average = np.append(moving_average, system.calib)
    if idx -window_size > 10:
            
            if compensation_times < compensation_times_thrd:
                if system.calib - np.mean(moving_average) > threshold:

                    calib_anamoly[idx] = 1
                    tau_mpc[0] += meal_compensation
                    compensation_times += 1
            elif compensation_times > 0:
                if compensation_gap < compensation_gap_sleep:
                    compensation_gap += 1
                else:
                    compensation_times = 0
                    compensation_gap = 0
                

            
        # elif system.calib - np.mean(moving_average) > -threshold:
        #     moving_average = np.append(moving_average, system.calib)
    
    
    
    
    tau[idx] = tau_mpc[0]
    # ---------- SIMULATION LOOP  ----------
    G_predicted[idx] = system.predicted_g(tau_mpc)[0]
    system.t_next(tau_mpc)
    
    tau_ini = tau_mpc
    if system.meal:
        # system.G_t += 1080 # full nutrition constant infusion: 216 mg/min = 1,080mg/5min => 311g/day
        if (idx>=bfast_start) & (idx<bfast_end):
            system.G_t += bfast[idx-bfast_start]*90_000 # 85g 2.5 hr acting carb meal
        if (idx>=lunch_start) & (idx<lunch_end): 
            system.G_t += lunch[idx-lunch_start]*110_000 # 80g 3 hr acting carb meal
        if (idx>=dinner_start) & (idx<dinner_end): 
            system.G_t += dinner[idx-dinner_start]*111_000 # 95g 2.5 hr acting carb meal 
    G_t[idx] = system.G_t
    
    calib[idx] = system.calib
    
   

        
    
print(sum(tau))
    
# Append result for this simulation
plt.subplot(2, 1, 1)
plt.plot(np.arange(N_iterations)*timescale/60, np.array(G_t)/g, '--', label="Glucose Level")
plt.plot(np.arange(N_iterations)*timescale/60, np.array(G_predicted)/g, '--', label="Predicted Glucose Level")
plt.ylabel(r"G_t, mg/dL")
plt.legend()
plt.axhline(70,color='green',linestyle='--',linewidth=0.75)
plt.axhline(140,color='green',linestyle='--',linewidth=0.75)
# plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    
    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    

plt.subplot(2, 1, 2)
plt.plot(np.arange(N_iterations)*timescale/60, tau, linewidth=0,color='r',marker='.',label="Insulin Infusion")
plt.ylabel(r"I_u, mU")
plt.xlabel("Time")
plt.legend()
plt.grid()
if system.meal:
    plt.axvline(bfast_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_start*timescale/60,color='red',linestyle='--',linewidth=0.75)
    
    plt.axvline(bfast_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(lunch_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.axvline(dinner_end*timescale/60,color='orange',linestyle='--',linewidth=0.75)
    plt.text(bfast_start*timescale/60+0.1, 110, 'Breakfast', rotation=90)
    plt.text(lunch_start*timescale/60+0.1, 110, 'Lunch', rotation=90)
    plt.text(dinner_start*timescale/60+0.1, 110, 'Dinner', rotation=90)
plt.show()



