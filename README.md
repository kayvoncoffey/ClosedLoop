Closed Loop AP Project

The goal of this project is to build a closed-loop AP (artificial pancreas) controller to automatically manage diabetes treatment. The controller should be wholly self-contained (full closed loop, no patient input), robust and patient-specific (parameters tunable to handle inter-patient and intra-patient variability), safe (avoid hypoglycemia), and effective (increase TiR). 

In particular, this project aims to 
1. Use a full compartmental accounting model of nonlinear delay differential equations to model glucose-insulin dynamics in the body, and compare results against the linearized systems used today 
2. Use a model predictive controller that is tunable to implement single and dual hormone treatment variations, and estimate the impact of adding dual hormone treatment. 
3. Develop predictive estimate of plant parameters to allow intra and inter patient specific treatment. 
	- glucose sensitivity as uncertainty parameter in robust-MPC solution
	- NN model to condition robust-MPC estimate of current insulin sensitivity 
	- model to fit/initialize patient-specific system parameters
	
This project is based on the compartmental accounting model of G-I dynamics developed by Sturis, Tolic, Li, Topp, Kissler, Bortz and many others
- Kissler/Bortz paper: https://arxiv.org/pdf/1403.0638.pdf
- Li et al paper: https://pubmed.ncbi.nlm.nih.gov/16712872/
- Sturis and Tolic: https://www.sciencedirect.com/science/article/abs/pii/S0022519300921805

The MPC controller is built in python using the do-mpc library. Documentation: https://www.do-mpc.com/en/latest/index.html.

The project is structured in the following modules:
1. The nonlinear delay differential equation model of G-I dynamics, along with all patient-specific parameters is defined in `model.py`
2. The MPC controller with single / dual hormone schemes, objective function, and constraints is built in `controller.py` 
3. The state estimator is specified in `estimator.py`.
4. A model simulator is built in `simulator.py`
5. Finally, the model, estimator, simulator, and controller are initialized, the simulation is run, and results are plotted in `main_graphics.py`
6. Two additional scripts are presented: `main.py` which runs the same simulation and plots results without animation, and `main_exe.py` which runs one single step of the controller. This is the file designed to run at each time step in an actual application. 
