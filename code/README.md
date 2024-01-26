Code Setup

```main_graphics.py```: driver file to run a simulation and output animated graphic.

This first file sets the global project parameters for a given simulation run:
- timescale: set to 1 or 5 to indicate 1 minute or 5 minute time scale. Fine timescales will require more computation time but result in a smoother simulation. All system parameters will scale to the specified time scale so that parameters defined in $min^{-1}$ units will not break.
- HOURS: the number of hours to run the simulation for
- g, i: these are the unit conversions for glucose and insulin. g converts $mg/dL$ inputs to $mg$ which is used in the model. i converts $\mu U/mL$ to $mU$ which is used in the model. <br />
	=> One Unit of insulin is the biological equivalent of 34.7 $\mu g$ of the pure insulin molecule. This was determined in the early 20th century as the amount of pure insulin required to bring a rabbit's blood glucose down 2.5 $mmol/L$. This is a $\textit{bioefficiency}$ measure not a $\textit{mass}$ measure so different types of insulin will have a different amounts that constitute one unit. This is accounted for by insulin producers by standardizing to Units. Producers sell vials of insulin that contain U-100 insulin, which means the insulin is in a concentration of 100 Units per $mL$. This corresponds to a different number of micrograms per $mL$ for different insulin types and manufacturers. Long story short, insulin vials contain 100 Units per $mL$ of solution. Dosing is done in terms of Units which are then measured out using $mL$ - so to give one Unit of insulin you should inject 0.01 $mL$ of U-100 insulin.  <br />
	=> The model variables are specified in mass terms ($mg$ for glucose and $mU$ for insulin), but the user-inputs and graphics should be in terms of concentrations ($mg/dL$ for blood-glucose and $\mu U/mL$ for plasma-insulin). <br />
- The model, simulator, controller, and estimator modules are then initialized and initial conditions are set. <br />
- Then the closed-loop simulation is run. Optionally, meal times and amounts are specified as un-observed changes to the glucose infusion. <br />
- Noise is simulated as a normal random draw scaled by 100 (stdev scales by 10).  <br />
- Lastly, the graphics module is initialized, plots are drawn, and the animation is saved out to a file. <br />


```model.py``` : Defines the G-I dynamics developed by Li, Sturis, Tolic etc. 
This file sets the constant model parameters, specifies the model variables and variable parameters, defines scaling constants and wraps the model in a callable template. 

Each time the model needs to be instantiated, the `template_model()` function can be called. This is done in the main_graphics.py file.

```simulator.py```:

```estimator.py```:

```controller.py```:

```main_exe.py```: specifies a time history at a given point and runs a single step of the controller. designed for implementation. 