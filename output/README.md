$\textbf{Robust MPC Artificial Pancreas System}$

Cells in the body break down glucose into ATP in order to generate energy. This glucose is absorbed from the blood stream via the hormone insulin. Diabetes Mellitus is an autoimmune disease characterized by the body's inability to produce sufficient levels of insulin. In Type 2 diabetes, the body's sensitivity to insulin is severely reduced, resulting in an insufficient supply of insulin in the blood stream. In Type 1 diabetes, the body loses its ability to produce insulin at all, resulting in no insulin in the blood stream. In both cases, the insufficient supply of insulin means cells in the body are not able to utilize the glucose in the blood stream which impairs cell function and can lead to complications including cardiovascular disease, kidney disease, nerve damage, diabetic retinopathy and many others. 

In order to model the glucose-insulin dynamics in the body, Sturis Tolic Li Bortz and many others have developed a system of nonlinear delay differential equations based on a compartmental accounting of glucose and insulin absorption and production mechanisms. The system is built as follows and depicted graphically in figure 1.  

Glucose is added to the bloodstream in two main ways  
1. Ingestion - food is consumed and, through digestion, glucose is extracted into the bloodstream.
2. Hepatic production - low levels of glucose in the bloodstream and high levels of insulin trigger the release of stored glucose in the liver.

Glucose is removed from the bloodstream in two main ways
1. Utilization by CNS tissue - cranial nervous system tissue utilizes glucose without requiring 
2. Utilization by muscle and fat tissue - muscle and fat cells use insulin to uptake glucose from the bloodstream. 

Insulin is added to the blood stream in two main ways
1. Pancreatic production - the pancreas produces insulin in response to high levels of blood glucose concentration 
2. Infusion - in diabetic patients, synthetic insulin is injected into interstitial fluid and then absorbed into the bloodstream in response to high blood glucose concentration or in anticipation of a meal

Insulin is removed from the bloodstream in one main ways
1. Clearance - insulin in the bloodstream is metabolized by the presence of human insulin degrading enzyme (IDE) 

$\textbf{Figure 1}$  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output/GI_dynamics.png" width="50%" height="50%">

These sources and sinks of glucose and insulin are modeled using the following set of nonlinear delay differential equations.

$\dot{G} = G_{in} + f_1(I(t-\tau_1)) - f_2(G(t)) - \gamma[1+s(m-m_b)]f_3(G(t))f_4(I(t)) $  
$\dot{I} = I_{in} + \beta f_5(G(t-\tau_2)) - \frac{V_{max}I(t)}{K_m+I(t)}$

Where the equations $$f_i, i=1,2,3,4$$ along with all corresponding parameters are derived and presented in the work of Sturis, Tolic, Li, Topp, Kissler, Bortz and many others. 


A model predictive controller is developed to control this system. The schema in figure 2 shows the design.

$\textbf{Figure 2}\newline$  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output/MPC_controller.png" width="50%" height="50%">


The controller uses the full compartmental accounting model of glucose-insulin dynamics in the body, can compute a single or dual-hormone optimal solution, and uses a MHE with robust-MPC approach to model the insulin sensitivity parameter at each time step which allows intra and inter patient specific treatment.  

These properties achieve the goal of this project by creating a fully closed-loop AP that requires no patient input, is robust to patient-specific traits, and is effective in simulation at maintaining euglycemia. 

In non-diabetics, blood glucose exhibits ultradian rhythms with a period of 1-2 hours. This behavior emerges in the uncontrolled G-I system as the delay parameters undergo hopf bifurcations. In the controlled case, these oscillations are produced in both the single and dual hormone treatment cases when glucose measurement noise is introduced. When there is no noise in the glucose measurement, the controlled system resolves to a steady state unless true glucose sensitivity is driven in an oscillatory way - then the lag in glucose sensitivity estimation induces ultradian oscillations. 

$\textbf{Simulation A1}$: Shows the oscillatory ultradian rhythms that are reproduced by the controlled system when there is no measurement noise, in the single-hormone case.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/single_hormone/estimate_Isensitivity_0noise.gif" width="70%" height="70%">

$\textbf{Simulation A2}$: Shows the oscillatory ultradian rhythms that are reproduced by the controlled system when there is no measurement noise, in the dual-hormone case.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/dual_hormone/estimate_Isensitivity_0noise.gif" width="70%" height="70%">

$\textbf{Simulation A3}$: Shows the oscillatory ultradian rhythms that are reproduced by the controlled system when measurement noise (Gaussian with stdev 7) is added, in the single-hormone case.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/single_hormone/estimate_Isensitivity_50noise.gif" width="70%" height="70%">

$\textbf{Simulation A3}$: Shows the oscillatory ultradian rhythms that are reproduced by the controlled system when measurement noise (Gaussian with stdev 7) is added, in the dual-hormone case.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/dual_hormone/estimate_Isensitivity_50noise.gif" width="70%" height="70%">

The controller can also react to meal time glucose consumption. The glucose absorption profile that results from carbohydrate consumption can vary in shape, duration, and intensity depending on the nutritional content of the meal - carbohydrates from fats (think pizza) result in a delayed spike in blood sugar while simple sugars like fructose will spike blood sugar immediately. 

The body can react to different absorption profiles because it only includes two delays - the delay between elevated glucose levels and insulin entering the bloodstream and the delay between reduced glucose levels / high insulin levels and hepatic glucose production. However, the MPC controlled AP introduces 2 more delays. There is an additional 5-10 minute delay between the blood glucose value and the glucose value measured by the CGM in the plasma. There is another 15-20 minute delay between the administration of insulin into the interstitial fluid and the time it enters the blood stream. 

Ahead-of-time knowledge of the nutrient profile of the food consumed at meal time would greatly improve the controllers ability to manage blood glucose. However, the goal is to develop a treatment protocol that requires no patient intervention. This is likely the area of greatest potential advancement for this project. 

However, even without such ahead-of-time knowledge or a sophisticated estimation scheme, the controller is able to react to rising blood glucose levels resulting from meal ingestion and administer an optimal dose of insulin to recover and maintain euglycemia. 

The following images showcase the simulation of a 12 hour day including 3 meals. A small breakfast, then sizeable lunch with slow absorption profile and a large dinner with quick absorption profile are administered at 2, 5, and 7 hours into the day. The day is initialized with a morning spike in blood sugar.   

$\textbf{Figure B1}$  Shows the single-hormone case with no measurement noise.   
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/single_hormone/sim_3meal_0noise.gif" width="70%" height="70%">

$\textbf{Figure B2}$ Shows the dual-hormone case with no measurement noise.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/dual_hormone/sim_3meal_0noise.gif" width="70%" height="70%">

$\textbf{Figure B3}$ Shows the single-hormone case with measurement noise.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/single_hormone/sim_3meal_50noise.gif" width="70%" height="70%">

$\textbf{Figure B4}$: Shows the dual-hormone case with measurement noise.  
<img src="https://github.com/kayvoncoffey/ClosedLoop/blob/main/output//simulations/dual_hormone/sim_3meal_50noise.gif" width="70%" height="70%">






