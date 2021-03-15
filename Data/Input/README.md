Store the flapping wing aerodynamics data (**flapping_wing_aerodynamics.mat**) from (add_link) and the LRQSM data (**flapping_wing_aerodynamics_lasso_fit**) here

The input data file **flapping_wing_aerodynamics.mat** includes the time history of:
• [Euler angles of the wing] - **ds_pos** (stroke, deviation, rotation),
• [The kinematic variables derived from **ds_pos**] - **ds_u_raw** (7 variables, see the paper for the definitions and the order of variables),
• [Aerodynamic forces and moments] - **ds_y_raw** (5 variables, see the paper for the definitions and the order of variables),
• [The standardized versions of **ds_u_raw** and **ds_y_raw**] - **ds_u** and **ds_y** with the mean **ds_mean_u** and **ds_mean_y** and the standard deviation "ds_std_u" and **ds_std_y** vectors. Only the training data is considered when normalizing.

The input data file "flapping_wing_aerodynamics_lasso_fit.mat" includes:
• [The kinematic features derived from **ds_u_raw**] - **ds_uLR_raw** (11 variables, see the paper for the definitions and the order of variables),
• [Aerodynamic forces and moments] - **ds_y** (identical to the variable above),
• [Lasso Model] - **LRmodel** (the lasso model fitted to **ds_y** with **ds_uLR_raw** as the input)
• [Lasso Predictions] - **lr_y** (the the lasso predictions for all trajectories)
