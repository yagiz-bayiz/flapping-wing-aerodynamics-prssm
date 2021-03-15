Store the PR-SSM predictions (predict_train_n_test.mat) here.\
• [Prediction means] - **gp_mean_training** for training (512 trajectories) and **gp_mean_test** for test (36 trajectories),\
• [Prediction variations] - **gp_var_training** for training (512 trajectories) and **gp_var_test** for test (36 trajectories),\
• [Latent state means] - **latent_mean_training** for training (512 trajectories) and **latent_mean_test** for test (36 trajectories),\
• [Latent state variations] - **latent_var_training** for training (512 trajectories) and **latent_var_test** for test (36 trajectories),\
• [Ground truth, partitioned version of "ds_y"] - **gt_training** for training (512 trajectories) and **gt_test** for test (36 trajectories),\
• [Inputs, partitioned version of "ds_u"] - **in_training** for training (512 trajectories) and **in_test** for test (36 trajectories),\
• [Corresponding positions, partitioned version of "ds_ps"] - **pos_training** for training (512 trajectories) and **pos_test** for test (36 trajectories),\
• [Mean and standard deviation vectors that denormalize the inputs] - **mean_in** (identical to **ds_mean_u** from input data) and **std_in** (identical to **ds_std_u** from input data),\
• [Mean and standard deviation vectors that denormalize the predictions and ground truth] - **mean_out** (identical to **ds_mean_y** from input data) and **std_out** (identical to **ds_std_y** from input data).
