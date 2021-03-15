# Modelling Flapping Wing Aerodynamics with PR-SSM
This is the companion code for the state-space modelling of flapping wing aerodynamics with Probabilistic Recurrent State-Space Models (PR-SSM) (Doerr et al., ICML 2018). The results of this study is reported in the manuscript "State-space aerodynamic model reveals high force control authority and predictability in flapping flight", and can be found in (link). Please cite the above paper when reporting, reproducing or extending the results. The majority of the code for modelling the aerodynamics with PR-SSM was modified from the source code (**https://github.com/boschresearch/PR-SSM**).

## Prerequesites
The PR-SSM code implemented here depends on Tensorflow 1.14 (Tensorflow-gpu is prefered). The data sets used in this study is available at **doi:10.17632/cjj6y9ryms.1**.

To use the code or replicate the results in the manuscript the data sets must be placed in the correct paths. The input files (*flapping_wing_aerodynamics.mat* and *flapping_wing_aerodynamics_lasso.mat*) should be stored in *./Data/Input/* folder. 

The PR-SSM code will generate the new model logs and the data files in the *./Data/Output/**dir_name*** (replace ***dir_name*** with the desired directory name).

The results and the model logs for the model used in the manuscript can be found in the *./Data/Output/20_08_15_IN_LK7Normalized_OUT_FMwoFa5_Dimx12_Epochs1000_512Train/* directory. To perform the postprocessing code provided in *./Matlab_Postprocess/* path, the output data set *predict_train_n_test.mat* should be stored in the */matfiles/* output directory.

