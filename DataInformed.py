import os
import json
import time
from database.data_manager import FlappingWingAerodynamics
from training.trainer import Trainer
from outputs.outputs import Outputs
from model.prssm import PRSSM

# model
model_sel = PRSSM

# dataset
ds_sel = FlappingWingAerodynamics  # set to your new dataset class - create your own
seq_len = 470  # length of sub-trajectories for training
lik_seq_length_factor = 0.8
predict_len = int(seq_len * lik_seq_length_factor)
seq_stride = seq_len    # distance between two sub-trajectories, in this application sub-trajectories are completely
# separate

# directories
parent_dir = os.path.dirname(os.getcwd())
in_dir = parent_dir + '/Data/Input/'
out_dir = parent_dir + '/Data/Output/' + 'Output_folder/'  # Replace the folder name with the desired one
model_dir = out_dir[:]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# training settings
num_gpus = 1  # (for multi-GPU //// not implemented in this version needs improvement)
gpus = ['/device:GPU:' + str(x) for x in range(num_gpus)]
epochs = 5  # number of epochs for training (make sure in training plot afterwards that converged)
test_data = True

# config
answer = input('Start a new simulation? [Y/N]\n')
if answer.upper() in ["Y", "YES"]:
    print('Starting a new simulation. The config files are updated...')
    train = True
    retrain = False
    model_config = {
        # dataset
        'batch_size': 16,  # batch size
        'shuffle': 10000,  # shuffle buffer size
        'lik_seq_length_factor': lik_seq_length_factor,
        # method
        'dim_x': 12,  # dimensionality of latent state
        'ind_pnt_num': 100,  # number of inducing points
        'samples': 30,  # number of particles
        'learning_rate': 0.03,
        'recog_len': 60,  # 2*t' in paper, number of steps for recognition model
        'recog_model': 'zeros',
        'zeta_pos': 2.,
        'zeta_mean': 0.1 ** 2,
        'zeta_var': 0.1 ** 2,
        'var_x': 0.005 ** 2,
        'var_y': 0.05 ** 2,
        'gp_var': 0.5 ** 2,
        'gp_len': 2.,
        # computation
        'gpus': gpus,
        'epochs': epochs,
        # directories
        'in_dir': in_dir,
        'out_dir': out_dir
    }

    with open(out_dir + 'training_config.json', 'w') as fout:
        json.dump(model_config, fout)

    print('Number of epochs:' + str(epochs))
    print('Latent state dim:' + str(model_config['dim_x']))
    time.sleep(5)

elif answer.upper() in ["N", "NO"]:
    with open(out_dir + 'training_config.json', "r") as read_file:
        model_config = json.load(read_file)

    answer2 = input('Retrain? [Y/N]\n')
    if answer2.upper() in ["Y", "YES"]:
        print('Continuing an old simulation. The configuration is uploaded...')
        train = True
        retrain = True
        print('Creating new folder for the retrained folder')
        out_dir = model_config['out_dir'][:-1] + '_RT/'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        model_config['out_dir'] = out_dir
        model_config['epochs'] = epochs
        with open(out_dir + 'training_config.json', 'w') as fout:
            json.dump(model_config, fout)

        print('Number of epochs:' + str(epochs))
        print('Latent state dim:' + str(model_config['dim_x']))
        time.sleep(5)

    elif answer2.upper() in ["N", "NO"]:
        print('Predictions from an old simulation. The configuration is uploaded...')
        print('Number of epochs:' + str(epochs))
        print('Latent state dim:' + str(model_config['dim_x']))
        time.sleep(5)
        train = False
        retrain = False

model_config.update({'ds': ds_sel})

# evaluation
output_sel = Outputs  # can create new class deriving from it if need richer outputs

#
# Run
#
# load
outputs = output_sel(out_dir)
ds = ds_sel(seq_len, seq_stride, in_dir)
outputs.set_ds(ds)
model = model_sel(ds.dim_u, ds.dim_y, model_config)
outputs.set_model(model, predict_len, model_config['dim_x'])
# train
if train:
    trainer = Trainer(model, model_dir)
    trainer.train(ds, epochs, retrain=retrain, test_data=test_data)
    outputs.set_trainer(trainer)

# evaluate
outputs.create_all()
