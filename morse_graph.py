import torch
import CMGDB
from functools import partial
import os
from src.models import *
import joblib
import numpy as np
from src.config import Config
import argparse
import time

@torch.no_grad()
def g_base(x, dynamics_model, device):

    x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
    output = dynamics_model(x_tensor).cpu().numpy()

    return output


if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='Leslie.txt')
    parser.add_argument('--init',help='Initial subdivisions',type=int,default=0)
    parser.add_argument('--smin',help='Min subdivisions',type=int,default=10)
    parser.add_argument('--smax',help='Min subdivisions',type=int,default=12)

    args = parser.parse_args()
    subdiv_min = args.smin
    subdiv_max = args.smax
    subdiv_init = args.init

    config_fname = args.config_dir + args.config

    config = Config(config_fname)
    
    num_pts = config.num_pts
    ex_index = config.ex_index
   # base_output_dir = config.base_output_dir 
    output_dir = config.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  base_output_dir = f'output/Leslie/23.5_23.5/{num_pts}'
    model_dir = config.model_dir
    model_path = os.path.join(model_dir, 'dynamics.pt')
    dynamics_model = torch.load(model_path)#LatentDynamics(config)
    encoder_path = os.path.join(model_dir, 'encoder.pt')
    encoder = torch.load(encoder_path)#Encoder(config)
    dynamics_model.to(device)
    dynamics_model.eval()
    encoder.to(device)
    encoder.eval()

    # evaluate the dynamics model on the training data
    # take the min and max in each dimension

   # base_data_dir = f'data/Leslie/28.9_29.8_22.0'
    base_data_dir = config.data_dir
    train_data_path = os.path.join(base_data_dir, '2train.csv')
    test_data_path = os.path.join(base_data_dir, '2test.csv')
    train_data = np.loadtxt(train_data_path, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_data_path, delimiter=',', skiprows=1)

    # base_output_dir = config.base_output_dir 
    # x_scaler_path = os.path.join(base_output_dir, 'scalers/x_scaler.gz')
    # y_scaler_path = os.path.join(base_output_dir, 'scalers/y_scaler.gz')
    scaler_dir = config.scaler_dir
    scaler_path = os.path.join(scaler_dir, 'scaler.gz')

   # x_scaler = joblib.load(x_scaler_path)
   # y_scaler = joblib.load(y_scaler_path)
    scaler = joblib.load(scaler_path)

    high_dims = config.high_dims
    x_train = train_data[:, :high_dims]
    #print('x_train shape:', x_train.shape)
    y_train = train_data[:, high_dims:]
    x_test = test_data[:, :high_dims]
    y_test = test_data[:, high_dims:]

    # x_train_scaled = x_scaler.transform(x_train)
    # y_train_scaled = y_scaler.transform(y_train)
    # x_test_scaled = x_scaler.transform(x_test)
    # y_test_scaled = y_scaler.transform(y_test)

    x_train_scaled = scaler.transform(x_train)
    y_train_scaled = scaler.transform(y_train)
    x_test_scaled = scaler.transform(x_test)
    y_test_scaled = scaler.transform(y_test)

    all_data = np.vstack((x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled))
    
    # apply the model to all data

    all_data_tensor = torch.as_tensor(all_data, dtype=torch.float32, device=device)
    with torch.no_grad():
        all_data_latent = encoder(all_data_tensor).cpu().numpy()
    lower_bounds_x = np.min(all_data_latent[:,0])
    upper_bounds_x = np.max(all_data_latent[:,0])
    lower_bounds_y = np.min(all_data_latent[:,1])
    upper_bounds_y = np.max(all_data_latent[:,1])
    
    g = partial(g_base, dynamics_model=dynamics_model, device=device)

    def G(rect):
        return CMGDB.BoxMap(g, rect, padding=True)

    lower_bounds = [lower_bounds_x, lower_bounds_y]
    upper_bounds = [upper_bounds_x, upper_bounds_y]

    subdiv_limit = config.subdiv_limit

    model = CMGDB.Model(subdiv_min, subdiv_max, subdiv_init, subdiv_limit, lower_bounds, upper_bounds, G)

    morse_graph, map_graph = CMGDB.ComputeConleyMorseGraph(model)
    
    MG_dir = os.path.join(output_dir, 'MG')

    morse_graph_plot = CMGDB.PlotMorseGraph(morse_graph)
    morse_graph_plot.render(os.path.join(MG_dir, 'morse_graph'), format='png', view=False, cleanup=False)

    morse_sets_plot = CMGDB.PlotMorseSets(morse_graph, xlim=[lower_bounds[0], upper_bounds[0]], ylim=[lower_bounds[1], upper_bounds[1]], fig_fname=os.path.join(MG_dir, 'morse_sets'))

    end_time = time.perf_counter()
    duration_mins = round((end_time - start_time)//60)

    filename = os.path.join(config.output_dir, 'mg_params_log.txt')
    with open(filename, "w") as f:
        f.write("--- MG Computation Parameters ---\n")
        f.write(f"Lower bounds: {lower_bounds}\n")
        f.write(f"Upper bounds: {upper_bounds}\n")
        f.write("------------------------------\n")
        f.write(f"Subdivision init: {subdiv_init}\n")
        f.write(f"Subdivision min: {subdiv_min}\n")
        f.write(f"Subdivision max: {subdiv_max}\n")
        f.write(f"Subdivision limit: {subdiv_limit}\n")
        f.write("------------------------------\n")
        f.write(f"Program duration: {duration_mins} minutes")