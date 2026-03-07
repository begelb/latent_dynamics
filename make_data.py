import math
import numpy as np
import os
from src.true_dynamics_models import LeslieModel3D, LeslieModel4D, RedCoralModel, LeslieContraction
from src.config import Config
import argparse
import json
from scipy.stats import qmc

def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

def generate_header(n):
    x_parts = [f"x{i}" for i in range(n)]
    y_parts = [f"y{i}" for i in range(n)]
    
    return ",".join(x_parts + y_parts)

def sample_data(t_str, model, n_samples, n_iterations, skip, config, sampling_method):
    if sampling_method == 'uniform':
        initial_conditions = sample_random_pts(model.lower_bounds, model.upper_bounds, n_samples)

    elif sampling_method == 'sobol':
        if t_str == 'train':
            sampler = qmc.Sobol(d=len(model.lower_bounds), scramble=True, seed=42)
        elif t_str == 'test':
            sampler = qmc.Sobol(d=len(model.lower_bounds), scramble=True, seed=9999)
        
        raw_samples = sampler.random(n=n_samples)
        
        initial_conditions = qmc.scale(raw_samples, model.lower_bounds, model.upper_bounds)
        
    X = []
    Y = []
    for point in initial_conditions:
        for iteration in range(n_iterations):
            result = model.f(point)
            if iteration >= skip:
                X.append(point)
                Y.append(result)
            point = result

    data = np.hstack((np.asarray(X), np.asarray(Y)))

    dimension = len(model.lower_bounds)
    header = generate_header(dimension)

    data_dir = config.data_dir

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    np.savetxt(os.path.join(data_dir, t_str + '.csv'), data, delimiter=",", header = header, comments="", fmt="%.8f")
    save_metadata(t_str, model, n_samples, n_iterations, skip, config, sampling_method)

def save_metadata(t_str, model, n_samples, n_iterations, skip, config, sampling_method):
    metadata = {
        "dataset_name": t_str,
        "system": config.system,
        "dimension": len(model.lower_bounds),
        "n_samples": n_samples,
        "n_iterations": n_iterations,
        "skip_initial_steps": skip,
        "lower_bounds": list(model.lower_bounds),
        "upper_bounds": list(model.upper_bounds),
        "sampling_method": sampling_method,
        "model_params": getattr(model, 'params', {}), # Captures th1, th2, etc. if available
    }

    meta_path = os.path.join(config.data_dir, f"{t_str}_metadata.json")
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Metadata saved to {meta_path}")

# Example usage inside your main() or sample_data():
# save_metadata('train', model, n_samples_train, n_iterations, skip, config)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir',help='Directory of config files',type=str,default='config/')
    parser.add_argument('--config',help='Config file inside config_dir',type=str,default='coral.txt')
    parser.add_argument('--verbose',help='Print training output',action='store_true',default=True)

    args = parser.parse_args()
    config_fname = args.config_dir + args.config

    config = Config(config_fname)

    output_folder = config.output_dir
    system = config.system

    ''' TO DO: Add these parameters to the config files '''

    if system == 'coral':
        model = RedCoralModel()
        n_samples_total = 1000
        n_iterations = 20
        skip = 0
        n_samples_train = 100
        n_samples_test = 100
    
    elif system == 'leslie3d':
        model = LeslieModel3D(th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7)
        n_samples_total = 4000#5000
        n_iterations = 30#40
        skip = 10

    elif system == 'leslie_contraction':
        model = LeslieContraction()
        n_samples_total = 1000
        n_iterations = 10
        skip = 0

    print('Lower bounds: ', model.lower_bounds)
    print('Upper bounds: ', model.upper_bounds)

    sample_data('train', model, n_samples_train, n_iterations, skip, config, 'sobol')
    sample_data('test', model, n_samples_test, n_iterations, skip, config, 'sobol')

if __name__ == "__main__":
    main()