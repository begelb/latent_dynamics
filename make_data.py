import math
import numpy as np
import os
from src.true_dynamics_models import LeslieModel3D, LeslieModel4D, RedCoralModel
from src.config import Config
import argparse

def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

def generate_header(n):
    x_parts = [f"x{i}" for i in range(n)]
    y_parts = [f"y{i}" for i in range(n)]
    
    return ",".join(x_parts + y_parts)

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

    if system == 'coral':
        model = RedCoralModel()
        n_samples_total = 1000
        n_iterations = 20
        skip = 0
    
    elif system == 'leslie3d':
        model = LeslieModel3D(th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7)
        n_samples_total = 4000#5000
        n_iterations = 30#40
        skip = 10

    dimension = len(model.lower_bounds)

    print('Lower bounds: ', model.lower_bounds)
    print('Upper bounds: ', model.upper_bounds)

    for t_str in ['train', 'test']:

        if t_str == 'train':
            n_samples = int(0.8 * n_samples_total)
        else:
            n_samples = int(0.2 * n_samples_total)

        initial_conditions = sample_random_pts(model.lower_bounds, model.upper_bounds, n_samples)
        
        X = []
        Y = []
        for point in initial_conditions:
            for iteration in range(n_iterations):
                result = model.f(point)
                if iteration >= skip:
                    Y.append(result)
                    X.append(point)
                point = result

        data = np.hstack((np.asarray(X), np.asarray(Y)))

        header = generate_header(dimension)

        data_dir = config.data_dir

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        np.savetxt(os.path.join(data_dir, t_str), data, delimiter=",", header = header, comments="", fmt="%.8f")

if __name__ == "__main__":
    main()