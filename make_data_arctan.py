import math
import numpy as np
import os

class Bistable:
    def __init__(self, dimension):
        self.lower_bounds = [-2] * dimension
        self.upper_bounds = [2] * dimension

    def f(self, x):
            return [np.arctan(4*x[0])] + [x[i]/4 for i in range(1, len(x))]
    

def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))


if __name__ == "__main__":
    dimension = 10
    model = Bistable(dimension=dimension)
    
    n_samples_total = 100
    n_iterations = 10
    skip = 0

    for str in ['train', 'test']:

        if str == 'train':
            n_samples = int(0.8 * n_samples_total)
            print('n_samples train: ', n_samples)
        else:
            n_samples = int(0.2 * n_samples_total)
            print('n_samples test: ', n_samples)

        initial_conditions = sample_random_pts(model.lower_bounds, model.upper_bounds, n_samples)
        
        X = []
        results = []
        Y = []
        all_pts = []
        for point in initial_conditions:
            for iteration in range(n_iterations):
                if iteration >= skip:
                    result = model.f(point)
                    results.append(result)
                    Y.append(result)
                    X.append(point)
                    all_pts.append(point)
                    if iteration == n_iterations - 1:
                        #print('here')
                        all_pts.append(result)
                    point = result
                # print('point: ', point)
                # print('point as array ', np.asarray(point))
                # print('initial conditions ', initial_conditions)
                # print('shape initial conditions', initial_conditions.shape)
                # print('shape point ', np.asarray(point).shape)
            # intial_conditions = np.concatenate((initial_conditions, np.asarray(point)), axis=0)
                    if iteration < n_iterations - 1:
                        initial_conditions = np.concatenate([initial_conditions, np.asarray(point).reshape(1, dimension)], axis=0)
                # print('init condition shape ', initial_conditions.shape)
            # initial_conditions = np.append(initial_conditions, np.asarray(point), axis=0)
        #   print(f"f({point}) = {result}")
        print('results: ', np.asarray(results))
        results = np.asarray(results)

        data = np.hstack((np.asarray(X), np.asarray(Y)))

        # save 80% of the data to train

        if not os.path.exists(f"data/arctan/"):
            os.makedirs(f"data/arctan")

            # 1. Define your variables
        prefixes = ['x', 'y']  # The base names for your columns
        n = dimension                  # The number of entries for *each* prefix

        # 2. Generate the list of header names
        # This is a nested list comprehension
        header_list = [f"{p}{i}" for p in prefixes for i in range(n)]

        # 3. Join the list into a single string
        header_string = ",".join(header_list)
        
        np.savetxt(f"data/arctan/{str}.csv", data, delimiter=",", header = header_string, comments="", fmt="%.8f")

        header_list = [f"x{i}" for i in range(n)]

        # 3. Join the list into a single string
        header_string = ",".join(header_list)

        if str == 'train':
            np.savetxt(f"data/arctan/all_pts.csv", np.asarray(all_pts), delimiter=",", header = header_string, comments="", fmt="%.8f")
        
        print(data)


    
