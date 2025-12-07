import math
import numpy as np
import os

class LeslieModel:
    def __init__(self, th1=19.6, th2=23.68, th3=23.68, survival_p1=0.8, survival_p2=0.6, lower_bounds=[0, 0, 0], upper_bounds=[90.0, 70.0, 70.0]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.survival_p1 = survival_p1 #0.7
        self.survival_p2 = survival_p2 #0.7

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2])), self.survival_p1 * x[0], self.survival_p2 * x[1]]
    
    
def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

if __name__ == "__main__":
    leslie_model = LeslieModel(th1=29, th2=29, th3=29, survival_p1=0.8, survival_p2=0.6)
    print("Leslie Model Parameters:")
    print(f"th1: {leslie_model.th1}, th2: {leslie_model.th2}, th3: {leslie_model.th3}")
    
    n_samples_total = 1000
    n_iterations = 20#10
    skip = 0

    for str in ['train', 'test']:

        if str == 'train':
            n_samples = int(0.8 * n_samples_total)
        else:
            n_samples = int(0.2 * n_samples_total)

        initial_conditions = sample_random_pts(leslie_model.lower_bounds, leslie_model.upper_bounds, n_samples)
        
        X = []
        results = []
        Y = []
        all_pts = []
        for point in initial_conditions:
            for iteration in range(n_iterations):
                if iteration > skip:
                    result = leslie_model.f(point)
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
                        initial_conditions = np.concatenate([initial_conditions, np.asarray(point).reshape(1, 3)], axis=0)
                # print('init condition shape ', initial_conditions.shape)
            # initial_conditions = np.append(initial_conditions, np.asarray(point), axis=0)
        #   print(f"f({point}) = {result}")
        print('results: ', np.asarray(results))
        results = np.asarray(results)

        data2 = np.hstack((np.asarray(X), np.asarray(Y)))

        data = np.hstack((initial_conditions, results))

        # save 80% of the data to train

        if not os.path.exists(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}"):
            os.makedirs(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}")
        
       # np.savetxt(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/{str}.csv", data, delimiter=",", header = "x0,x1,x2,y0,y1,y2", comments="", fmt="%.8f")
        np.savetxt(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/2{str}.csv", data2, delimiter=",", header = "x0,x1,x2,y0,y1,y2", comments="", fmt="%.8f")
        if str == 'train':
            np.savetxt(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/all_pts.csv", np.asarray(all_pts), delimiter=",", header = "x0,x1,x2", comments="", fmt="%.8f")
        
        print(data)


    
