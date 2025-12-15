import math
import numpy as np
import os

class LeslieModel2D:
    def __init__(self, th1=19.6, th2=23.68, th3=23.68, survival_p1=0.7, survival_p2=0.7, lower_bounds=[0, 0, 0], upper_bounds=[110, 76, 70]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.survival_p1 = survival_p1 #0.7
        self.survival_p2 = survival_p2 #0.7

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2])), self.survival_p1 * x[0], self.survival_p2 * x[1]]

class LeslieModel4D:
    def __init__(self, th1=55, th2=55, th3=55, th4=55, p1=0.8, p2=0.6, p3=0.1, lower_bounds=[0, 0, 0, 0], upper_bounds=[203, 162, 98, 10]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.p1 = p1 #0.7
        self.p2 = p2 #0.7
        self.p3 = p3

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2] + self.th4 * x[3]) * math.exp(-0.1 * (x[0] + x[1] + x[2] + x[3])), self.p1 * x[0], self.p2 * x[1], self.p3 * x[2]]
    
    
def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

if __name__ == "__main__":
    dimension = 4
    output_folder = 'Leslie_4D'
    print("Leslie Model Parameters:")
    if dimension == 2:
        leslie_model = LeslieModel2D(th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7)
        print(f"th1: {leslie_model.th1}, th2: {leslie_model.th2}, th3: {leslie_model.th3}")
    elif dimension == 4:
        leslie_model = LeslieModel4D()
        print(f"th1: {leslie_model.th1}, th2: {leslie_model.th2}, th3: {leslie_model.th3}, th4: {leslie_model.th4}")

    print('Lower bounds: ', leslie_model.lower_bounds)
    print('Upper bounds: ', leslie_model.upper_bounds)
    
    n_samples_total = 500
    n_iterations = 40#10
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
                        initial_conditions = np.concatenate([initial_conditions, np.asarray(point).reshape(1, dimension)], axis=0)
                # print('init condition shape ', initial_conditions.shape)
            # initial_conditions = np.append(initial_conditions, np.asarray(point), axis=0)
        #   print(f"f({point}) = {result}")
      #  print('results: ', np.asarray(results))
        results = np.asarray(results)

        data2 = np.hstack((np.asarray(X), np.asarray(Y)))

        print('max in last dim: ', np.max(data2[:, 3]))

        data = np.hstack((initial_conditions, results))

        # save 80% of the data to train

        if not os.path.exists(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}"):
            os.makedirs(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}")
        
       # np.savetxt(f"data/Leslie/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/{str}.csv", data, delimiter=",", header = "x0,x1,x2,y0,y1,y2", comments="", fmt="%.8f")
        if dimension == 3:
            header = "x0,x1,x2,y0,y1,y2"
        elif dimension == 4:
            header = "x0,x1,x2,x3,y0,y1,y2,y3"
        
        if dimension == 3:
            np.savetxt(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/2{str}.csv", data2, delimiter=",", header = header, comments="", fmt="%.8f")
            if str == 'train':
                np.savetxt(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}/all_pts.csv", np.asarray(all_pts), delimiter=",", header = "x0,x1,x2", comments="", fmt="%.8f")
        elif dimension == 4:
            np.savetxt(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}_{leslie_model.th4}/2{str}.csv", data2, delimiter=",", header = header, comments="", fmt="%.8f")
            if str == 'train':
                np.savetxt(f"data/{output_folder}/{leslie_model.th1}_{leslie_model.th2}_{leslie_model.th3}_{leslie_model.th4}/all_pts.csv", np.asarray(all_pts), delimiter=",", header = "x0,x1,x2,x3", comments="", fmt="%.8f")
    


    
