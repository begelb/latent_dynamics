import math
import numpy as np
import os

''' TO DO: Move models into separate file '''

class LeslieModel3D:
    def __init__(self, th1=19.6, th2=23.68, th3=23.68, survival_p1=0.7, survival_p2=0.7, lower_bounds=[0, 0, 0], upper_bounds=[220, 154, 108]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.survival_p1 = survival_p1 
        self.survival_p2 = survival_p2 

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2]) * math.exp(-0.1 * (x[0] + x[1] + x[2])), self.survival_p1 * x[0], self.survival_p2 * x[1]]

class LeslieModel4D:
    def __init__(self, th1=80, th2=80, th3=80, th4=80, p1=0.5, p2=0.7, p3=0.7, lower_bounds=[0, 0, 0, 0], upper_bounds=[295, 148, 104, 73]):
        self.th1 = th1
        self.th2 = th2
        self.th3 = th3
        self.th4 = th4
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.p1 = p1 
        self.p2 = p2 
        self.p3 = p3

    def f(self, x):
        return [(self.th1 * x[0] + self.th2 * x[1] + self.th3 * x[2] + self.th4 * x[3]) * math.exp(-0.1 * (x[0] + x[1] + x[2] + x[3])), self.p1 * x[0], self.p2 * x[1], self.p3 * x[2]]


class RedCoralModel:
    def __init__(self, 
                 b=None, 
                 survival_rates=None, 
                 surface_area=36, 
                 lower_bounds=None, 
                 upper_bounds=None):
        """
        Initialize the Red Coral Population Model.
        
        :param b: List of 13 birth rates (reproductive coefficients).
        :param survival_rates: List of 12 survival probabilities (class i to i+1).
        :param surface_area: Surface area for density calculation (default 36 cm^2).
        """
        # Default birth rates from red_coral.py
        self.b = b if b is not None else [
            0, 0, 2.89, 10.03, 21.59, 39.02, 56.41, 77.72, 103.23, 131.87, 164.57, 201.46, 242.65
        ]
        
        # Default survival rates from red_coral.py
        self.survival_rates = survival_rates if survival_rates is not None else [
            0.889, 0.633, 0.697, 0.517, 0.437, 0.287, 0.571, 0.333, 0.75, 1, 0.333, 1
        ]
        
        self.surface_area = surface_area
        
        # Default bounds based on observed stable equilibrium capacity
        self.lower_bounds = lower_bounds if lower_bounds is not None else [0.0] * 13
     #   self.upper_bounds = upper_bounds if upper_bounds is not None else [1000] * 13
        self.upper_bounds = upper_bounds if upper_bounds is not None else [
            1300, 1150, 750, 520, 270, 120, 35, 20, 7, 5, 5, 2, 2
        ]
        self.dim = 13

    def f(self, x):
        """
        Transition function: calculates the population state at t+1 given state x at t.
        """
        # Calculate adult population density (excluding the first class of recruits)
        # rho = (Total Population - Recruits) / Surface Area
        pop_density = (sum(x) - x[0]) / self.surface_area
        
        # Density-dependent larval survival function L(rho)
        # As density increases, competition reduces the survival of new larvae.
        larval_survival = 2.94 / (pop_density + 520 * math.exp(-0.14 * pop_density))
        
        # Calculate Class 1 (Recruits): 
        # Number of larvae produced by all classes * probability of survival/settlement
        x1_next = larval_survival * sum(x[i] * self.b[i] for i in range(len(x)))
        
        # Calculate Classes 2-13:
        # Simple survival transitions: x_{i+1}(t+1) = x_i(t) * survival_rate_i
        x_rest_next = [x[i] * self.survival_rates[i] for i in range(len(x) - 1)]
       # x_rest_next = [x[i-1] * self.survival_rates[i-1] for i in range(1, self.dim)]
        
        # Combine into the next state vector
        return [x1_next] + x_rest_next

import matplotlib.pyplot as plt

def plot_trajectory_endpoints(data, n_samples, n_iterations, skip, dimension, title_suffix=""):
    """
    Plots the first two coordinates (y0, y1) of the last step of each trajectory
    to visualize where they converge.
    """
    # Number of data points per trajectory
    n_per_traj = n_iterations - skip
    
    # Indices of the very last step for each initial condition
    # For each trajectory i, the last index is (i+1) * n_per_traj - 1
    last_step_indices = [(i + 1) * n_per_traj - 1 for i in range(n_samples)]
    
    # Extract the 'next state' coordinates (y0 and y1) for these indices
    # y0 is at column index 'dimension', y1 is at 'dimension + 1'
    y0_endpoints = data[last_step_indices, dimension]
    y1_endpoints = data[last_step_indices, dimension + 1]
    
    # Count unique endpoints to estimate number of attractors
    # (Rounded to 4 decimal places to handle floating point noise)
    unique_points = set(zip(np.round(y0_endpoints, 4), np.round(y1_endpoints, 4)))
    
    plt.figure(figsize=(10, 7))
    plt.scatter(y0_endpoints, y1_endpoints, alpha=0.6, s=15, c='crimson', edgecolors='none')
    
    plt.xlabel('$x_0$ (First Coordinate)')
    plt.ylabel('$x_1$ (Second Coordinate)')
    plt.title(f'Trajectory Endpoints - {title_suffix}\nEstimated Unique Attractors: {len(unique_points)}')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot
    filename = f"attractors_{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    print(f"Number of unique final points found: {len(unique_points)}")

def check_zero_convergence(data, n_samples, n_iterations, skip, threshold=100):
    """
    Counts how many trajectories end with a total population sum < threshold.
    """
    # 1. Determine how many data rows belong to each trajectory
    n_per_traj = n_iterations - skip
    
    # 2. Get the index of the final 'Y' state for each trajectory
    last_step_indices = [(i + 1) * n_per_traj - 1 for i in range(n_samples)]
    
    # 3. Extract the final states (the 'Y' part of your data2 array)
    # Your data2 is [X, Y], so Y starts at index 'dimension'
    dimension = int(data.shape[1] / 2)
    final_states = data[last_step_indices, dimension:]
    
    # 4. Calculate the sum of all population classes for the final state
    final_sums = np.sum(final_states, axis=1)
    
    # 5. Count how many are below the threshold
    close_to_zero_mask = final_sums < threshold
    num_collapsed = np.sum(close_to_zero_mask)
    percentage = (num_collapsed / n_samples) * 100

    print(f"--- Convergence Analysis (Threshold: {threshold}) ---")
    print(f"Trajectories ending close to zero: {num_collapsed} / {n_samples}")
    print(f"Percentage of collapse: {percentage:.2f}%")
    
    return num_collapsed

# Example Usage:
model = RedCoralModel()
initial_state = [1, 2, 3] + [10] * 2 + [8, 8, 9, 6, 177, 100, 200, 300]
next_state = model.f(initial_state)
print(next_state)

def sample_random_pts(lower_bounds, upper_bounds, n):
    return np.random.uniform(np.asarray(lower_bounds), np.asarray(upper_bounds), (n, len(lower_bounds)))

def generate_header(n):
    """
    Generates a header string: x0,x1,...,xn-1,y0,y1,...,yn-1
    """
    # Create the x list and y list using f-strings
    x_parts = [f"x{i}" for i in range(n)]
    y_parts = [f"y{i}" for i in range(n)]
    
    # Combine both lists and join with commas
    return ",".join(x_parts + y_parts)

if __name__ == "__main__":
    system = 'coral'
    if system == 'coral':
        output_folder = 'coral'
        dimension = 13
        model = RedCoralModel()

    if system == 'leslie':
        dimension = 3
        if dimension == 4:
            output_folder = 'Leslie_4D'
        elif dimension == 3:
            output_folder = 'Leslie_3D_larger_domain_tail_only'
        print("Leslie Model Parameters:")
        if dimension == 3:
            model = LeslieModel3D(th1=28.9, th2=29.8, th3=22.0, survival_p1=0.7, survival_p2=0.7)
            print(f"th1: {model.th1}, th2: {model.th2}, th3: {model.th3}")
        elif dimension == 4:
            model = LeslieModel4D()
            print(f"th1: {model.th1}, th2: {model.th2}, th3: {model.th3}, th4: {model.th4}")

    print('Lower bounds: ', model.lower_bounds)
    print('Upper bounds: ', model.upper_bounds)
    
    if system == 'leslie':
        n_samples_total = 4000#5000
        n_iterations = 30#40
        skip = 10

    elif system == 'coral':
        n_samples_total = 1000
        n_iterations = 20
        skip = 0

    for str in ['train', 'test']:

        if str == 'train':
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

        data2 = np.hstack((np.asarray(X), np.asarray(Y)))

        print('max in dim 0: ', np.max(data2[:, 0]))
        print('index: ', np.argmax(data2[:, 0]))

        print('---')

        print('max in dim 1: ', np.max(data2[:, 1]))
        print('index: ', np.argmax(data2[:, 1]))

        print('---')

        print('max in dim 2: ', np.max(data2[:, 2]))
        print('index: ', np.argmax(data2[:, 2]))

        print('---')
        for k in range(13):
            print(np.mean(data2[:, k]))
        
        if system == 'leslie':
            if dimension == 3:
                header = "x0,x1,x2,y0,y1,y2"
                if not os.path.exists(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}"):
                    os.makedirs(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}")
            elif dimension == 4:
                header = "x0,x1,x2,x3,y0,y1,y2,y3"
                if not os.path.exists(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}_{model.th4}"):
                    os.makedirs(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}_{model.th4}")
            
            if dimension == 3:
                np.savetxt(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}/2{str}.csv", data2, delimiter=",", header = header, comments="", fmt="%.8f")

            elif dimension == 4:
                np.savetxt(f"data/{output_folder}/{model.th1}_{model.th2}_{model.th3}_{model.th4}/2{str}.csv", data2, delimiter=",", header = header, comments="", fmt="%.8f")

        elif system == 'coral':
            header = generate_header(13)
            if not os.path.exists(f"data/{output_folder}/data"):
                    os.makedirs(f"data/{output_folder}/data")
            np.savetxt(f"data/{output_folder}/data/2{str}.csv", data2, delimiter=",", header = header, comments="", fmt="%.8f")

        plot_trajectory_endpoints(data2, n_samples, n_iterations, skip, dimension, title_suffix=f"{system} {str}")
        check_zero_convergence(data2, n_samples, n_iterations, skip, threshold=100)




    
