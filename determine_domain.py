#from src.Leslie_models import LeslieModel4D
import numpy as np
import math

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

dimension = 4
n_iterations = 20
leslie_model = LeslieModel4D()


lower = [0] * dimension
init_upper = [1] * dimension
found_max_counter = 0
upper = init_upper

checked_ceil = False
took_ceiling = False

while found_max_counter < dimension or not checked_ceil:

    print('Current upper bounds: ', upper)

    initial_conditions = sample_random_pts(lower, upper, 100)
        
    X = []
    Y = []
    for point in initial_conditions:
        for iteration in range(n_iterations):
            result = leslie_model.f(point)
            Y.append(result)
            X.append(point)
            point = result

    data = np.hstack((np.asarray(X), np.asarray(Y)))

    max_per_column = data.max(axis=0)
    formatted_strings = [f"{item:.4f}" for item in max_per_column]
    max_per_column_formatted = ", ".join(formatted_strings)

    print('max per column: ', max_per_column_formatted)

    new_upper = []

    for dim in range(dimension):
        m_0 = max_per_column[dim]
        m_1 = max_per_column[dim+4]
        final_max = max(m_0, m_1)
        print('final max: ', f"{final_max:.4f}")
        if final_max > upper[dim]:
            new_upper.append(final_max)
        else:
            new_upper.append(upper[dim])
            found_max_counter += 1
            print('Found max counter: ', found_max_counter)

    upper = new_upper

    checked_ceil_per_dim = []
    if took_ceiling:
        for dim in range(dimension):
            m_0 = max_per_column[dim]
            m_1 = max_per_column[dim+4]
            final_max = max(m_0, m_1)
            print('final max: ', f"{final_max:.4f}")
            if final_max > upper[dim]:
                new_upper.append(final_max)
                checked_ceil_per_dim.append(False)
            else:
                new_upper.append(upper[dim])
                checked_ceil_per_dim.append(True)
        checked_ceil = all(checked_ceil_per_dim)
    
    if found_max_counter == 4:
        print('taking ceiling')
        took_ceiling = True
        upper_int = [math.ceil(item) for item in upper]
        upper = upper_int

print('Recommended upper bounds of initial: ', upper_int)


