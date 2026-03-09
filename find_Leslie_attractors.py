import numpy as np
import os
import pickle

def find_leslie_attractors(th1, th2, th3, s1=0.7, s2=0.7, 
                           lower_bounds=[0, 0, 0], 
                           upper_bounds=[220, 154, 108], 
                           num_seeds=10000, iterations=100000, tol=1e-9):
    """
    Identifies unique periodic attractors by iterating from random starting points
    within a specified 3D bounding box.
    """
    def step(x):
        x0, x1, x2 = x
        sum_x = np.sum(x)
        next_x0 = (th1*x0 + th2*x1 + th3*x2) * np.exp(-0.1 * sum_x)
        next_x1 = s1 * x0
        next_x2 = s2 * x1
        return np.array([next_x0, next_x1, next_x2])

    attractors = []

    # ex_index = 3
    # plot_data_dir = f'output/Leslie_3D_larger_domain_tail_only1/{ex_index}/models'


    # with open(os.path.join(plot_data_dir, 'preimage_plot_data.pkl'), 'rb') as f:
    #     data = pickle.load(f)


    # x_list, y_list, z_list, pt_colors = data['x'], data['y'], data['z'], data['colors']

    # print('C: ', len(set(pt_colors)))

    # x_arr = np.array(x_list)
    # y_arr = np.array(y_list)
    # z_arr = np.array(z_list)
    # colors_arr = np.array(pt_colors)

    # M4_indices = np.where(colors_arr == '#008080')[0]

    # x_final = x_arr[M4_indices]
    # y_final = y_arr[M4_indices]
    # z_final = z_arr[M4_indices]

    
    # seeds_from_final = np.column_stack((x_final, y_final, z_final))

    with open('output/Leslie_3D_larger_domain_tail_only1/3/preimage_samples_k4_20pts.pkl', 'rb') as f:
        points = pickle.load(f)
  #  print(len(seeds_from_final))
    # Generate random initial seeds across the user-provided bounds
   # seeds = np.random.uniform(low=lower_bounds, high=upper_bounds, size=(num_seeds, 3))
    
    seeds_from_final = points#[point]
    print(seeds_from_final)
    #for start_pt in seeds:
    for start_pt in seeds_from_final:
        curr = start_pt
        # Burn-in phase
        for _ in range(iterations - 100):
            curr = step(curr)
        
        # Collection phase
        orbit = []
        for _ in range(100):
            curr = step(curr)
            orbit.append(curr)
            
        # Identify unique points in the cycle
        unique_pts = []
        for pt in orbit:
            if not any(np.allclose(pt, u, atol=tol) for u in unique_pts):
                unique_pts.append(pt)
        
        # Ignore the trivial attractor [0,0,0] if found
        if np.allclose(unique_pts[0], [0, 0, 0], atol=tol):
            continue

        # Check for uniqueness among found attractors
        is_new = True
        for existing in attractors:
            if len(existing) == len(unique_pts):
                matches = sum(any(np.allclose(p, e, atol=tol) for e in existing) for p in unique_pts)
                if matches == len(unique_pts):
                    is_new = False
                    break
        
        if is_new:
            attractors.append(unique_pts)
            
    return attractors

# Parameters for experiment index 6
params = {'th1': 28.9, 'th2': 29.8, 'th3': 22.0}
lower = [0, 0, 0]
upper = [220, 154, 108]

found_attractors = find_leslie_attractors(**params, lower_bounds=lower, upper_bounds=upper)

print(f"Detected {len(found_attractors)} non-trivial attractors in the range {lower} to {upper}:")
for i, attr in enumerate(found_attractors):
    print(f"Attractor {i} (Period {len(attr)}):")
    for pt in attr:
        print(f"  {pt}")