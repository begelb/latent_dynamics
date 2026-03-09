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